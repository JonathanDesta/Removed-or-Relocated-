"""CPU-only paper revision workflow. Reads saved records; never loads a model.

Notebook cells and terminal commands call this same entry point:
    python scripts/paper_revision.py init --project /path/maheep-yksa --out /new/directory
    python scripts/paper_revision.py tests --project /path/maheep-yksa --out /new/directory
    python scripts/paper_revision.py figure1 --project /path/maheep-yksa --out /new/directory
Use --help for all stages. 'all' starts a fresh output directory and runs every stage.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from algoverse import figures
from artifact_index import _sha256, rebase
from boundary_counts_report import count_conditions, load_rows_strict

DEFAULT_PROJECT = Path.home() / "Library/CloudStorage/GoogleDrive-jdjonathandesta@gmail.com/My Drive/maheep-yksa"
STAGES = ("init", "tests", "figure1", "figure2", "figure3", "figure4", "figure5",
          "counts", "provenance", "verify")
PATHS = (("l07", "qwen7b"), ("l13", "qwen7b"), ("l08", "llama8b"))
STEPS = (8, 70, 281)
ARMS = ("ed", "ec", "id", "ic")
BASENAMES = {
    "figure1": "heatmap_qwen_clean", "figure2": "recovery_rt",
    "figure3": "probe5_offer_stratum_qwen7b", "figure4": "probe4_llama8b_m0direction",
    "figure5": "transfer_negotiation_vs_insider",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def code_identity(repo=REPO):
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout
    paths = [*repo.glob("scripts/*.py"), *repo.glob("src/algoverse/*.py"),
             *repo.glob("tests/test_*.py"), repo / "Camera-Ready Figures Runbook (Mac).ipynb"]
    return {"commit": git("rev-parse", "HEAD").strip(),
            "working_tree_status": git("status", "--short"),
            "sha256": {str(p.relative_to(repo)): _sha256(p) for p in sorted(paths) if p.is_file()}}


def recovery_paths(project):
    return [project / "results" / ("e3-%s-t%03d-%s-%s-s42" % (arm, step, layer, model)) / "rows.jsonl"
            for layer, model in PATHS for step in STEPS for arm in ARMS]


def gates(project):
    root = project / "reports/figure-records"
    return [root / ("gate-v2-%s%s.json" % (layer, suffix))
            for layer, suffix in (("l07", ""), ("l10", ""), ("l13", ""), ("l21", ""),
                                  ("l08", "-llama8b"), ("l24", "-llama8b"))]


def sweeps(project):
    r = project / "results"
    return [("M_D (deceptive)", r / "sweep-md-qwen7b-s42-step281"),
            *[("edit " + w, r / ("sweep-e1-%s-qwen7b-s42" % w))
              for w in ("l07", "l10", "l13", "l21")]]


def probe_paths(project, llama=False):
    r = project / "results"
    if llama:
        return {key: r / ("diag-probe4-rt-%s-llama8b-%s-d1" % (key, fit)) / "interp.jsonl"
                for key, fit in (("m0", "own"), ("md", "fixed"),
                                 ("me-l08", "fixed"), ("me-l24", "fixed"))}
    return {key: r / ("diag-probe5-rt-%s-qwen7b-%s-own8" % (key, fit)) / "interp.jsonl"
            for key, fit in (("m0", "own"), ("md8", "fixed"), ("md", "fixed"),
                             ("me-l07", "fixed"), ("me-l10", "fixed"),
                             ("me-l13", "fixed"), ("me-l21", "fixed"))}


def validate_probe(path, key, llama=False):
    rows = jsonl(path)
    analysis = "probe_auroc" if llama else "probe_auroc_stratified:offer"
    points = [r for r in rows if r.get("analysis") == analysis]
    expected_layers = set(range(32 if llama else 28))
    if len(points) != len(expected_layers) or {r.get("layer") for r in points} != expected_layers:
        raise ValueError("missing or duplicate probe layer: %s" % path)
    expected_fit = "diag-probe4-rt-m0-llama8b" if llama else "diag-probe5-rt-m0-qwen7b"
    labels = set()
    for row in points:
        config = row["config"]
        expected = {"n_test": 305, "n_test_lied": 51} if llama else {
            "n_test": 100, "n_test_lied": 51, "stratum": "offer",
            "n_stratum": 82, "n_stratum_lied": 33}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("wrong probe population: %s" % path)
        if config.get("feature_position") != "mean_response_tokens" or config.get("exclude_final_line"):
            raise ValueError("wrong probe feature: %s" % path)
        fit = config.get("fit_source")
        if key == "m0":
            if fit is not None or row.get("adapter_path") is not None:
                raise ValueError("M0 profile must use its own base-model fit")
        elif (config.get("fit") != "fixed_direction_from:" + expected_fit or
              not fit or fit.get("fit_run_id") != expected_fit or fit.get("adapter_path") is not None):
            raise ValueError("profile does not use the M0 direction: %s" % path)
        labels.add(config.get("label_source"))
    if len(labels) != 1:
        raise ValueError("mixed probe test-set lineage: %s" % path)
    return points


def training_manifests(project):
    c = project / "checkpoints"
    pairs = [("original/md-qwen7b", c / "md-qwen7b-s42/train_manifest (1).json"),
             ("original/md-llama8b", c / "md-llama8b-s42/train_manifest.json")]
    pairs += [("edit/" + p.parent.name, p) for layer, model in
              (("l07", "qwen7b"), ("l10", "qwen7b"), ("l13", "qwen7b"),
               ("l21", "qwen7b"), ("l08", "llama8b"), ("l24", "llama8b"))
              for p in [c / ("edit-%s-%s-s42/train_manifest.json" % (layer, model))]]
    pairs += [("continuation/" + p.parent.name, p) for layer, model in PATHS
              for arm in ("ed", "ec")
              for p in [c / ("e2-%s-%s-%s-s42/train_manifest.json" % (arm, layer, model))]]
    pairs += [("intact-continuation/" + p.parent.name, p)
              for model in ("qwen7b", "llama8b") for arm in ("id", "ic")
              for p in [c / ("s2-%s-%s-s42/train_manifest.json" % (arm, model))]]
    return pairs


def supporting_inputs(project):
    r = project / "results"
    out = []
    for layer in ("l07", "l13"):
        out += [r / ("e1-%s-qwen7b-s42-reloc-base/rows.jsonl" % layer),
                r / ("e3-ed-t281-%s-qwen7b-s42-reloc-base/rows.jsonl" % layer)]
        out += sorted((r / ("sweep-e3-ed-%s-t281-qwen7b-s42" % layer)).glob("*/rows.jsonl"))
    return out


def source_inventory(project):
    inputs = recovery_paths(project) + gates(project)
    inputs += [p for _, p in training_manifests(project)]
    for _, root in sweeps(project):
        paths = sorted(root.glob("*/rows.jsonl"))
        if len(paths) != 28:
            raise ValueError("expected 28 layers under %s, found %d" % (root, len(paths)))
        inputs += paths
    inputs += list(probe_paths(project).values()) + list(probe_paths(project, True).values())
    records = project / "reports/figure-records"
    inputs += [records / x for x in ("recovery-qwen.jsonl", "recovery-l07.jsonl",
               "recovery-l13.jsonl", "recovery-l08-llama.jsonl", "tau-llama.jsonl", "tau-insider.jsonl")]
    for p in gates(project):
        gate = read_json(p)
        inputs += [rebase(x, project) for kind in ("rows", "competence")
                   for x in gate["inputs"][kind].values()]
    for name in ("tau-llama.jsonl", "tau-insider.jsonl"):
        inputs += [rebase(row["rows_path"], project) for row in jsonl(records / name)]
    inputs += supporting_inputs(project)
    # Preserve the existing figure files, not just the records used to render.
    for basename in BASENAMES.values():
        inputs += [p for ext in ("pdf", "png")
                   for p in [project / "figures" / (basename + "." + ext)] if p.exists()]
    return sorted(set(p.resolve() for p in inputs))


def validate_recovery_files(project):
    paths = recovery_paths(project)
    for path in paths:
        rows = load_rows_strict(path)
        if {r.get("run_id") for r in rows} != {path.parent.name}:
            raise ValueError("wrong run ID: %s" % path)
        counts = count_conditions(rows)
        if any(counts[c]["n"] != 295 for c in counts):
            raise ValueError("incomplete recovery population: %s" % path)
    return paths


class Workflow:
    def __init__(self, project, output):
        self.project = project.resolve()
        self.out = output.resolve()
        self.records = self.project / "reports/figure-records"
        self.figures = self.out / "figures"
        self.reports = self.out / "reports"
        self.state_path = self.out / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else None

    def save(self):
        write_json(self.state_path, self.state)

    def init(self):
        if self.out.exists():
            raise ValueError("choose a fresh --out directory: %s" % self.out)
        if any(p.name in ("results", "checkpoints") for p in (self.out, *self.out.parents)):
            raise ValueError("output cannot be inside results or checkpoints")
        self.out.mkdir(parents=True)
        self.figures.mkdir()
        self.reports.mkdir()
        self.state = {"created": now(), "project": str(self.project), "output": str(self.out),
                      "completed": [], "failed": None, "invocations": [], "code": code_identity()}
        self.save()
        sources = source_inventory(self.project)
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise ValueError("missing inputs: " + ", ".join(missing))
        self.state["source_sha256"] = {str(p): _sha256(p) for p in sources}
        self.save()
        print("New output: %s (%d source files pinned)" % (self.out, len(sources)))

    def command(self, script, *args, label, inputs=(), records=(), outputs=(), executable=None):
        argv = [executable or sys.executable, "-u", str(REPO / script), *map(str, args)]
        item = {"figure": label, "argv": argv, "timestamp": now(), "status": "started",
                "commit": self.state["code"]["commit"], "code_identity": self.state["code"],
                "inputs": list(map(str, inputs)), "records": list(map(str, records)),
                "outputs": list(map(str, outputs))}
        def log():
            with (self.reports / "commands.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")
        log()
        print("$ " + shlex.join(argv), flush=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        result = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
        item.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    finished=now(), status="passed" if result.returncode == 0 else "failed")
        log()
        self.state["invocations"].append(item)
        self.save()
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            raise RuntimeError("%s failed (exit %d); see commands.jsonl" % (label, result.returncode))
        return result.stdout

    def render(self, stage, *args, inputs=(), records=()):
        basename = BASENAMES[stage]
        outputs = [self.figures / (basename + "." + ext) for ext in ("png", "pdf")]
        return self.command("scripts/make_figures.py", *args, "--out-dir", self.figures,
                            "--basename", basename, label=stage, inputs=inputs,
                            records=records, outputs=outputs)

    def tests(self):
        pure = ("test_boundary_counts_pure.py", "test_artifact_index_pure.py",
                "test_paper_revision_pure.py")
        plotting = ("test_edit_heatmap.py", "test_figure_emitters.py", "test_plotting.py")
        for suite in pure + plotting:
            self.command("tests/" + suite, label="test:" + suite,
                         executable="python3" if suite in pure else sys.executable)

    def figure1(self):
        columns, args, inputs = [], ["edit-heatmap", "--n-layers", 28, "--title", ""], []
        for key, root in sweeps(self.project):
            by_layer = {}
            for path in sorted(root.glob("*/rows.jsonl")):
                rows = load_rows_strict(path)
                layer = rows[0]["bypassed_layer"]
                if layer in by_layer or any(r["bypassed_layer"] != layer for r in rows):
                    raise ValueError("ambiguous sweep layer: %s" % path)
                by_layer[layer] = rows
            if set(by_layer) != set(range(28)):
                raise ValueError("missing sweep layers: %s" % root)
            columns.append((key, by_layer))
            args += ["--sweep", "%s=%s" % (key, root)]
            inputs.append(root)
            if key.startswith("edit "):
                manifest = self.project / ("checkpoints/edit-%s-qwen7b-s42/train_manifest.json" % key[5:])
                args += ["--edit-manifest", "%s=%s" % (key, manifest)]
                inputs.append(manifest)
        data = figures.edit_heatmap_cells(columns, 28)
        cells = [c for group in data["cells"].values() for c in group.values()]
        if sum(c["status"] == "voided_validity" for c in cells) != 18 or sum(c["n_clean"] == 0 for c in cells) != 9:
            raise ValueError("heatmap validity coverage differs from reviewed records")
        cell = data["cells"]["edit l07"][2]
        rows = dict(columns)["edit l07"][2]
        counts = count_conditions(rows)["incentive"]
        if (counts["n_valid"], counts["n_deceptive"], counts["n_trunc"]) != (34, 16, 66):
            raise ValueError("unexpected l07/L2 counts")
        print("l07/L2: 34 clean incentive responses; 16 deceptive; 66 truncated.")
        write_json(self.reports / "heatmap-cells.json", data)
        self.render("figure1", *args, inputs=inputs)

    def figure2(self):
        sources = validate_recovery_files(self.project)
        records = self.records / "recovery-qwen.jsonl"
        rows = jsonl(records)
        keys = {(r["env"], r["checkpoint_step"]) for r in rows}
        if len(rows) != 6 or keys != {(l, t) for l in ("l07", "l13") for t in STEPS}:
            raise ValueError("missing or duplicate recovery record")
        self.render("figure2", "rt", records, "--title", "",
                    "--env-label", "l07=Qwen edit l07", "--env-label", "l13=Qwen edit l13",
                    "--annotate", "l07:8", "--xlabel", "checkpoint index t",
                    "--note", "Checkpoint 70 follows 71 optimizer updates (282 updates total).",
                    "--note", "Scenario-bootstrap 95% intervals; boundary-rate Wilson intervals: Appendix D.",
                    inputs=sources, records=[records])

    def probe(self, llama=False):
        paths = probe_paths(self.project, llama)
        labels = set()
        for key, path in paths.items():
            points = validate_probe(path, key, llama)
            labels.add(points[0]["config"]["label_source"])
        if len(labels) != 1:
            raise ValueError("probe curves do not share a response population")
        args = ["probe-curves", "--title", "", "--analysis",
                "probe_auroc" if llama else "probe_auroc_stratified:offer"]
        for key, path in paths.items():
            args += ["--interp", "%s=%s" % (key, path)]
        self.render("figure4" if llama else "figure3", *args, inputs=paths.values())

    def figure3(self):
        self.probe(False)

    def figure4(self):
        self.probe(True)

    def figure5(self):
        source = self.records / "tau-llama.jsonl"
        selected = [r for r in jsonl(source) if r.get("model") == "Llama-3.1-8B" and r.get("label") == "M_0"]
        if len(selected) != 1 or not selected[0]["rows_path"].endswith("m0-baseline-llama8b-rep/rows.jsonl"):
            raise ValueError("Figure 5 requires the repeated Llama baseline")
        transfer = self.reports / "tau-transfer-llama.jsonl"
        insider = self.records / "tau-insider.jsonl"
        for row in jsonl(insider):
            if row.get("model") == "Llama-3.1-8B" and row.get("label") in ("M_0", "M_D"):
                if Path(row["rows_path"]).name != "rows.jsonl":
                    raise ValueError("Figure 5 requires ratified marker-window records")
        self.command("scripts/emit_figure_records.py", "transfer", "--model", "Llama-3.1-8B",
                     "--tau", "Offer Negotiation=%s" % source,
                     "--tau", "Insider Trading=%s" % insider, "--out", transfer,
                     label="Figure 5 input regrouping", records=[source, insider], outputs=[transfer])
        self.render("figure5", "tau-bars", transfer, "--title", "",
                    "--note", "Llama-3.1-8B; negotiation M0: repeated baseline (tau = 0.105).",
                    "--note", "Insider Trading: ratified marker window; 95% scenario-bootstrap intervals.",
                    records=[transfer, source, insider])

    def counts(self):
        sources = validate_recovery_files(self.project)
        args = ["--project", self.project, "--out", self.reports / "boundary-counts.txt"]
        for path in sources:
            args += ["--rows", path]
        for path in gates(self.project):
            args += ["--gate-record", path]
        self.command("scripts/boundary_counts_report.py", *args, label="Boundary counts",
                     inputs=sources, records=gates(self.project),
                     outputs=[self.reports / "boundary-counts.txt"])

    def provenance(self):
        invocation_path = self.reports / "figure-invocations.json"
        entries = [r for r in self.state["invocations"] if not r["figure"].startswith("test:")]
        entries.append({"figure": "Relocation runtime audit", "inputs": list(map(str, supporting_inputs(self.project))),
                        "historical_status": "Existing records inspected; no new sweep or relocation decision.",
                        "code_identity": self.state["code"], "commit": self.state["code"]["commit"]})
        entries.append({"figure": "Historical Gate-1 and sweep-selection decisions", "status": "unavailable",
                        "historical_status": "Original decision outputs were not archived in the reviewed packet. "
                        "This workflow does not fabricate historical decisions or random-draw records."})
        write_json(invocation_path, entries)
        args = ["--project", self.project, "--invocations", invocation_path,
                "--out-md", self.reports / "artifact-index.md",
                "--out-json", self.reports / "artifact-index.json"]
        for phase, path in training_manifests(self.project):
            args += ["--training-manifest", "%s=%s" % (phase, path)]
        self.command("scripts/artifact_index.py", *args, label="Artifact index",
                     inputs=[p for _, p in training_manifests(self.project)],
                     records=[invocation_path],
                     outputs=[self.reports / "artifact-index.md", self.reports / "artifact-index.json"])
        self.write_text_edits()

    def write_text_edits(self):
        text = """Remaining manuscript edits (not applied to the paper)
N01: Reduce ATTRIB main text to six pages.
N02: Append the completed NeurIPS checklist for the inherited FLLMPT requirements.
N03: Figure 1: grey/red-outline cells are unmeasurable; dot = attempted with zero usable responses;
     hatching, if present, = never attempted. 18 voided cells, nine zero-clean; all cells attempted.
     l07/L2: 34 clean incentive responses, 16 deceptive, 66 truncated. Drop 'indistinguishable'.
N04: Figure 3 now uses the M0 direction for md8. Qwen offer stratum: 82 responses/33 lies;
     Llama Figure 4: 305 responses/51 lies; deletion diagnostic: separate 100-response set.
N05: Both Llama windows were edited; only l08 was continued.
N06: Use 'bypass-nominated candidate windows and control windows'. Add:
     'The experiment establishes recoverability under renewed supervision, without distinguishing
     retained machinery from new learning.'
N07: Describe reported model dtype by phase using artifact-index.md. The model dtype field does
     not mean compute dtype. Adapters are fp32; quantized compute and training autocast are fp16.
N08: Include boundary-counts.txt (all six gates and 36 recovery arm/checkpoint files).
     Wilson intervals describe individual rates; retain bootstrap intervals for gaps/ratios.
N09: Figure 2: edited, not lesioned. At checkpoint 8, l07 tau_ED=1, tau_ID=.210, control gaps=0.
     Checkpoint 70 follows 71 optimizer updates.
N10: Distinguish three close recovery studies from the subset that remeasures location.
N11: Insider Trading: no positive incentive-driven gap. Llama L14 A=.03 [0,.07] does not meet
     the criterion. Drift bound is <= .25 nats.
N12: Figure 5 now uses m0-baseline-llama8b-rep, tau=.104918, CI [.068852,.140984].
     Earlier figure used the original run, tau=.085246. Retain both historical records.
     Name the ratified marker window. Disclose relocation runtime differences in artifact-index.md.
     The local artifact index contains absolute paths; anonymize it before external submission.
N13: Remove the unsupported hypothetical linking unique probe decodability to bypass causality.
N14: Replace 'Qwen, :,' in reference [26] and correct abstract grammar. Use the regenerated
     figures at full text width; recheck readability after incorporation into the paper.
"""
        (self.reports / "manuscript-edits.txt").write_text(text, encoding="utf-8")
        print(text)

    def verify(self):
        mismatches = [path for path, digest in self.state["source_sha256"].items()
                      if not Path(path).is_file() or _sha256(path) != digest]
        if mismatches:
            raise ValueError("source files changed: " + ", ".join(mismatches))
        if code_identity()["sha256"] != self.state["code"]["sha256"]:
            raise ValueError("code changed during this run; use a fresh output directory")
        expected = [self.figures / (base + "." + ext) for base in BASENAMES.values()
                    for ext in ("png", "pdf")]
        expected += [self.reports / name for name in ("heatmap-cells.json", "boundary-counts.txt",
                     "tau-transfer-llama.jsonl", "artifact-index.md", "artifact-index.json",
                     "figure-invocations.json", "manuscript-edits.txt")]
        for path in expected:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("missing or empty output: %s" % path)
        index = read_json(self.reports / "artifact-index.json")
        bad = [i for f in index["figures"] for kind in ("inputs", "record_files", "outputs")
               for i in f.get(kind, []) if i["status"] in ("MISSING", "EMPTY")]
        bad += [m for m in index["training"] if m["status"] != "ok"]
        if bad:
            raise ValueError("missing or empty indexed artifact")
        write_json(self.out / "verification.json",
                   {"verified": now(), "source_files_unchanged": len(self.state["source_sha256"]),
                    "outputs": {str(p.relative_to(self.out)): _sha256(p) for p in expected},
                    "automated_checks": "passed", "visual_review": "inspect figures at paper width"})
        print("PASS: all outputs exist; %d source hashes unchanged." % len(self.state["source_sha256"]))

    def run(self, stage):
        # Refusing a reused destination must not modify an earlier run's
        # state, including when its state.json was loaded by __init__.
        if stage == "init" and self.out.exists():
            raise ValueError("choose a fresh --out directory: %s" % self.out)
        if stage != "init":
            if not self.state or self.state.get("project") != str(self.project):
                raise ValueError("run init first with the same project/output")
            if self.state.get("failed"):
                raise ValueError("this output records a failure; start a fresh output directory")
            index = STAGES.index(stage)
            required = STAGES[:index] if stage in ("provenance", "verify") else ("init",)
            if stage not in ("init", "tests"):
                required = (*required, "tests")
            if any(s not in self.state["completed"] for s in required):
                raise ValueError("required stages are incomplete: " + ", ".join(required))
            if stage in self.state["completed"]:
                raise ValueError("stage already completed; start a fresh run to regenerate")
        try:
            getattr(self, stage)()
            self.state["completed"].append(stage)
            self.save()
        except Exception as exc:
            if self.state is not None and self.out.exists():
                self.state["failed"] = {"stage": stage, "time": now(), "error": str(exc)}
                self.save()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--out", type=Path, required=True, help="fresh versioned output directory")
    args = parser.parse_args(argv)
    workflow = Workflow(args.project, args.out)
    for stage in STAGES if args.stage == "all" else (args.stage,):
        workflow.run(stage)


if __name__ == "__main__":
    main()
