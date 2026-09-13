"""Artifact index: figure -> rendering command -> records -> result runs.

Why: the review found no record of which runs and which commands produced
the paper's figures (some were rendered ad hoc). This script builds that
index from an invocations file the figure runbook writes as it renders
(one entry per figure: the argv, the record files it read, the results
inputs behind them) and from all rows of every input results file
(run id, timestamp, library versions, dtype/quantization, model revision,
adapter digest, and for Insider Trading rows the grading window). Absent
inputs are listed as MISSING, never dropped. Stdlib only; writes only the
two --out files.

    python3 scripts/artifact_index.py --project ~/maheep-yksa \
        --invocations ~/maheep-yksa/reports/figure-invocations.json \
        --out-md ~/maheep-yksa/reports/artifact-index.md \
        --out-json ~/maheep-yksa/reports/artifact-index.json

Invocation entry schema:
  {"figure": "Fig 1", "basename": "heatmap_qwen_clean", "argv": [...],
   "records": ["<record path>", ...], "inputs": ["<rows or interp path or
   sweep root>", ...], "commit": "<sha>", "timestamp": "<iso>"}
A path under another machine's copy of the project (".../maheep-yksa/
results/x/rows.jsonl") is rebased onto --project at "results/".
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

VERSION_FIELDS = ("torch_version", "transformers_version", "peft_version",
                  "bitsandbytes_version")


def rebase(path, project):
    """Rebase a foreign path onto the project at its results/ segment."""
    text = str(path)
    marker = "/results/"
    if marker in text:
        return Path(project) / "results" / text.split(marker, 1)[1]
    return Path(text)


def read_first_row(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return None


def _gen_config(row):
    gen = row.get("gen_config") or {}
    if isinstance(gen, str):
        try:
            import ast
            gen = ast.literal_eval(gen)
        except (ValueError, SyntaxError):
            gen = {}
    return gen if isinstance(gen, dict) else {}


def grading_window(path):
    name = Path(path).name
    if name == "rows.jsonl":
        return "ratified marker window (rows.jsonl)"
    if "whole_report" in name:
        return "whole-report re-grade (sensitivity sidecar)"
    return ""


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_input(path, project):
    """One index entry for an input file.

    rows.jsonl / interp.jsonl files get every recorded runtime identity; any
    other file (a training manifest, a record file passed as an input) is
    indexed by size and sha256 so it is still pinned, never dropped.
    """
    local = rebase(path, project)
    entry = {"path": str(path), "local_path": str(local)}
    if not local.is_file():
        entry["status"] = "MISSING"
        return entry
    entry.update({"bytes": local.stat().st_size, "sha256": _sha256(local)})
    if local.suffix != ".jsonl":
        entry.update({"status": "file"})
        return entry
    rows = []
    with local.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("%s:%d: invalid JSON" % (local, number)) from exc
    if not rows:
        entry["status"] = "EMPTY"
        return entry
    row = rows[0]
    gen = _gen_config(row)
    load = gen.get("load_profile") or {}
    entry.update({
        "status": "ok",
        "run_id": row.get("run_id"),
        "timestamp": row.get("timestamp"),
        "model_id": row.get("model_id"),
        "adapter_path": row.get("adapter_path"),
        "checkpoint_step": row.get("checkpoint_step"),
        "arm": row.get("arm"),
        "versions": {k: gen.get(k) for k in VERSION_FIELDS},
        "dtype": load.get("dtype"),
        "quant": gen.get("quant") or load.get("four_bit"),
        "model_revision": gen.get("model_revision"),
        "adapter_digest": gen.get("adapter_digest"),
    })
    if "insider" in str(local):
        entry["grading_window"] = grading_window(local)
    if local.name == "interp.jsonl":
        config = row.get("config") or {}
        entry["probe"] = {k: config.get(k) for k in
                          ("fit", "feature_position", "n_test", "n_test_lied", "label_source")}
    # Keep the first-row fields for existing consumers, but explicitly
    # enumerate identities over the WHOLE file. No null is inferred from
    # a nearby run or from the installed packages on the rendering Mac.
    identities, probes = {}, {}
    for item in rows:
        g = _gen_config(item)
        lp = g.get("load_profile") or {}
        ident = {"versions": {k: g.get(k) for k in VERSION_FIELDS},
                 "dtype": lp.get("dtype"), "quant": g.get("quant") or lp.get("four_bit"),
                 "model_revision": g.get("model_revision"),
                 "adapter_digest": g.get("adapter_digest")}
        key = json.dumps(ident, sort_keys=True)
        identities.setdefault(key, dict(ident, n_rows=0))["n_rows"] += 1
        if local.name == "interp.jsonl":
            config = item.get("config") or {}
            probe = {k: config.get(k) for k in (
                "fit", "fit_source", "feature_position", "n_test", "n_test_lied",
                "n_stratum", "n_stratum_lied", "stratum", "label_source",
                "aggregation", "exclude_final_line")}
            probes[json.dumps(probe, sort_keys=True)] = probe
    entry["row_count"] = len(rows)
    entry["run_ids"] = sorted({str(item.get("run_id")) for item in rows})
    entry["runtime_identities"] = list(identities.values())
    entry["mixed_runtime"] = len(identities) > 1
    entry["probe_identities"] = list(probes.values())
    entry["missing_runtime_fields"] = sorted({
        key for ident in identities.values() for key, value in ident["versions"].items()
        if value is None})
    return entry


def describe_training(spec):
    phase, sep, path = spec.partition("=")
    if not sep or not phase or not path:
        raise ValueError("--training-manifest expects PHASE=PATH")
    p = Path(path)
    if not p.is_file():
        return {"phase": phase, "path": path, "status": "MISSING"}
    record = json.loads(p.read_text(encoding="utf-8"))
    return {"phase": phase, "path": path, "status": "ok", "sha256": _sha256(p),
            "model_id": record.get("model_id"), "model_dtype": record.get("dtype"),
            "adapter_dtype": record.get("adapter_dtype"), "quant": record.get("quant_label"),
            "packages": record.get("packages"), "objective": record.get("objective"),
            "train_layers": (record.get("config") or {}).get("train_layers")}


def expand_inputs(inputs, project):
    """Sweep roots expand to every <tag>-lNN/rows.jsonl; files pass through."""
    files = []
    for spec in inputs:
        local = rebase(spec, project)
        if local.is_dir():
            layer_dirs = sorted(
                child for child in local.iterdir()
                if re.search(r"-l(\d+)$", child.name) and (child / "rows.jsonl").is_file()
            )
            if layer_dirs:
                files.extend(str(child / "rows.jsonl") for child in layer_dirs)
                continue
        files.append(str(spec))
    return files


def record_inputs(record_path, project):
    """rows_path fields carried by a record file (tau / transfer records)."""
    local = rebase(record_path, project)
    if not local.is_file():
        return []
    found = []
    text = local.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
        items = loaded if isinstance(loaded, list) else [loaded]
    except json.JSONDecodeError:
        items = [json.loads(l) for l in text.splitlines() if l.strip()]
    for item in items:
        if isinstance(item, dict) and item.get("rows_path"):
            found.append(item["rows_path"])
        if isinstance(item, dict) and item.get("record") == "edit_gate":
            for paths in (item.get("inputs") or {}).values():
                if isinstance(paths, dict):
                    found.extend(value for value in paths.values() if isinstance(value, str))
    return found


def build_index(invocations, project, training_manifests=()):
    figures = []
    for entry in invocations:
        inputs = list(entry.get("inputs") or [])
        for record in entry.get("records") or []:
            inputs.extend(record_inputs(record, project))
        seen, ordered = set(), []
        for path in expand_inputs(inputs, project):
            if path not in seen:
                seen.add(path)
                ordered.append(path)
        figures.append({
            "figure": entry.get("figure"),
            "basename": entry.get("basename"),
            "command": shlex.join([str(a) for a in entry.get("argv") or []]),
            "commit": entry.get("commit"),
            "timestamp": entry.get("timestamp"),
            "status": entry.get("status", "recorded"),
            "code_identity": entry.get("code_identity"),
            "historical_status": entry.get("historical_status"),
            "records": [str(r) for r in entry.get("records") or []],
            "record_files": [describe_input(p, project) for p in entry.get("records") or []],
            "outputs": [describe_input(p, project) for p in entry.get("outputs") or []],
            "inputs": [describe_input(p, project) for p in ordered],
        })
    return {"project": str(project), "figures": figures,
            "training": [describe_training(spec) for spec in training_manifests]}


def _versions(entry):
    v = entry.get("versions") or {}
    return "torch %s / transformers %s / peft %s / bnb %s" % (
        v.get("torch_version"), v.get("transformers_version"),
        v.get("peft_version"), v.get("bitsandbytes_version"))


def render_markdown(index):
    lines = ["# Artifact index", "",
             "Each figure: the command that rendered it, the record files it read, and "
             "every results file behind those records with its recorded runtime identity.",
             "Every input, record and output is hashed in the JSON companion. "
             "Runtime identities are collected from all rows; unavailable fields stay null. "
             "This is a local audit index: absolute paths can identify the authors.", ""]
    if index.get("training"):
        lines += ["## Training precision by phase", "",
                  "Model dtype is the manifest's reported model dtype, not the compute dtype. "
                  "The loader uses fp16 4-bit compute; training uses fp16 autocast with a scaler. "
                  "These settings are supported by the recorded code hashes, not inferred from model dtype.", "",
                  "| Phase/run | Model dtype | Adapter dtype | Quant | Packages |",
                  "|---|---|---|---|---|"]
        for item in index["training"]:
            lines.append("| %s | %s | %s | %s | %s |" % (
                item["phase"], item.get("model_dtype", item["status"]),
                item.get("adapter_dtype"), item.get("quant"), item.get("packages")))
        lines.append("")
    for fig in index["figures"]:
        lines.append("## %s — `%s`" % (fig["figure"], fig["basename"]))
        lines.append("")
        lines.append("- commit `%s`, rendered %s" % (fig.get("commit"), fig.get("timestamp")))
        lines.append("- command: `%s`" % fig["command"])
        if fig.get("historical_status"):
            lines.append("- historical status: " + fig["historical_status"])
        if fig.get("code_identity"):
            lines.append("- code: working-tree status and SHA-256 fingerprints are in the JSON companion")
        if fig["records"]:
            lines.append("- records: " + ", ".join("`%s`" % r for r in fig["records"]))
        lines.append("")
        lines.append("| run_id | file | first row | versions | dtype / quant | revision | adapter digest | note |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for inp in fig["inputs"]:
            if inp.get("status") == "file":
                lines.append("| file | `%s` | | | | | | sha256 %s, %d bytes |"
                             % (inp["path"], inp["sha256"][:12], inp["bytes"]))
                continue
            if inp.get("status") != "ok":
                lines.append("| %s | `%s` | | | | | | %s |" % (inp.get("status"), inp["path"], inp.get("status")))
                continue
            note = inp.get("grading_window", "")
            if inp.get("probe"):
                note = "fit=%s; n_test=%s" % (inp["probe"].get("fit"), inp["probe"].get("n_test"))
            note += "; %d runtime identity/identities; sha256 %s" % (
                len(inp["runtime_identities"]), inp["sha256"][:12])
            if inp["missing_runtime_fields"]:
                note += "; missing runtime fields: " + ", ".join(inp["missing_runtime_fields"])
            versions = " ; ".join(_versions(i) for i in inp["runtime_identities"])
            lines.append("| %s | `%s` | %s | %s | %s / %s | %s | %s | %s |" % (
                inp.get("run_id"), Path(inp["local_path"]).relative_to(index["project"])
                if str(inp["local_path"]).startswith(index["project"]) else inp["local_path"],
                (inp.get("timestamp") or "")[:19], versions, inp.get("dtype"),
                inp.get("quant"), (inp.get("model_revision") or "")[:12],
                (inp.get("adapter_digest") or "")[:12], note))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--invocations", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--training-manifest", action="append", default=[],
                        help="PHASE=PATH to a saved train_manifest JSON (repeatable)")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    for out in (args.out_md, args.out_json):
        resolved = Path(out).resolve()
        if any(p.name == "results" for p in (resolved, *resolved.parents)):
            raise SystemExit("outputs must not be under results/")
    invocations = json.loads(Path(args.invocations).read_text(encoding="utf-8"))
    index = build_index(invocations, project, args.training_manifest)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(index), encoding="utf-8")
    n_missing = sum(1 for f in index["figures"] for i in f["inputs"] if i.get("status") == "MISSING")
    print("indexed %d figures, %d inputs (%d missing) -> %s, %s"
          % (len(index["figures"]), sum(len(f["inputs"]) for f in index["figures"]),
             n_missing, args.out_md, args.out_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
