"""Stratified probe readout: tabulate what the diag-probe5 rows say.

Reads every <root>/<prefix>*/interp.jsonl written by run_probe_transfer.py
(2026-09-02 stratified design) and prints, per output:

  main      transfer AUROC over the whole test set: mean over layers, best
            layer, and how many layers' 95% CI sits above / below 0.5
  offer     the same restricted to scenarios WITH a real outside offer —
            the stratum where "lied" cannot mean "no-offer scenario"
  no_offer  the complement (where both classes exist)
  source:*  per generating run, for pooled test sets
  pairs     matched-scenario accuracy P(score_lied > score_honest)
  controls  the confound probes' AUROC against the LIED label: scenario
            type (has_offer) and generator identity (pooled sets). A
            deception reading has to beat these floors. Every line also
            shows separation in EITHER sign (max of AUROC and 1-AUROC):
            a control at 0.03 vs the lied label is a 0.97 floor.

Reading guide: the decisive line is `offer` on the own8 sets — one
generator, offer scenarios only — where neither scenario type nor style
can carry the label. On the pairs sets the generator control is perfect
BY CONSTRUCTION (M_0 wrote every honest row, M_D every lying row), so a
pairs result only counts if the own8 offer stratum agrees.

Nothing here is a paper quantity on its own; it is the table the rows
already contain. Stdlib only.

    python scripts/probe_matrix_report.py --root results --prefix diag-probe5-
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

PAT = re.compile(r"(?P<prefix>.*?)(?P<pos>fp|rx|rt|cl)-(?P<ckpt>m0|md8|md|me-l\d+)-(?P<fam>qwen7b|llama8b)-(?P<fit>own|fixed)-(?P<set>[A-Za-z0-9_]+)$")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize_curve(rows):
    vals = [(r["layer"], r["value"], r.get("ci_low"), r.get("ci_high")) for r in rows if r.get("value") is not None]
    if not vals:
        return None
    mean = sum(v for _, v, _, _ in vals) / len(vals)
    best = max(vals, key=lambda t: t[1])
    worst = min(vals, key=lambda t: t[1])
    above = sum(1 for _, v, lo, hi in vals if lo is not None and lo > 0.5)
    below = sum(1 for _, v, lo, hi in vals if hi is not None and hi < 0.5)
    # separation in EITHER sign: an AUROC of 0.02 separates as well as 0.98
    # (the probe's "untruthful" side just lands on the honest rows)
    sep = [max(v, 1.0 - v) for _, v, _, _ in vals]
    return dict(n=len(vals), mean=mean, best=best, worst=worst, above=above, below=below,
                sep_mean=sum(sep) / len(sep), sep_max=max(sep))


def fmt(s):
    if s is None:
        return "(single-class or absent)"
    return ("mean %.3f | best %.3f@L%02d CI[%s,%s] | worst %.3f@L%02d | CI>0.5 %2d/%d | CI<0.5 %2d | either-sign sep mean %.3f max %.3f"
            % (s["mean"], s["best"][1], s["best"][0],
               "n/a" if s["best"][2] is None else "%.2f" % s["best"][2],
               "n/a" if s["best"][3] is None else "%.2f" % s["best"][3],
               s["worst"][1], s["worst"][0], s["above"], s["n"], s["below"],
               s["sep_mean"], s["sep_max"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prefix", default="diag-probe5-")
    args = parser.parse_args(argv)
    root = Path(args.root)
    dirs = sorted(p for p in root.glob(args.prefix + "*") if (p / "interp.jsonl").is_file())
    if not dirs:
        raise SystemExit("no %s*/interp.jsonl under %s" % (args.prefix, root))
    print("%d outputs under %s" % (len(dirs), root))
    order = {"m0": 0, "md8": 1, "md": 2}
    parsed = []
    for d in dirs:
        m = PAT.match(d.name)
        if not m:
            print("  (skipping unrecognised dir %s)" % d.name)
            continue
        parsed.append((m.group("fam"), m.group("pos"), m.group("set"), order.get(m.group("ckpt"), 9),
                       m.group("ckpt"), m.group("fit"), d))
    for fam, pos, test_set, _, ckpt, fit, d in sorted(parsed):
        rows = load(d / "interp.jsonl")
        by = collections.defaultdict(list)
        for r in rows:
            by[r.get("analysis")].append(r)
        main_rows = by.get("probe_auroc", [])
        cfg = main_rows[0]["config"] if main_rows else {}
        print("\n== %s %s set=%s %s [%s]  n_test=%s lied=%s  %s"
              % (fam, pos, test_set, ckpt, fit, cfg.get("n_test"), cfg.get("n_test_lied"), d.name))
        print("   main      " + fmt(summarize_curve(main_rows)))
        strata = collections.defaultdict(list)
        for analysis, group in by.items():
            if analysis and analysis.startswith("probe_auroc_stratified:"):
                strata[analysis.split(":", 1)[1]].extend(group)
        for name in sorted(strata):
            n = strata[name][0]["config"].get("n_stratum"); k = strata[name][0]["config"].get("n_stratum_lied")
            print("   %-9s " % name[:9] + fmt(summarize_curve(strata[name])) + "  (n=%s, lied=%s)" % (n, k))
        pairs = by.get("probe_pair_accuracy", [])
        if pairs:
            s = summarize_curve(pairs)
            print("   pairs     " + fmt(s) + "  (%s scenario pairs)" % pairs[0]["config"].get("n_pairs"))
        controls = collections.defaultdict(list)
        for analysis, group in by.items():
            if analysis and analysis.startswith("control_probe_auroc:"):
                controls[analysis.split(":", 1)[1]].extend(group)
        for name in sorted(controls):
            s = summarize_curve(controls[name])
            own = [r["config"].get("auroc_vs_target") for r in controls[name]
                   if r.get("config", {}).get("auroc_vs_target") is not None]
            print("   ctrl:%-4s " % name[:4] + fmt(s)
                  + ("  (predicts its own target at mean %.3f)" % (sum(own) / len(own)) if own else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
