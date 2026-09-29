"""Print the Stage-1 sweep report: every swept layer against the bounds.

    python scripts/sweep_report.py \\
        --base $PROJECT/results/md-qwen7b-s42-step281/rows.jsonl \\
        --layer 0=$PROJECT/results/sweep-md-qwen7b-s42-step281/md-qwen7b-s42-step281-l00/rows.jsonl \\
        --layer 1=$PROJECT/results/sweep-md-qwen7b-s42-step281/md-qwen7b-s42-step281-l01/rows.jsonl \\
        --competence base=$PROJECT/results/md-qwen7b-s42-step281/competence.jsonl \\
        --competence 0=$PROJECT/results/sweep-md-qwen7b-s42-step281/md-qwen7b-s42-step281-l00/competence.jsonl \\
        --m0-competence 0.96

Every --layer N=PATH appears in the table -- voided, breaching, unmeasurable,
or empty. --competence takes a layer number or the literal "base" (the
intact swept model's benchmark values, the reference the drop/rise checks
need; without it those checks read "n/e"). --m0-competence is M_0's
negotiation task-competence from Gate 1; omitted, the competence-drop check
reads "n/e" -- never a silent pass, but enough to make the verdict
"incomplete".

Which file to pass as `base=`: the swept checkpoint's OWN competence.jsonl
(the M_D baseline run's, written by run_baseline.py --competence), which
carries all three baseline metrics -- mmlu_acc, gsm8k_exact_match and
wikitext2_ppl. Every drop is a same-model delta, so M_0's file is refused
(its adapter identity differs). The sweep driver's out_root/base-competence.jsonl
is NOT a substitute either: it holds the unprobed wikitext2_ppl row only, so
the MMLU and GSM8K checks would read "n/e". Passing BOTH files under `base=`
raises -- two sessions' wikitext2_ppl rows do not agree bit-for-bit, and the
duplicate-metric guard rejects them as conflicting.

The Stage-3 sweeps (E,D-t281 and the just-edited M_E) do not go through
this report: their delta curve is produced by scripts/relocation_report.py.

--dev (the DEV model only) stamps every line as not publishable.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.sweep import BASE_KEY, sweep_report


def parse_layer_pairs(pairs, allow_base=False, merge_sources=False):
    result = {}
    for pair in pairs or []:
        key, _, path = pair.partition("=")
        if not path:
            raise SystemExit("expected N=PATH, got %r" % pair)
        if allow_base and key == BASE_KEY:
            norm = BASE_KEY
        else:
            try:
                norm = int(key)
            except ValueError:
                raise SystemExit(
                    "expected an integer layer%s in %r"
                    % (" or 'base'" if allow_base else "", pair)
                )
        if norm in result and not merge_sources:
            raise SystemExit("key %r given twice" % key)
        if merge_sources:
            result.setdefault(norm, []).append(path)
        else:
            result[norm] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, metavar="PATH",
                        help="intact-run rows.jsonl (bypassed_layer null)")
    parser.add_argument("--layer", action="append", required=True,
                        metavar="N=PATH",
                        help="one swept layer's rows.jsonl; repeatable")
    parser.add_argument("--competence", action="append", default=None,
                        metavar="N=PATH",
                        help="competence.jsonl per layer, or base=PATH for "
                             "the intact reference; repeatable, optional")
    parser.add_argument("--m0-competence", type=float, default=None,
                        help="M_0 negotiation task-competence (the "
                             "competence-drop reference); omitted -> check "
                             "not evaluated")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true",
                        help="DEV model only: stamp every line as not "
                             "publishable")
    args = parser.parse_args()

    sweep_report(
        args.base,
        parse_layer_pairs(args.layer),
        m0_competence=args.m0_competence,
        competence_inputs=parse_layer_pairs(
            args.competence, allow_base=True, merge_sources=True
        ),
        n_boot=args.n_boot,
        seed=args.seed,
        dev=args.dev,
    )
