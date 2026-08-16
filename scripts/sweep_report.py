"""Print the Stage-1 sweep selection report, or the full-pool confirmation.

Selection report (plan D6):

    python scripts/sweep_report.py \
        --base results/sweep-md-qwen7b-base/rows.jsonl \
        --layer 0=results/sweep-md-qwen7b-l00/rows.jsonl \
        --layer 1=results/sweep-md-qwen7b-l01/rows.jsonl \
        --competence base=results/sweep-md-qwen7b-base/competence.jsonl \
        --competence 0=results/sweep-md-qwen7b-l00/competence.jsonl \
        --m0-competence 0.96 \
        --item16-decision "spec item 16 confirmed at 0.25 nats, <date/ref>"

Every --layer N=PATH appears in the table -- disqualified, unmeasurable, or
empty. --competence takes a layer number or the literal "base" (the intact
model's benchmark values, the reference the drop/rise checks need; without
it those checks read "n/e"). --m0-competence is M_0's negotiation
task-competence from Gate 1; omitted, item 2's negotiation check reads
"n/e", never a silent pass.

--item16-decision is the D5 tripwire: the recorded DEV-calibration
confirm-or-revise decision on the 0.25-nat JSD bound. Without it the report
refuses a research-model verdict; --dev bypasses for the DEV model only and
stamps every line.

Confirmation mode (plan D7), after l* is selected:

    python scripts/sweep_report.py --confirm \
        --confirm-base results/md-a3-full/rows.jsonl \
        --confirm-bypassed results/md-l17-n305/rows.jsonl \
        --item16-decision "..."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algoverse.sweep import BASE_KEY, confirm_report, sweep_report


def parse_layer_pairs(pairs, allow_base=False):
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
        if norm in result:
            raise SystemExit("key %r given twice" % key)
        result[norm] = path
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", metavar="PATH",
                        help="intact-run rows.jsonl (bypassed_layer null)")
    parser.add_argument("--layer", action="append", metavar="N=PATH",
                        help="one swept layer's rows.jsonl; repeatable")
    parser.add_argument("--competence", action="append", default=None,
                        metavar="N=PATH",
                        help="competence.jsonl per layer, or base=PATH for "
                             "the intact reference; repeatable, optional")
    parser.add_argument("--m0-competence", type=float, default=None,
                        help="M_0 negotiation task-competence (item 2 "
                             "reference); omitted -> check not evaluated")
    parser.add_argument("--item16-decision", default=None,
                        help="recorded DEV-calibration confirm-or-revise "
                             "decision reference (required for a research-"
                             "model verdict)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true",
                        help="DEV model only: bypass the item-16 tripwire "
                             "and stamp the report")
    parser.add_argument("--confirm", action="store_true",
                        help="full-pool confirmation mode (plan D7)")
    parser.add_argument("--confirm-base", metavar="PATH",
                        help="full-pool intact rows.jsonl (confirm mode)")
    parser.add_argument("--confirm-bypassed", metavar="PATH",
                        help="full-pool rows.jsonl for l* bypassed "
                             "(confirm mode)")
    args = parser.parse_args()

    if args.confirm:
        if not args.confirm_base or not args.confirm_bypassed:
            raise SystemExit(
                "--confirm needs --confirm-base and --confirm-bypassed"
            )
        confirm_report(
            args.confirm_base,
            args.confirm_bypassed,
            item16_decision=args.item16_decision,
            n_boot=args.n_boot,
            seed=args.seed,
            dev=args.dev,
        )
    else:
        if not args.base or not args.layer:
            raise SystemExit("need --base and at least one --layer N=PATH")
        sweep_report(
            args.base,
            parse_layer_pairs(args.layer),
            m0_competence=args.m0_competence,
            competence_inputs=parse_layer_pairs(args.competence, allow_base=True),
            item16_decision=args.item16_decision,
            n_boot=args.n_boot,
            seed=args.seed,
            dev=args.dev,
        )
