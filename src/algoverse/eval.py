"""
This module contains the benchmarking and evaluation methods for models.

The heart of it is run_negotiation_eval: give it a READY model object plus
a list of scenarios, and it generates, scores, and appends one
self-describing JSONL row per scenario x condition. It takes a model
OBJECT, never a model name, so the plain base model, a LoRA checkpoint,
and a layer-bypassed model all evaluate through identical code.

Rows are appended to disk the moment they are scored, and a re-run skips
work that is already on disk (resume). Colab free-tier sessions die
mid-run; with resume, a disconnect costs one batch, not the session.

Capability benchmarks (MMLU, GSM8K), perplexity, and the Gate-1 report
join this module in the next build stage.
"""

import datetime

import torch

from algoverse.tasks import (
    CONDITIONS,
    get_scenarios,
    render_messages,
    score_response,
)
from algoverse.utils import append_jsonl, read_jsonl, set_seed

# Every results row carries all of these fields, always. Self-describing
# rows are what let any file be analyzed alone, long after the run.
ROW_FIELDS = [
    "run_id", "timestamp", "model_id", "adapter_path",
    "bypassed_layer", "patch_layer", "patch_source",
    "checkpoint_step", "arm",
    "condition", "scenario_id", "split", "scenario_params",
    "response_text", "hit_max_tokens",
    "claimed_value", "true_value", "deceptive", "deception_type",
    "understated", "valid", "invalid_reason", "extraction_method",
    "seed", "gen_config",
]


def _eos_ids(model, tokenizer) -> list:
    """All token ids that legitimately end a generation, as a plain list."""
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return [eos]
    return list(eos or [])


def generate_batch(model, tokenizer, message_lists, max_new_tokens=256,
                   do_sample=False, batch_size=4):
    """Run chat prompts through the model in batches.

    Args:
    - message_lists: a list of chat-message lists (what render_messages
      returns), NOT raw strings. The tokenizer's own chat template does the
      formatting; nothing here is model-specific.

    Returns a list of (response_text, hit_max_tokens), aligned with the
    input. hit_max_tokens=True means generation ran out of budget before
    emitting an end token, i.e. the reply was CUT OFF. The scorer treats
    that as its own invalid category, never as refusal or honesty.
    """
    # Decoder-only models must pad on the LEFT for batched generation:
    # right padding would put pad tokens between the prompt and the new
    # text, and the model would generate garbage. Classic silent bug,
    # handled here in exactly one place.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    eos_ids = _eos_ids(model, tokenizer)

    results = []
    for start in range(0, len(message_lists), batch_size):
        batch = message_lists[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        for i in range(len(batch)):
            new_tokens = output[i][prompt_length:].tolist()
            # Cut off = used the whole budget and never emitted an end token.
            hit_max = len(new_tokens) >= max_new_tokens and not any(
                eos in new_tokens for eos in eos_ids
            )
            text = tokenizer.decode(
                output[i][prompt_length:], skip_special_tokens=True
            ).strip()
            results.append((text, hit_max))
    return results


def run_negotiation_eval(model, tokenizer, scenarios, run_id, out_path,
                         model_id, conditions=CONDITIONS,
                         adapter_path=None, bypassed_layer=None,
                         patch_layer=None, patch_source=None,
                         checkpoint_step=None, arm=None,
                         batch_size=4, max_new_tokens=256, do_sample=False,
                         seed=42, resume=True, quant_label=None,
                         use_llm_fallback=False, llm_provider="anthropic",
                         llm_model=None) -> list:
    """Evaluate one model on the negotiation task. THE central function.

    For every scenario x condition: render the prompt, generate, score,
    and append one full row to out_path immediately. With resume=True
    (default), (run_id, scenario_id, condition) triples already in the
    file are skipped, so re-running after a crash continues where it left
    off and re-running a finished job generates nothing.

    The intervention fields (bypassed_layer, patch_layer, patch_source,
    checkpoint_step, arm) are pure bookkeeping stamped onto every row; the
    actual intervention must already live inside the model object handed
    in. Bypass and patch are recorded separately because they are
    different causal evidence and must never blur together in analysis.

    Returns every row for this run_id (previously existing + newly made).
    """
    from pathlib import Path

    set_seed(seed)
    out_path = Path(out_path)

    existing = []
    if resume and out_path.exists():
        existing = [r for r in read_jsonl(out_path) if r.get("run_id") == run_id]
    done = {(r["run_id"], r["scenario_id"], r["condition"]) for r in existing}

    todo = [
        (scenario, condition)
        for scenario in scenarios
        for condition in conditions
        if (run_id, scenario["scenario_id"], condition) not in done
    ]
    print(
        "run %s: %d rows already done, %d to generate"
        % (run_id, len(existing), len(todo))
    )

    gen_config = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "quant": quant_label,
    }

    new_rows = []
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        messages = [
            render_messages(scenario, condition) for scenario, condition in chunk
        ]
        generations = generate_batch(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            batch_size=batch_size,
        )
        for (scenario, condition), (response_text, hit_max) in zip(chunk, generations):
            scoring = score_response(
                scenario, condition, response_text,
                hit_max_tokens=hit_max,
                use_llm_fallback=use_llm_fallback,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            row = {
                "run_id": run_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model_id": model_id,
                "adapter_path": adapter_path,
                "bypassed_layer": bypassed_layer,
                "patch_layer": patch_layer,
                "patch_source": patch_source,
                "checkpoint_step": checkpoint_step,
                "arm": arm,
                "condition": condition,
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "scenario_params": {
                    "company_offer": scenario["company_offer"],
                    "true_outside_offer": scenario["true_outside_offer"],
                    "role": scenario["role"],
                    "company": scenario["company"],
                },
                "response_text": response_text,
                "hit_max_tokens": hit_max,
                "seed": seed,
                "gen_config": gen_config,
            }
            row.update(scoring)
            append_jsonl(out_path, row)  # on disk BEFORE the next batch runs
            new_rows.append(row)
        print("  scored %d/%d" % (min(start + batch_size, len(todo)), len(todo)))

    return existing + new_rows


def smoke_test(model_id=None, n_scenarios=6, out_dir="results/smoke") -> None:
    """End-to-end proof on a tiny model, no GPU needed.

    Loads Qwen2.5-0.5B-Instruct, evaluates n_scenarios x 2 conditions,
    then asserts the things the whole project depends on: every schema
    field present on every row, resume generates nothing on a second call,
    and the metrics compute from the rows. Prints two sample responses for
    eyeballing. Takes a few minutes on a laptop.

    The 0.5B model's actual deception numbers mean NOTHING; this test is
    about plumbing, not science.
    """
    from pathlib import Path

    from algoverse.metrics import tau_with_ci, task_competence
    from algoverse.models import DEV_MODEL, load_model_and_tokenizer

    model_id = model_id or DEV_MODEL
    out_path = Path(out_dir) / "rows.jsonl"
    if out_path.exists():
        out_path.unlink()  # a smoke test always starts from scratch

    print("loading %s ..." % model_id)
    model, tokenizer = load_model_and_tokenizer(model_id, quant="none")

    scenarios = get_scenarios("selection", n=n_scenarios, seed=0)
    rows = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id="smoke", out_path=out_path, model_id=model_id,
        quant_label="none",
    )

    expected = len(scenarios) * 2
    assert len(rows) == expected, "expected %d rows, got %d" % (expected, len(rows))
    for row in rows:
        missing = [field for field in ROW_FIELDS if field not in row]
        assert not missing, "row missing fields: %s" % missing

    # Resume: a second call over the same work must generate nothing.
    rows_again = run_negotiation_eval(
        model, tokenizer, scenarios,
        run_id="smoke", out_path=out_path, model_id=model_id,
        quant_label="none",
    )
    assert len(rows_again) == expected, "resume changed the row count"

    gap = tau_with_ci(rows, n_boot=200, seed=0)
    competence = task_competence(rows)
    print("\nsample responses:")
    for row in rows[:2]:
        print(
            "  [%s | true=%s] %r"
            % (row["condition"], row["true_value"], row["response_text"][:140])
        )
    print("\ntau=%s  CI=[%s, %s]" % (gap["tau"], gap["tau_ci_low"], gap["tau_ci_high"]))
    print(
        "invalid rates: incentive=%s control=%s | competence=%s"
        % (
            gap["invalid_rate_incentive"],
            gap["invalid_rate_control"],
            competence["competence"],
        )
    )
    print("\nSMOKE TEST PASSED: %d rows, schema complete, resume clean" % expected)
