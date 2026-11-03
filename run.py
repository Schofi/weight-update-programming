"""Run the one-generation, forward-only self-edit experiment."""
import argparse
import json
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import transformers

from bench import cases, grade, probe_score
from editor import (Editor, Recipe, active, compile_recipe, observe, propose, read_adapter,
                    save_adapter)


def evaluate(editor, subset, memory="", buffers=None):
    rows = []
    for case in subset:
        with observe(editor, buffers) if buffers is not None else nullcontext():
            probe, p_probe = editor.choose(case.probe_prompt(), memory)
        result = probe_score(case, probe)
        repair, p_repair = editor.choose(case.repair_prompt(probe, result), memory)
        rows.append({"id": case.key, "probe": probe, "probe_input": case.probes()[probe],
                     "probe_score": result, "repair": repair, "probe_probs": p_probe,
                     "repair_probs": p_repair, "informative": probe == case.good_slot,
                     **grade(case, repair)})
    return rows


def summary(rows):
    return {"n": len(rows), **{key: sum(row[key] for row in rows)
            for key in ("informative", "repaired", "preserved", "joint", "restored_states",
                        "broken_states", "regressed_states", "intact_states")}}


def discovery_notes(subset, rows):
    lines = ["Order of coefficients: flag1, flag2, style. All discoveries started at [1,1,0]."]
    for case, row in zip(subset, rows):
        label = "AB"[row["probe"]]
        fixed = case.fields[case.repair_columns()[row["repair"]]]
        lines.append(f"{'/'.join(case.fields)}: tried {label} {row['probe_input']}; "
                     f"score {row['probe_score']:g}; informative={row['informative']}; "
                     f"patched {fixed}; all checks passed={row['joint']}.")
    return "\n".join(lines)


def rank_candidate(rows):
    counts = summary(rows)
    return counts["joint"], counts["informative"], counts["preserved"]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    gpu = torch.device(args.device).type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    editor = Editor(args.model, args.device)
    costs = {}
    def sync():
        if gpu:
            torch.cuda.synchronize(args.device)

    def measured(key, func):
        sync()
        ts = time.perf_counter()
        before = (editor.calls, editor.input_tokens, editor.output_tokens)
        result = func()
        sync()
        after = (editor.calls, editor.input_tokens, editor.output_tokens)
        costs[key] = dict(zip(("forwards", "processed_input_tokens", "output_tokens"),
                              [y - x for x, y in zip(before, after)]))
        costs[key]["seconds"] = round(time.perf_counter() - ts, 3)
        print(f"{key}: {costs[key]['forwards']} forwards in {costs[key]['seconds']} s", flush=True)
        return result

    discovery = cases("discovery")
    validation = cases("validation")
    buffers = {name: [] for name in editor.layers}
    records = measured("discovery", lambda: evaluate(editor, discovery, buffers=buffers))
    notes = discovery_notes(discovery, records)
    dump(out / "discovery.json", {"records": records, "writer_input": notes})
    options = {"prompt": [], "self_lora": []}
    factors_list = []
    for i in range(args.candidates):
        memory = measured(f"prompt_writer_{i}", lambda: editor.summarize(notes, sample=i > 0))
        prompt_rows = measured(f"prompt_val_{i}", lambda: evaluate(editor, validation, memory))
        options["prompt"].append({"memory": memory, "rows": prompt_rows})
        recipe = measured(f"lora_writer_{i}", lambda: propose(editor, notes, sample=i > 0))
        factors = measured(f"compile_{i}", lambda: compile_recipe(
            editor, recipe, buffers, [case.good_slot for case in discovery]))
        factors_list.append(factors)
        with active(editor, recipe.module, factors):
            patch_rows = measured(f"lora_val_{i}", lambda: evaluate(editor, validation))
        options["self_lora"].append({"recipe": asdict(recipe), "rows": patch_rows})
    winners = {arm: max(range(len(items)), key=lambda i: rank_candidate(items[i]["rows"]))
               for arm, items in options.items()}
    dump(out / "validation.json", {"candidates": options, "selected": winners})
    idx = winners["self_lora"]
    recipe = Recipe(**options["self_lora"][idx]["recipe"])
    save_adapter(out / "adapter", editor.base, recipe, factors_list[idx])
    recipe, factors = read_adapter(out / "adapter")
    fixed = Recipe(editor.layers[-1], "difference", "unembedding", 0.25)
    fixed_factors = measured("fixed_compile", lambda: compile_recipe(
        editor, fixed, buffers, [case.good_slot for case in discovery]))
    test = cases("test")
    rows = {"base": measured("test_base", lambda: evaluate(editor, test)),
            "prompt": measured("test_prompt", lambda: evaluate(
                editor, test, options["prompt"][winners["prompt"]]["memory"]))}
    with active(editor, recipe.module, factors):
        rows["self_lora"] = measured("test_self_lora", lambda: evaluate(editor, test))
    with active(editor, fixed.module, fixed_factors):
        rows["fixed_lora"] = measured("test_fixed_lora", lambda: evaluate(editor, test))
    dump(out / "test.json", rows)
    metrics = {k: summary(v) for k, v in rows.items()}
    pairs = {}
    for arm in ("prompt", "self_lora", "fixed_lora"):
        base = rows["base"]
        other = rows[arm]
        pairs[arm] = {"gained": sum(y["joint"] and not x["joint"] for x, y in zip(base, other)),
                      "lost": sum(x["joint"] and not y["joint"] for x, y in zip(base, other)),
                      "probe_flips": sum(x["probe"] != y["probe"] for x, y in zip(base, other))}
    report = {"model": editor.base, "device": args.device, "seed": args.seed,
              "candidate_budget_per_arm": args.candidates,
              "gpu_name": torch.cuda.get_device_name(args.device) if gpu else None,
              "dtype": str(next(editor.model.parameters()).dtype),
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "python": platform.python_version(), "recipe": asdict(recipe),
              "fixed_recipe": asdict(fixed), "distinct_test_probe_prompts": len({c.probe_prompt() for c in test}),
              "selected": winners, "adapter_parameters": sum(t.numel() for t in factors),
              "metrics": metrics, "paired_vs_base": pairs, "costs": costs,
              "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(args.device) / 2**20, 1)
              if gpu else None, "total_wall_seconds": round(time.perf_counter() - started, 2)}
    dump(out / "report.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="Local Qwen text-compatible checkpoint directory")
    cli.add_argument("--out", required=True, help="New directory for this run")
    cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli.add_argument("--candidates", type=int, default=2)
    cli.add_argument("--seed", type=int, default=19)
    args = cli.parse_args()
    if args.candidates < 1:
        cli.error("--candidates must be positive")
    with torch.inference_mode():
        run(args)
