"""A paired GSM8K / MATH-500 pilot for one-step, math-specific self-editing."""
import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import transformers

from editor import Editor, active, read_adapter, save_adapter
from math_eval import FILES, sample_tasks, solve


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(rows):
    return dict(n=len(rows), correct=sum(r["correct"] is True for r in rows),
                grading_errors=sum(r["correct"] is None for r in rows),
                missing_box=sum(r["predicted"] is None for r in rows),
                hit_cap=sum(r["hit_cap"] for r in rows),
                generated_tokens=sum(r["generated_tokens"] for r in rows))


@torch.inference_mode()
def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    discovery = sample_tasks("gsm8k_train", args.dev, args.seed)
    holdout = {k: sample_tasks(k, n, args.seed) for k, n in
               [("gsm8k_test", args.gsm), ("math500", args.math)]}
    questions = [t["question"].strip() for t in discovery + sum(holdout.values(), [])]
    if len(set(questions)) != len(questions):
        raise ValueError("Repeated questions in selected splits")
    settings = dict(seed=args.seed, limit=args.tokens, batch_size=args.batch_size,
                    thinking=not args.no_thinking)
    manifest = {"decoding": {**settings, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                             "presence_penalty": 1.5, "samples_per_question": 1},
                "split_ids": {k: [t["id"] for t in tasks] for k, tasks in
                              {"discovery": discovery, **holdout}.items()},
                "model": str(Path(args.model).resolve()),
                "data_sha256": {k: v[0] for k, v in FILES.items()}}
    dump(out / "protocol.json", manifest)
    editor = Editor(args.model, args.device)
    costs = {}
    def evaluate(arm, tasks, note=""):
        name = arm
        t0 = time.perf_counter()
        def checkpoint(rows):
            with (out / f"{name}.jsonl").open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(name, metrics(rows), f"{time.perf_counter()-t0:.1f}s", flush=True)
        rows = solve(editor, tasks, note=note, on_batch=checkpoint, **settings)
        costs[name] = round(time.perf_counter()-t0, 2)
        return rows

    dev_rows = evaluate("discovery", discovery)
    from math_edit import design, compile_math
    model_recipe, fixed_recipe, memory, trace = design(editor, dev_rows)
    with_factors = compile_math(editor, [model_recipe, fixed_recipe], discovery, dev_rows, settings["thinking"])
    save_adapter(out / "adapter", editor.base, model_recipe, with_factors[0])
    model_recipe, factors = read_adapter(out / "adapter")
    dump(out / "design.json", {"self_recipe": asdict(model_recipe), "fixed_recipe": asdict(fixed_recipe),
                              "prompt_note": memory, "writer_trace": trace})
    per_arm = {}
    paired = {}
    base_correct = {}
    for arm, ctx, note in [
        ("base", nullcontext(), ""), ("prompt", nullcontext(), memory),
        ("self_lora", active(editor, model_recipe.module, factors), ""),
        ("fixed_lora", active(editor, fixed_recipe.module, with_factors[1]), ""),
    ]:
        per_arm[arm] = {}
        paired[arm] = {}
        with ctx:
            all_rows = evaluate(arm, sum(holdout.values(), []), note)
            for dataset in holdout:
                rows = [r for r in all_rows if r["source"] == dataset]
                per_arm[arm][dataset] = metrics(rows)
                if arm == "base":
                    base_correct.update({r["id"]: r["correct"] for r in rows})
                paired[arm][dataset] = {
                    "gained": [r["id"] for r in rows if r["correct"] and not base_correct[r["id"]]],
                    "lost": [r["id"] for r in rows if not r["correct"] and base_correct[r["id"]]]}
    report = {"model": editor.base, "device": str(editor.device), "torch": str(torch.__version__),
              "transformers": transformers.__version__, "discovery": metrics(dev_rows),
              "decoding": manifest["decoding"], "arms": per_arm, "paired_vs_base": paired,
              "wall_seconds": round(time.perf_counter()-started, 2), "generation_seconds": costs,
              "adapter_parameters": sum(x.numel() for x in factors)}
    dump(out / "report.json", report)
    print(json.dumps(per_arm, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True)
    cli.add_argument("--out", required=True)
    cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli.add_argument("--dev", type=int, default=4)
    cli.add_argument("--gsm", type=int, default=8)
    cli.add_argument("--math", type=int, default=8)
    cli.add_argument("--tokens", type=int, default=1024)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--seed", type=int, default=19)
    cli.add_argument("--no-thinking", action="store_true")
    args = cli.parse_args()
    if min(args.batch_size, args.tokens) < 1:
        cli.error("batch size and token limit must be positive")
    run(args)
