"""Retest the frozen base and saved math LoRA with thinking disabled."""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from editor import Editor, active, read_adapter
from math_eval import FILES, sample_tasks, solve
from run_math import dump, metrics


@torch.inference_mode()
def run(source, out, device):
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    cfg = dict(protocol["decoding"])
    if not cfg["thinking"] or protocol["data_sha256"] != {k: v[0] for k, v in FILES.items()}:
        raise ValueError("Expected a matching, thinking-enabled run")
    if not (source / "report.json").is_file():
        raise ValueError("The thinking-enabled run must be complete")
    tasks = []
    for dataset in ("gsm8k_test", "math500"):
        frozen_ids = protocol["split_ids"][dataset]
        part = sample_tasks(dataset, len(frozen_ids), cfg["seed"])
        if [t["id"] for t in part] != frozen_ids:
            raise ValueError("Could not reconstruct the originally sampled questions")
        tasks.extend(part)
    out.mkdir(parents=True, exist_ok=False)
    cfg["thinking"] = False
    dump(out / "protocol.json", {**protocol, "decoding": cfg, "source_run": str(source.resolve()),
                                "arms": ["base", "self_lora"]})
    editor = Editor(protocol["model"], device)
    recipe, factors = read_adapter(source / "adapter")
    report = {}
    paired = {}
    baseline = {}
    start = time.perf_counter()
    for arm, ctx in [("base", nullcontext()), ("self_lora", active(editor, recipe.module, factors))]:
        def checkpoint(batch):
            with (out / f"{arm}.jsonl").open("a", encoding="utf-8") as f:
                for row in batch:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(arm, metrics(batch), flush=True)
        with ctx:
            rows = solve(editor, tasks, seed=cfg["seed"], limit=cfg["limit"],
                         batch_size=cfg["batch_size"], thinking=False, on_batch=checkpoint)
        report[arm] = {s: metrics([r for r in rows if r["source"] == s])
                       for s in ("gsm8k_test", "math500")}
        if arm == "base":
            baseline = {r["id"]: r["correct"] for r in rows}
        else:
            paired = {"gained": [r["id"] for r in rows if r["correct"] and not baseline[r["id"]]],
                      "lost": [r["id"] for r in rows if not r["correct"] and baseline[r["id"]]]}
    dump(out / "report.json", {"arms": report, "paired_vs_base": paired,
                              "wall_seconds_excluding_load": round(time.perf_counter()-start, 2)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = cli.parse_args()
    run(args.source, args.out, args.device)
