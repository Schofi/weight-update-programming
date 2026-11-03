"""Versioned public math tasks, free-form generation, and final-answer grading."""
import hashlib
import json
import random
import re
import subprocess
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import torch
from transformers import GenerationConfig, LogitsProcessor


DATA = Path(__file__).resolve().parent / "data" / "math"
FILES = {
    "gsm8k_train": ("17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465", 7473),
    "gsm8k_test": ("3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14", 1319),
    "math500": ("35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132", 500),
}
INSTRUCTION = r"Solve the math problem. Show your reasoning, then put the final answer in \boxed{}."


def sample_tasks(name, count, seed):
    raw = (DATA / f"{name}.jsonl").read_bytes()
    digest, size = FILES[name]
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Unexpected data revision for {name}")
    rows = raw.splitlines()
    if len(rows) != size or not 1 <= count <= size:
        raise ValueError(f"Invalid sample size for {name}: {count}")
    selected = []
    for i in random.Random(seed).sample(range(size), count):
        row = json.loads(rows[i])
        if name == "math500":
            selected.append(dict(id=row["unique_id"], source=name, question=row["problem"],
                                 gold=row["answer"], solution=row["solution"]))
        else:
            solution, gold = row["answer"].rsplit("####", 1)
            selected.append(dict(id=f"{name}/{i}", source=name, question=row["question"],
                                 gold=gold.strip(), solution=solution.strip()))
    return selected


def final_answer(text, thinking):
    if thinking:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    # Only the last complete final box is accepted. Intermediate reasoning is never graded.
    marker = r"\boxed{"
    pos = text.rfind(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if not depth:
                return text[start:i].strip() or None
    return None


@lru_cache(maxsize=2048)
def correct(gold, pred):
    if pred is None:
        return False
    if gold.strip() == pred.strip():
        return True
    def scalar(value):
        number = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        m = re.fullmatch(rf"({number})\s*(\\?%)?", value.strip())
        if m:
            return Fraction(m[1].replace(",", "")) / (100 if m[2] else 1)
        return None
    g, p = scalar(gold), scalar(pred)
    if g is not None and p is not None:
        return g == p
    worker = Path(__file__).with_name("math_symbolic.py")
    result = subprocess.run([sys.executable, str(worker)], input=json.dumps([gold, pred]).encode(),
                            capture_output=True, timeout=30, check=True)
    return json.loads(result.stdout)


def chat(editor, question, thinking, note=""):
    content = INSTRUCTION + ("\n" + note if note else "")
    return editor.tokenizer.apply_chat_template(
        [{"role": "system", "content": content}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


class PresencePenalty(LogitsProcessor):
    """Subtract once for each distinct generated token, excluding the prompt."""
    def __init__(self, start, penalty):
        self.start, self.penalty = start, penalty

    def __call__(self, input_ids, scores):
        seen = input_ids[:, self.start:]
        return scores.scatter(1, seen, scores.gather(1, seen) - self.penalty)


def solve(editor, tasks, *, seed, limit, batch_size, note="", thinking=True, on_batch=None):
    tok = editor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    end = editor.model.generation_config.eos_token_id or tok.eos_token_id
    ends = {end} if isinstance(end, int) else set(end)
    cfg = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                           max_new_tokens=limit, use_cache=True, eos_token_id=end,
                           pad_token_id=tok.pad_token_id)
    rows = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        inp = tok([chat(editor, t["question"], thinking, note) for t in batch], padding=True,
                  add_special_tokens=False, return_tensors="pt").to(editor.device)
        width = inp.input_ids.shape[1]
        torch.manual_seed(seed + start)
        ids = editor.model.generate(**inp, generation_config=cfg,
                                    logits_processor=[PresencePenalty(width, 1.5)])
        results = []
        grading_errors = []
        for task, token_ids in zip(batch, ids[:, width:].tolist()):
            stop = next((i for i, x in enumerate(token_ids) if x in ends), len(token_ids))
            # Keep the think delimiter, which some tokenizers mark as special.
            text = tok.decode(token_ids[:stop], skip_special_tokens=False)
            pred = final_answer(text, thinking)
            ended = stop < len(token_ids)
            try:
                passed = correct(task["gold"], pred)
            except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
                passed = None
                grading_errors.append((task["id"], str(exc)))
            results.append(dict(id=task["id"], source=task["source"], question=task["question"],
                                gold=task["gold"], text=text, predicted=pred,
                                correct=passed, generated_tokens=stop + int(ended),
                                hit_cap=not ended and stop == limit, seed=seed + start))
            if task["source"] == "gsm8k_train":
                results[-1]["continuation_ids"] = token_ids[:stop]
        rows.extend(results)
        if on_batch:
            on_batch(results)
        if grading_errors:
            raise RuntimeError(f"Symbolic grading failed; rescore saved predictions: {grading_errors}")
    return rows
