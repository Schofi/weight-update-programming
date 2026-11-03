"""Compile a math-specific rank-one edit from paired discovery continuations."""
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from editor import Recipe
from math_eval import chat


def design(editor, rows):
    system = "You are editing your own math-solving model. Use only the supplied practice results."
    notes = "\n".join(f"Problem: {r['question']}\nYour ending: {r['text'][-360:]}\n"
                      f"Verified answer: {r['gold']}. Correct={r['correct']}; hit cap={r['hit_cap']}."
                      for r in rows)
    layers = ["model.layers.15.mlp.down_proj", "model.layers.19.mlp.down_proj"]
    directions = ["top32", "top64"]
    trace = []
    prompt = notes + "\nDesign one edit subtracting a failed-minus-reference activation direction."
    for instruction, values in [
        ("Choose the MLP layer: A=15, B=19.", layers),
        ("Choose how many most-aligned input columns to edit: A=32, B=64.", directions),
        ("Choose subtraction strength: A=0.5, B=1.0.", [0.5, 1.0]),
    ]:
        choice, probs = editor.choose(prompt + "\n" + instruction, system=system)
        trace.append({"instruction": instruction, "choice": values[choice], "ab_probs": probs})
        prompt += "\n" + instruction + " Selected " + str(values[choice]) + "."
    recipe = Recipe(trace[0]["choice"], trace[1]["choice"], "subtract_error_direction", trace[2]["choice"])
    fixed = Recipe(layers[1], "top64", "subtract_error_direction", 1.)
    inputs = editor.tokens(notes + "\nWrite one short general tip for solving NEW math problems. "
                           "Omit all example-specific facts and numbers.", system=system)
    ids = editor.model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True,
                                pad_token_id=editor.tokenizer.eos_token_id)
    memory = editor.tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return recipe, fixed, memory, trace


def math_factors(weight, delta, top, gain):
    # Selected columns of W' = W - gain * u (u^T W) exactly form a rank-one update.
    u = F.normalize(delta.float(), dim=0)
    coeff = u @ weight.detach().float()
    selected = coeff.abs().topk(top).indices
    a = torch.zeros_like(coeff)
    a[selected] = -gain * coeff[selected]
    return a[None, :].cpu().contiguous(), u[:, None].cpu().contiguous()


def compile_math(editor, recipes, tasks, rows, thinking):
    modules = {r.module: editor.model.get_submodule(r.module) for r in recipes}
    deltas = {name: torch.zeros(layer.out_features, device=editor.device) for name, layer in modules.items()}
    count = 0
    for task, row in zip(tasks, rows, strict=True):
        if task["id"] != row["id"]:
            raise ValueError("Mismatched discovery trajectory")
        if row["correct"]:
            continue
        prompt = chat(editor, task["question"], thinking)
        reference = (task["solution"] + ("\n</think>\n" if thinking else "\n")
                     + r"\boxed{" + task["gold"] + "}")
        prefix = editor.tokenizer.encode(prompt, add_special_tokens=False)
        ref_ids = editor.tokenizer.encode(reference, add_special_tokens=False)
        for sign, tail in ((1., row["continuation_ids"]), (-1., ref_ids)):
            if not tail:
                tail = [editor.tokenizer.eos_token_id]
            ids = torch.tensor([prefix + tail], device=editor.device)
            width = min(32, len(tail))
            def capture(name):
                def hook(module, args, output):
                    deltas[name].add_(output[0, -width:].float().mean(0), alpha=sign)
                return hook
            with ExitStack() as stack:
                for name, layer in modules.items():
                    stack.callback(layer.register_forward_hook(capture(name)).remove)
                editor.model(input_ids=ids, use_cache=False, logits_to_keep=1)
        count += 1
    # With no observed errors, the zero direction gives an identity adapter.
    return [math_factors(modules[r.module].weight, deltas[r.module]/max(count, 1),
                         int(r.read.removeprefix("top")), r.gain) for r in recipes]
