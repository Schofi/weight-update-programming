"""Frozen Qwen editor, constrained edit language and reversible rank-one adapters."""
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench import SYSTEM


class Editor:
    def __init__(self, model_path, device):
        self.device = device
        self.base = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.base, local_files_only=True)
        dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else (
            torch.float16 if str(device).startswith("cuda") else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base, local_files_only=True, dtype=dtype, attn_implementation="eager"
        ).eval().requires_grad_(False).to(device)
        ids = [self.tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
        if any(len(tokens) != 1 for tokens in ids) or ids[0] == ids[1]:
            raise ValueError("A/B must be distinct single tokens.")
        self.labels = [tokens[0] for tokens in ids]
        layers = [k for k, _ in self.model.named_modules() if k.endswith(".mlp.down_proj")]
        if len(layers) < 2:
            raise ValueError("Expected a decoder with at least two MLP down projections.")
        self.layers = [layers[len(layers) // 2], layers[-1]]
        self.calls = self.input_tokens = self.output_tokens = 0
        self.model.register_forward_pre_hook(self._count, with_kwargs=True)

    def _count(self, module, args, kwargs):
        self.calls += 1
        self.input_tokens += kwargs["input_ids"].numel()

    def tokens(self, prompt, memory="", *, system=SYSTEM):
        messages = [{"role": "system", "content": system + ("\nDiagnostic note: " + memory if memory else "")},
                    {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    @torch.inference_mode()
    def choose(self, prompt, memory="", *, sample=False, system=SYSTEM):
        inputs = self.tokens(prompt, memory, system=system)
        out = self.model(**inputs, logits_to_keep=1, use_cache=False)
        logits = out.logits[0, -1, self.labels].float().cpu()
        probs = logits.softmax(0)
        if not torch.isfinite(probs).all():
            raise ValueError("Non-finite decision logits.")
        pick = torch.multinomial(probs, 1).item() if sample else probs.argmax().item()
        self.output_tokens += 1
        return pick, probs.tolist()

    @torch.inference_mode()
    def summarize(self, records, sample):
        prompt = ("From your diagnostic records, write a short practical instruction to yourself for "
                  "choosing a discriminating probe and then choosing the repair. Output the instruction only.\n"
                  + records)
        inputs = self.tokens(prompt)
        options = {"do_sample": True, "temperature": 0.8} if sample else {"do_sample": False}
        result = self.model.generate(**inputs, **options, max_new_tokens=48, use_cache=True,
                                     pad_token_id=self.tokenizer.eos_token_id)
        new = result[0, inputs["input_ids"].shape[1]:]
        self.output_tokens += len(new)
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


@dataclass(frozen=True)
class Recipe:
    module: str
    read: str
    write: str
    gain: float


def propose(editor, records, sample):
    header = ("Design a rank-one LoRA for YOUR probe-selection policy. Your diagnosis records follow. "
              "A and B in those records are local action labels. Compile an edit by choosing each "
              "instruction. Contrast statistics are computed from the labeled discovery probes.\n" + records)
    choices = [
        ("Choose MLP down projection: A middle layer; B final layer.", editor.layers),
        ("Choose input read direction: A class mean difference; B variance-scaled mean difference.",
         ["difference", "scaled"]),
        ("Choose output write direction: A output embedding A minus B; B difference of local MLP outputs.",
         ["unembedding", "local"]),
        ("Choose edit strength: A small (0.25); B large (1.0).", [0.25, 1.0]),
    ]
    selected = []
    for question, values in choices:
        pick, _ = editor.choose(header + "\n" + question + " Reply A or B.", sample=sample)
        selected.append(values[pick])
        header += "\n" + question + " Selected: " + str(values[pick])
    return Recipe(*selected)


@contextmanager
def observe(editor, buffers):
    def capture(name):
        def hook(module, args, output):
            buffers[name].append((args[0][0, -1].detach().float().cpu(),
                                  output[0, -1].detach().float().cpu()))
        return hook
    handles = []
    try:
        for name in buffers:
            handles.append(editor.model.get_submodule(name).register_forward_hook(capture(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def compile_recipe(editor, recipe, buffers, labels):
    xs, ys = (torch.stack(items) for items in zip(*buffers[recipe.module]))
    a = torch.tensor(labels) == 0
    read = xs[a].mean(0) - xs[~a].mean(0)
    if recipe.read == "scaled":
        variance = xs.var(0, unbiased=False)
        read /= variance + variance.mean().clamp_min(1e-8)
    if recipe.write == "unembedding":
        emb = editor.model.get_output_embeddings().weight.detach()
        write = (emb[editor.labels[0]].float() - emb[editor.labels[1]].float()).cpu()
    else:
        write = ys[a].mean(0) - ys[~a].mean(0)
    if min(read.norm().item(), write.norm().item()) < 1e-8:
        raise ValueError("Degenerate discovery contrast.")
    read = F.normalize(read, dim=0)
    # Calibrate the RMS update using discovery activations only.
    projection_rms = (xs @ read).square().mean().sqrt().clamp_min(1e-6)
    write = F.normalize(write, dim=0) * (recipe.gain * ys.norm(dim=1).mean() / projection_rms)
    return read.unsqueeze(0).contiguous(), write.unsqueeze(1).contiguous()


@contextmanager
def active(editor, name, factors):
    layer = editor.model.get_submodule(name)
    a, b = (t.to(layer.weight) for t in factors)
    if a.shape != (1, layer.in_features) or b.shape != (layer.out_features, 1):
        raise ValueError("Invalid rank-one factors for " + name)
    def hook(module, args, output):
        return output + F.linear(F.linear(args[0], a), b)
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def save_adapter(directory, base, recipe, factors):
    directory.mkdir(parents=True, exist_ok=False)
    name = "base_model.model." + recipe.module
    save_file({name + ".lora_A.weight": factors[0], name + ".lora_B.weight": factors[1]},
              str(directory / "adapter_model.safetensors"))
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
              "inference_mode": True, "r": 1, "lora_alpha": 1, "lora_dropout": 0.0,
              "target_modules": [recipe.module], "bias": "none"}
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "recipe.json").write_text(json.dumps(asdict(recipe), indent=2), encoding="utf-8")


def read_adapter(directory):
    recipe = Recipe(**json.loads((directory / "recipe.json").read_text(encoding="utf-8")))
    weights = load_file(str(directory / "adapter_model.safetensors"))
    key = "base_model.model." + recipe.module
    return recipe, (weights[key + ".lora_A.weight"], weights[key + ".lora_B.weight"])
