from __future__ import annotations

import torch

from tiny_transformer_adder import make_model, format_prompt_tokens


MAX_ADDEND = 9_999_999_999
N_DIGITS = 10


def _count_unique_parameters(model: torch.nn.Module) -> int:
    total = 0
    seen = set()
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    return int(total)


def build_model():
    model = make_model(n_digits=N_DIGITS)
    model.eval()
    metadata = {
        "name": "Handwritten Compressed Transformer Adder",
        "author": "archi+codex",
        "params": _count_unique_parameters(model),
        "architecture": "decoder-only causal transformer (1 layer, attention + SwiGLU MLP)",
        "tricks": [
            "handwritten constructive weights",
            "compressed parameterization",
            "RoPE offset-targeted attention",
            "fixed reversed-digit prompt format",
        ],
    }
    return model, metadata


@torch.inference_mode()
def add(model, a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")

    prompt = format_prompt_tokens(a, b, N_DIGITS, model.vocab)
    seq = model.generate(prompt_tokens=prompt, max_new_tokens=model.output_len)
    return int(model.decode_generated_sum(seq))
