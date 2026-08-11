#!/usr/bin/env python3
"""Deterministic Phase-2A probe for no-sink versus a scene-local RoPE epoch."""

import argparse
import json
import math
from pathlib import Path

import torch

from wan.modules.causal_model import block_relativistic_rope
from wan.modules.model import rope_params


ROOT = Path(__file__).resolve().parents[1]


def coordinate_trace():
    """Temporal positions from the live no-sink path for A frames 0--8, B 9--17."""
    return {
        "first_B": {
            "scene_cut": True,
            "current_qk_positions": [45, 46, 47],
            "scene_local_qk_positions": [0, 1, 2],
            "cache_write": "raw K/V written; clean pass preserves transformed K frame 9 at phase 45",
        },
        "second_B": {
            "scene_cut": False,
            "current_query_positions": [12, 13, 14],
            "scene_local_query_positions": [3, 4, 5],
            "current_key_positions": [45, 1, 2, 3, 4, 5],
            "scene_local_key_positions": [0, 1, 2, 3, 4, 5],
        },
        "third_B": {
            "scene_cut": False,
            "current_query_positions": [15, 16, 17],
            "scene_local_query_positions": [6, 7, 8],
            "current_key_positions": [45, 1, 2, 3, 4, 5, 6, 7, 8],
            "scene_local_key_positions": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    }


def _freqs(head_dim=12, sequence_length=64):
    complex_dim = head_dim // 2
    split = [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3]
    return torch.cat([rope_params(sequence_length, 2 * width) for width in split], dim=1)


def _rope(x, start_frame, scene_cut=False):
    return block_relativistic_rope(
        x, torch.tensor([[x.shape[1], 1, 1]]), _freqs(x.shape[-1]),
        start_frame=start_frame, scene_cut=scene_cut)


def _stats(left, right):
    error = (left - right).abs()
    return {"max_abs": float(error.max()), "mean_abs": float(error.mean())}


def _attention_stats(current_q, current_k, local_q, local_k, value):
    scale = math.sqrt(current_q.shape[-1])
    current_logits = torch.einsum("blhd,bshd->bhls", current_q, current_k) / scale
    local_logits = torch.einsum("blhd,bshd->bhls", local_q, local_k) / scale
    current_output = torch.einsum("bhls,bshd->blhd", current_logits.softmax(-1), value)
    local_output = torch.einsum("bhls,bshd->blhd", local_logits.softmax(-1), value)
    return {
        "attention_logits_max_abs": _stats(current_logits, local_logits)["max_abs"],
        "attention_logits_mean_abs": _stats(current_logits, local_logits)["mean_abs"],
        "attention_output_max_abs": _stats(current_output, local_output)["max_abs"],
        "attention_output_mean_abs": _stats(current_output, local_output)["mean_abs"],
    }


def _context(raw, transformed_sink):
    roped = _rope(raw, 0)
    roped[:, :1] = transformed_sink
    return roped


def run_equivalence_probe(seed=0):
    """Compare identical raw Q/K/V/cache tensors under the two live coordinate rules."""
    torch.manual_seed(seed)
    shape = (1, 3, 2, 12)
    raw_b1, raw_b2, raw_b3 = (torch.randn(shape, dtype=torch.float64) for _ in range(3))
    q_b1, q_b2, q_b3 = (torch.randn(shape, dtype=torch.float64) for _ in range(3))

    current_q1, current_k1 = _rope(q_b1, 9, scene_cut=True), _rope(raw_b1, 0, scene_cut=True)
    local_q1, local_k1 = _rope(q_b1, 0), _rope(raw_b1, 0)
    value_b1 = torch.randn_like(raw_b1)
    first = _attention_stats(current_q1, current_k1, local_q1, local_k1, value_b1)

    # Live clean pass re-writes B1 raw K, then preserves only its first frame already transformed.
    current_sink, local_sink = current_k1[:, :1].clone(), local_k1[:, :1].clone()
    context_b2_raw = torch.cat([raw_b1, raw_b2], dim=1)
    current_q2, local_q2 = _rope(q_b2, 12), _rope(q_b2, 3)
    current_k2, local_k2 = _context(context_b2_raw, current_sink), _context(context_b2_raw, local_sink)
    value_b2 = torch.randn_like(context_b2_raw)

    context_b3_raw = torch.cat([raw_b1, raw_b2, raw_b3], dim=1)
    current_q3, local_q3 = _rope(q_b3, 15), _rope(q_b3, 6)
    current_k3, local_k3 = _context(context_b3_raw, current_sink), _context(context_b3_raw, local_sink)
    value_b3 = torch.randn_like(context_b3_raw)

    raw_non_sink = _attention_stats(
        current_q2, current_k2[:, 1:], local_q2, local_k2[:, 1:], value_b2[:, 1:])
    return {
        "probe": "synthetic float64 RoPE/attention with identical raw Q/K/V and no model weights",
        "trace": coordinate_trace(),
        "first_B_denoise": first,
        "first_B_clean_pass": first.copy(),
        "clean_pass_sink_key": _stats(current_sink, local_sink),
        "second_B_full_context": _attention_stats(current_q2, current_k2, local_q2, local_k2, value_b2),
        "second_B_raw_non_sink_only": raw_non_sink,
        "third_B_full_context": _attention_stats(current_q3, current_k3, local_q3, local_k3, value_b3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/hard_cut_transition_phase2a_20260810/scene_local_rope_probe.json")
    args = parser.parse_args()
    report = run_equivalence_probe(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
