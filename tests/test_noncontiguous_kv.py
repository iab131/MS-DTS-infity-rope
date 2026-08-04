#!/usr/bin/env python3
"""Synthetic packed-retrieval check; it does not preserve source positions."""

import math
import unittest

import torch

from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, attention
from wan.modules.causal_model import block_relativistic_rope
from wan.modules.model import rope_apply, rope_params


def rope_at_source_positions(x, positions, freqs):
    """Test-only RoPE reference for non-contiguous temporal source positions."""
    heads, half_dim = x.size(2), x.size(3) // 2
    temporal, height, width = freqs.split(
        [half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3], dim=1)
    complex_x = torch.view_as_complex(x[0].float().reshape(x.size(1), heads, -1, 2))
    position_freqs = torch.cat([
        temporal.index_select(0, positions),
        height[:1].expand(x.size(1), -1),
        width[:1].expand(x.size(1), -1),
    ], dim=1).reshape(x.size(1), 1, -1)
    return torch.view_as_real(complex_x * position_freqs).flatten(2).unsqueeze(0).type_as(x)


class NonContiguousKVTest(unittest.TestCase):
    """Check source selection [0, 3, 7] packed to RoPE positions [0, 1, 2]."""

    def test_packed_retrieval_matches_fp32_reference(self):
        if not (torch.cuda.is_available() and FLASH_ATTN_2_AVAILABLE):
            self.skipTest("requires CUDA FlashAttention 2")

        torch.manual_seed(0)
        device, dtype, heads, head_dim = "cuda", torch.bfloat16, 2, 64
        source_positions = torch.tensor([0, 3, 7], device=device)
        grid = torch.tensor([[3, 1, 1]], device=device)
        freqs = torch.cat([
            rope_params(64, head_dim - 4 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
        ], dim=1).to(device)
        cache_k = torch.randn(1, 8, heads, head_dim, device=device, dtype=dtype)
        cache_v = torch.randn_like(cache_k)
        retrieved_k = cache_k.index_select(1, source_positions)
        retrieved_v = cache_v.index_select(1, source_positions)
        torch.testing.assert_close(retrieved_k, cache_k[:, [0, 3, 7]])

        # Retrieval intentionally packs non-contiguous source frames into [0, 1, 2].
        packed_k = block_relativistic_rope(retrieved_k, grid, freqs, start_frame=0)
        torch.testing.assert_close(packed_k, rope_apply(retrieved_k, grid, freqs))
        self.assertFalse(torch.allclose(
            packed_k, rope_at_source_positions(retrieved_k, source_positions, freqs)))
        packed_q = block_relativistic_rope(
            torch.randn_like(retrieved_k), grid, freqs, start_frame=3)

        output = attention(packed_q, packed_k, retrieved_v, causal=False)
        self.assertEqual(output.shape, (1, 3, heads, head_dim))
        scores = torch.einsum("blhd,bshd->bhls", packed_q.float(), packed_k.float())
        reference = torch.einsum(
            "bhls,bshd->blhd",
            torch.softmax(scores / math.sqrt(head_dim), dim=-1),
            retrieved_v.float(),
        )
        max_error = (output.float() - reference).abs().max().item()
        self.assertLessEqual(max_error, 1e-2)

        # Separate causal check: the existing attention mask must match FP32 triangular masking.
        causal_output = attention(packed_k, packed_k, retrieved_v, causal=True)
        self.assertEqual(causal_output.shape, (1, 3, heads, head_dim))
        causal_scores = torch.einsum("blhd,bshd->bhls", packed_k.float(), packed_k.float())
        causal_mask = torch.triu(torch.ones(3, 3, device=device, dtype=torch.bool), diagonal=1)
        causal_reference = torch.einsum(
            "bhls,bshd->blhd",
            torch.softmax(causal_scores.masked_fill(causal_mask, float("-inf")) / math.sqrt(head_dim), dim=-1),
            retrieved_v.float(),
        )
        causal_max_error = (causal_output.float() - causal_reference).abs().max().item()
        self.assertLessEqual(causal_max_error, 1e-2)
        print(f"FlashAttention max errors: noncausal={max_error:.7f}, causal={causal_max_error:.7f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
