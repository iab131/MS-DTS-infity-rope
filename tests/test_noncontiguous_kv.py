#!/usr/bin/env python3
"""Synthetic packed-retrieval check; it does not preserve source positions."""

import math
import unittest

import torch

from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, attention
from wan.modules.causal_model import (
    assemble_noncontiguous_context,
    block_relative_positions,
    block_relativistic_rope,
    materialize_retrieved_kv,
)
from wan.modules.model import rope_apply, rope_params
from pipeline.causal_inference import (
    capture_clean_kv_to_cpu,
    select_history_frame_refs,
    select_history_kv,
)


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


class NonContiguousContextAssemblyTest(unittest.TestCase):
    """CPU checks for the opt-in historical-KV prefix."""

    def test_history_replaces_non_sink_frames_with_matched_shape(self):
        frame_tokens = 2
        retrieved_k = torch.full((1, 2, 1, 1), 99.0)
        retrieved_v = torch.full((1, 2, 1, 1), 199.0)
        local_k = torch.cat([torch.full((1, frame_tokens, 1, 1), float(value)) for value in range(10, 16)], dim=1)
        local_v = torch.cat([torch.full((1, frame_tokens, 1, 1), float(value)) for value in range(20, 26)], dim=1)

        key, value = assemble_noncontiguous_context(
            {"k": retrieved_k, "v": retrieved_v}, local_k, local_v, frame_tokens, current_frames=3)

        self.assertEqual(key.shape, local_k.shape)
        self.assertEqual(value.shape, local_v.shape)
        torch.testing.assert_close(key[:, :frame_tokens], local_k[:, :frame_tokens])
        torch.testing.assert_close(value[:, :frame_tokens], local_v[:, :frame_tokens])
        torch.testing.assert_close(key[:, frame_tokens:2 * frame_tokens], retrieved_k)
        torch.testing.assert_close(value[:, frame_tokens:2 * frame_tokens], retrieved_v)
        torch.testing.assert_close(key[:, 2 * frame_tokens:], local_k[:, 2 * frame_tokens:])
        torch.testing.assert_close(value[:, 2 * frame_tokens:], local_v[:, 2 * frame_tokens:])

    def test_two_history_frames_replace_both_recent_non_sink_frames(self):
        frame_tokens = 1
        retrieved_k = torch.tensor([[[[90.]], [[91.]]]])
        retrieved_v = torch.tensor([[[[190.]], [[191.]]]])
        local_k = torch.tensor([[[[10.]], [[20.]], [[30.]], [[40.]], [[50.]], [[60.]]]])
        local_v = local_k + 100

        key, value = assemble_noncontiguous_context(
            {"k": retrieved_k, "v": retrieved_v}, local_k, local_v, frame_tokens, current_frames=3)

        self.assertEqual(key.flatten().tolist(), [10.0, 90.0, 91.0, 40.0, 50.0, 60.0])
        self.assertEqual(value.flatten().tolist(), [110.0, 190.0, 191.0, 140.0, 150.0, 160.0])

    def test_retrieved_local_and_current_positions_are_contiguous(self):
        retrieved = block_relative_positions(0, 2)
        local = block_relative_positions(0, 3, prefix_frames=2)
        current = block_relative_positions(3, 1, prefix_frames=2)

        self.assertEqual(retrieved.tolist(), [0, 1])
        self.assertEqual(local.tolist(), [2, 3, 4])
        self.assertEqual(current.tolist(), [5])

    def test_rope_prefix_uses_the_local_position_after_retrieval(self):
        x = torch.randn(1, 2, 1, 12)
        grid = torch.tensor([[2, 1, 1]])
        freqs = torch.cat([rope_params(16, 4), rope_params(16, 4), rope_params(16, 4)], dim=1)

        prefixed = block_relativistic_rope(x, grid, freqs, prefix_frames=2)
        offset_reference = block_relativistic_rope(x, grid, freqs, start_frame=2)

        torch.testing.assert_close(prefixed, offset_reference)

    def test_disabled_prefix_returns_the_original_context_tensors(self):
        local_k = torch.randn(1, 6, 2, 8)
        local_v = torch.randn(1, 6, 2, 8)

        key, value = assemble_noncontiguous_context(None, local_k, local_v, frame_tokens=1, current_frames=3)

        self.assertIs(key, local_k)
        self.assertIs(value, local_v)

    def test_coherent_selection_uses_final_frame_from_each_newest_source_block(self):
        captured = {
            2: {"frame_ids": [3, 4, 5]},
            4: {"frame_ids": [9, 10, 11]},
        }

        frames = select_history_frame_refs(captured, [2, 4], retrieval_count=2, mode="coherent_history")

        self.assertEqual(frames, [
            {"source_block": 2, "frame_index": 2, "global_frame_id": 5},
            {"source_block": 4, "frame_index": 2, "global_frame_id": 11},
        ])
        self.assertEqual(
            select_history_frame_refs(captured, [2, 4], retrieval_count=1, mode="coherent_history"),
            [{"source_block": 4, "frame_index": 2, "global_frame_id": 11}],
        )

    def test_random_selection_is_distinct_and_seeded(self):
        captured = {2: {"frame_ids": [3, 4, 5]}, 4: {"frame_ids": [9, 10, 11]}}

        first = select_history_frame_refs(captured, [2, 4], retrieval_count=2, mode="random_history", random_seed=7)
        second = select_history_frame_refs(captured, [2, 4], retrieval_count=2, mode="random_history", random_seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len({frame["global_frame_id"] for frame in first}), 2)

    def test_oracle_selection_uses_the_requested_global_frame(self):
        captured = {
            3: {"frame_ids": [6, 7, 8]},
            6: {"frame_ids": [15, 16, 17]},
        }

        self.assertEqual(
            select_history_frame_refs(
                captured, [3, 6], retrieval_count=1, mode="same_entity_history",
                manual_frame_id=8),
            [{"source_block": 3, "frame_index": 2, "global_frame_id": 8}],
        )
        self.assertEqual(
            select_history_frame_refs(
                captured, [3, 6], retrieval_count=1, mode="wrong_entity_history",
                manual_frame_id=17),
            [{"source_block": 6, "frame_index": 2, "global_frame_id": 17}],
        )

    def test_two_frame_oracle_selection_preserves_the_manual_order(self):
        captured = {13: {"frame_ids": [36, 37, 38]}}

        self.assertEqual(
            select_history_frame_refs(
                captured, [13], retrieval_count=2, mode="same_entity_history",
                manual_frame_ids=[37, 38]),
            [
                {"source_block": 13, "frame_index": 1, "global_frame_id": 37},
                {"source_block": 13, "frame_index": 2, "global_frame_id": 38},
            ],
        )

    def test_captured_kv_is_cpu_resident_and_only_selected_frame_is_packed(self):
        raw_k = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
        raw_v = raw_k + 100
        caches = [{"noncontiguous_raw_k": raw_k, "noncontiguous_raw_v": raw_v}]

        layers = capture_clean_kv_to_cpu(caches)
        self.assertEqual(caches, [{}])
        self.assertEqual(layers[0]["k"].device.type, "cpu")
        self.assertEqual(layers[0]["v"].device.type, "cpu")
        self.assertNotEqual(layers[0]["k"].data_ptr(), raw_k.data_ptr())
        self.assertNotEqual(layers[0]["v"].data_ptr(), raw_v.data_ptr())
        captured = {3: {"frame_ids": [6, 7, 8], "layers": layers}}
        packed = select_history_kv(
            captured, [{"source_block": 3, "frame_index": 1, "global_frame_id": 7}],
            num_layers=1, frame_tokens=1)

        self.assertEqual(packed[0]["k"].shape, (1, 1, 2, 2))
        torch.testing.assert_close(packed[0]["k"], raw_k[:, 1:2])
        self.assertIs(materialize_retrieved_kv(packed[0], torch.device("cpu")), packed[0])

    def test_history_selection_does_not_advance_torch_rng(self):
        captured = {2: {"frame_ids": [3, 4, 5]}, 4: {"frame_ids": [9, 10, 11]}}
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        select_history_frame_refs(captured, [2, 4], retrieval_count=1, mode="random_history", random_seed=7)
        torch.testing.assert_close(torch.rand(3), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
