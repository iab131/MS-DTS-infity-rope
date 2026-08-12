"""CPU invariants for the minimal opt-in scene-local RoPE epoch."""

import math
import unittest
from unittest.mock import patch

import torch

from pipeline.causal_inference import apply_memory_transition
from wan.modules.causal_model import (
    CausalWanSelfAttention,
    block_relativistic_rope,
    recent_only_key_positions,
    scene_local_key_positions,
)
from wan.modules.model import rope_params


def freqs():
    return torch.cat([rope_params(64, 4), rope_params(64, 4), rope_params(64, 4)], dim=1)


def cpu_attention(q, k, v):
    scores = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(q.shape[-1])
    return torch.einsum("bhls,bshd->blhd", scores.softmax(-1), v)


def epoch_module_and_cache(enabled):
    module = CausalWanSelfAttention(dim=12, num_heads=1, local_attn_size=6, sink_size=1, qk_norm=False)
    for linear in (module.q, module.k, module.v, module.o):
        linear.weight.data.copy_(torch.eye(12))
        linear.bias.data.zero_()
    cache = {
        "k": torch.zeros(1, 6, 1, 12), "v": torch.zeros(1, 6, 1, 12),
        "global_end_index": torch.tensor([9]), "local_end_index": torch.tensor([0]),
        "scene_cut": True,
    }
    if enabled:
        cache.update({"scene_local_rope_epoch": True, "scene_local_rope_epoch_start_frame": 9})
    return module, cache


class SceneLocalRopeEpochTest(unittest.TestCase):
    def test_epoch_key_positions_are_coherent_before_and_after_rolling(self):
        self.assertEqual(scene_local_key_positions(0, 3, 3).tolist(), [0, 1, 2])
        self.assertEqual(scene_local_key_positions(3, 3, 6).tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(scene_local_key_positions(6, 3, 9).tolist(), list(range(9)))
        self.assertEqual(scene_local_key_positions(21, 3, 21).tolist(), [0, *range(4, 24)])

    def test_first_b_is_equivalent_but_epoch_sink_is_phase_zero(self):
        x = torch.randn(1, 3, 12)
        baseline_module, baseline_cache = epoch_module_and_cache(False)
        epoch_module, epoch_cache = epoch_module_and_cache(True)
        with patch("wan.modules.causal_model.attention", cpu_attention):
            baseline = baseline_module(x, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                                       kv_cache=baseline_cache, current_start=9)
            epoch = epoch_module(x, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                                 kv_cache=epoch_cache, current_start=9)

        torch.testing.assert_close(epoch, baseline, atol=1e-6, rtol=1e-6)
        expected_sink = block_relativistic_rope(
            x[:, :1].view(1, 1, 1, 12), torch.tensor([[1, 1, 1]]), freqs(), start_frame=0)
        torch.testing.assert_close(epoch_cache["k"][:, :1], expected_sink)
        self.assertEqual(epoch_cache["global_end_index"].item(), baseline_cache["global_end_index"].item())
        self.assertEqual(epoch_cache["local_end_index"].item(), baseline_cache["local_end_index"].item())

    def test_transition_metadata_only_activates_for_hard_no_sink_epoch(self):
        def cache():
            return {"k": torch.zeros(1, 3, 1, 2), "v": torch.zeros(1, 3, 1, 2),
                    "local_end_index": torch.tensor([3]), "scene_cut": False}

        hard = cache()
        apply_memory_transition([hard], [{"is_init": True}], 1, "transition_no_sink", False, 0.3,
                                True, torch.device("cpu"), scene_local_rope_epoch=True,
                                current_start_frame=9)
        self.assertTrue(hard["scene_local_rope_epoch"])
        self.assertEqual(hard["scene_local_rope_epoch_start_frame"], 9)

        soft = cache()
        apply_memory_transition([soft], [{"is_init": True}], 1, "transition_no_sink", False, 0.3,
                                False, torch.device("cpu"), scene_local_rope_epoch=True,
                                current_start_frame=9)
        self.assertFalse(soft["scene_local_rope_epoch"])

        retained = cache()
        apply_memory_transition([retained], [{"is_init": True}], 1, "sink_only", False, 0.3,
                                True, torch.device("cpu"), scene_local_rope_epoch=True,
                                current_start_frame=9)
        self.assertFalse(retained["scene_local_rope_epoch"])

    def test_second_b_uses_coherent_epoch_for_query_and_cached_keys(self):
        x1, x2 = torch.randn(1, 3, 12), torch.randn(1, 3, 12)
        module, cache = epoch_module_and_cache(True)
        seen = []

        def capture_attention(q, k, v):
            seen.append((q.detach().clone(), k.detach().clone()))
            return torch.zeros_like(q)

        with patch("wan.modules.causal_model.attention", capture_attention):
            module(x1, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                   kv_cache=cache, current_start=9)
            cache["scene_cut"] = False
            module(x2, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                   kv_cache=cache, current_start=12)

        query, key = seen[-1]
        expected_query = block_relativistic_rope(
            x2.view(1, 3, 1, 12), torch.tensor([[3, 1, 1]]), freqs(), start_frame=3)
        raw_context = torch.cat([x1, x2], dim=1).view(1, 6, 1, 12)
        expected_key = block_relativistic_rope(
            raw_context, torch.tensor([[6, 1, 1]]), freqs(), temporal_positions=torch.arange(6))
        expected_key[:, :1] = cache["k"][:, :1]
        torch.testing.assert_close(query, expected_query)
        torch.testing.assert_close(key, expected_key)
        self.assertEqual(cache["global_end_index"].item(), 15)
        self.assertEqual(cache["local_end_index"].item(), 6)

    def test_flag_off_leaves_transition_no_sink_rope_path_unchanged(self):
        x = torch.randn(1, 3, 12)
        absent_module, absent_cache = epoch_module_and_cache(False)
        false_module, false_cache = epoch_module_and_cache(False)
        false_cache["scene_local_rope_epoch"] = False
        with patch("wan.modules.causal_model.attention", cpu_attention):
            absent = absent_module(x, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                                   kv_cache=absent_cache, current_start=9)
            explicit_false = false_module(x, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                                        kv_cache=false_cache, current_start=9)

        torch.testing.assert_close(explicit_false, absent)
        torch.testing.assert_close(false_cache["k"], absent_cache["k"])

    def test_recent_only_no_sink_uses_raw_recent_keys_then_establishes_new_sink(self):
        old, current = torch.randn(1, 2, 12), torch.randn(1, 3, 12)
        module = CausalWanSelfAttention(dim=12, num_heads=1, local_attn_size=6, sink_size=1, qk_norm=False)
        for linear in (module.q, module.k, module.v, module.o):
            linear.weight.data.copy_(torch.eye(12))
            linear.bias.data.zero_()
        cache = {
            "k": torch.zeros(1, 6, 1, 12), "v": torch.zeros(1, 6, 1, 12),
            "global_end_index": torch.tensor([9]), "local_end_index": torch.tensor([2]),
            "scene_cut": True, "recent_only_no_sink": True,
            "recent_only_no_sink_finalize": True,
        }
        cache["k"][:, :2] = old.view(1, 2, 1, 12)
        cache["v"][:, :2] = old.view(1, 2, 1, 12)
        seen = []
        with patch("wan.modules.causal_model.attention", lambda q, k, v: seen.append(k.detach().clone()) or torch.zeros_like(q)):
            module(current, torch.tensor([3]), torch.tensor([[3, 1, 1]]), freqs(), None,
                   kv_cache=cache, current_start=9)

        expected_context = block_relativistic_rope(
            torch.cat([old, current], dim=1).view(1, 5, 1, 12), torch.tensor([[5, 1, 1]]),
            freqs(), temporal_positions=recent_only_key_positions(5, 3, True))
        torch.testing.assert_close(seen[0], expected_context)
        expected_sink = block_relativistic_rope(
            current[:, :1].view(1, 1, 1, 12), torch.tensor([[1, 1, 1]]), freqs(), start_frame=0)
        torch.testing.assert_close(cache["k"][:, :1], expected_sink)
        torch.testing.assert_close(cache["k"][:, 1:3], current[:, 1:].view(1, 2, 1, 12))
        self.assertEqual(cache["local_end_index"].item(), 3)
        self.assertFalse(cache["recent_only_no_sink"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
