#!/usr/bin/env python3
"""CPU checks for the opt-in attention-as-memory policy scaffold."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pipeline.causal_inference import (
    apply_memory_transition, capture_clean_memory_block, memory_context_order,
    fixed_grid_denoising_schedule, fixed_grid_memory_active, pack_fixed_grid_selective_memory,
    capture_subject_latent_memory, transplant_subject_latent_memory,
    latent_patch_clean_cache_input, latent_patch_cache_write_mask,
    record_transition_sink,
    transition_attention_context,
)
from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks
from pipeline.content_routing import MemoryPolicyEventLogger, retrieval_allowed, route_memory
from pipeline.memory_store import MemoryStore
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.causal_model import (
    CausalWanAttentionBlock, CausalWanModel, CausalWanSelfAttention,
    assemble_memory_context, memory_context_layout, memory_context_rope_positions,
    sparse_historical_rope,
)


def layer_kv(frame_values, frame_tokens=2):
    key = torch.cat([
        torch.full((1, frame_tokens, 1, 2), float(value)) for value in frame_values
    ], dim=1)
    return {"k": key, "v": key + 100}


class AttentionMemoryPolicyTest(unittest.TestCase):
    def test_persistent_latent_patch_enters_only_block_eight_clean_cache(self):
        baseline = torch.zeros(1, 3, 1, 2, 2)
        patched = torch.ones_like(baseline)

        self.assertIs(latent_patch_clean_cache_input(baseline, patched, True, 8), patched)
        self.assertIs(latent_patch_clean_cache_input(baseline, patched, True, 9), baseline)
        self.assertIs(latent_patch_clean_cache_input(baseline, patched, False, 8), baseline)

    def test_persistent_cache_erosion_writes_only_visible_patch_core(self):
        source = [0] * 1560
        target = [0] * 1560
        for row in range(5, 10):
            for column in range(10, 15):
                source[row * 52 + column] = target[row * 52 + column] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({"height": 30, "width": 52,
                       "source_masks": {"6": source, "7": source},
                       "target_subject_mask": target}, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        masks = FixedGridMemoryMasks.from_json(handle.name)
        source_latents = torch.ones(1, 8, 1, 60, 104)
        baseline = torch.zeros(1, 3, 1, 60, 104)
        memory = capture_subject_latent_memory(source_latents, masks)
        patched, _ = transplant_subject_latent_memory(baseline, memory, masks)

        erode1, audit1 = latent_patch_cache_write_mask(memory, masks, baseline, 1)
        erode2, audit2 = latent_patch_cache_write_mask(memory, masks, baseline, 2)
        cache1 = latent_patch_clean_cache_input(baseline, patched, True, 8, erode1)
        cache2 = latent_patch_clean_cache_input(baseline, patched, True, 8, erode2)

        self.assertEqual(audit1["token_counts"], [9, 9, 9])
        self.assertEqual(audit2["token_counts"], [1, 1, 1])
        self.assertEqual(audit1["latent_cell_counts"], [36, 36, 36])
        self.assertEqual(audit2["latent_cell_counts"], [4, 4, 4])
        self.assertTrue(torch.equal(
            cache1, torch.where(erode1[None, :, None], patched, baseline)))
        self.assertTrue(torch.equal(
            cache2, torch.where(erode2[None, :, None], patched, baseline)))

    def test_subject_latent_memory_transplants_only_supported_target_cells(self):
        source_6 = [0] * 1560
        source_6[0] = 1
        source_7 = [0] * 1560
        source_7[0] = 1
        target = [0] * 1560
        target[0] = target[1] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({"height": 30, "width": 52,
                       "source_masks": {"6": source_6, "7": source_7},
                       "target_subject_mask": target}, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        masks = FixedGridMemoryMasks.from_json(handle.name)
        source_latents = torch.zeros(1, 8, 1, 60, 104)
        source_latents[:, 6].fill_(10)
        source_latents[:, 7].fill_(20)
        baseline = torch.ones(1, 3, 1, 60, 104)

        memory = capture_subject_latent_memory(source_latents, masks)
        patched, audit = transplant_subject_latent_memory(baseline, memory, masks)

        self.assertEqual([item["content"].device.type for item in memory], ["cpu", "cpu"])
        self.assertEqual(patched[:, 0, :, :2, :2].flatten().tolist(), [10.0] * 4)
        self.assertEqual(patched[:, 1, :, :2, :2].flatten().tolist(), [15.0] * 4)
        self.assertEqual(patched[:, 2, :, :2, :2].flatten().tolist(), [20.0] * 4)
        self.assertTrue(torch.equal(patched[:, :, :, :2, 2:4], baseline[:, :, :, :2, 2:4]))
        self.assertTrue(audit["outside_target_equal"])
        self.assertEqual(audit["supported_token_counts"], [1, 1, 1])

    def test_affine_subject_latent_memory_aligns_scaled_source_bbox_to_target(self):
        source_6 = [0] * 1560
        source_6[1 * 52 + 1] = 1
        source_7 = [0] * 1560
        source_7[1 * 52 + 1] = 1
        target = [0] * 1560
        target[5 * 52 + 8] = target[5 * 52 + 9] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({"height": 30, "width": 52,
                       "source_masks": {"6": source_6, "7": source_7},
                       "target_subject_mask": target}, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        masks = FixedGridMemoryMasks.from_json(handle.name)
        source_latents = torch.zeros(1, 8, 1, 60, 104)
        source_latents[:, 6].fill_(10)
        source_latents[:, 7].fill_(20)
        baseline = torch.ones(1, 3, 1, 60, 104)

        memory = capture_subject_latent_memory(source_latents, masks)
        patched, audit = transplant_subject_latent_memory(
            baseline, memory, masks, affine_align=True)

        target_pixels = (slice(None), slice(None), slice(None), slice(10, 12), slice(16, 20))
        torch.testing.assert_close(
            patched[:, 0][target_pixels[0], target_pixels[2], target_pixels[3], target_pixels[4]],
            torch.full((1, 1, 2, 4), 10.0), atol=1e-4, rtol=0)
        torch.testing.assert_close(
            patched[:, 1][target_pixels[0], target_pixels[2], target_pixels[3], target_pixels[4]],
            torch.full((1, 1, 2, 4), 15.0), atol=1e-4, rtol=0)
        torch.testing.assert_close(
            patched[:, 2][target_pixels[0], target_pixels[2], target_pixels[3], target_pixels[4]],
            torch.full((1, 1, 2, 4), 20.0), atol=1e-4, rtol=0)
        self.assertTrue(audit["outside_target_equal"])
        self.assertEqual(audit["supported_token_counts"], [2, 2, 2])
        self.assertEqual(audit["source_to_target_affines"]["6"]["scale_xy"], [3.0, 1.0])

    def test_affine_subject_latent_memory_uses_fp32_geometry_for_bfloat16_latents(self):
        masks = FixedGridMemoryMasks.from_json(
            Path(__file__).resolve().parents[1] /
            "docs/attention_memory_policy_fixed_grid_masks_20260809.json")
        source_latents = torch.zeros(1, 8, 1, 60, 104, dtype=torch.bfloat16)
        baseline = torch.zeros(1, 3, 1, 60, 104, dtype=torch.bfloat16)

        _, audit = transplant_subject_latent_memory(
            baseline, capture_subject_latent_memory(source_latents, masks), masks, affine_align=True)

        self.assertEqual(audit["supported_latent_cell_counts"], [1301, 1301, 1301])

    def test_fixed_grid_timestep_gate_uses_observed_execution_order(self):
        schedule = fixed_grid_denoising_schedule([1000.0, 937.5, 833.3333129882812, 625.0])
        self.assertEqual(schedule["execution_timesteps"], [1000.0, 937.5, 833.3333129882812, 625.0])
        self.assertEqual(schedule["noise_order"], "high_to_low")
        self.assertEqual(
            [fixed_grid_memory_active("all", True, index, 4) for index in range(4)],
            [True, True, True, True])
        self.assertEqual(
            [fixed_grid_memory_active("latest_1", False, index, 4) for index in range(4)],
            [False, False, False, True])
        self.assertEqual(
            [fixed_grid_memory_active("latest_2", False, index, 4) for index in range(4)],
            [False, False, True, True])
        self.assertEqual(
            [fixed_grid_memory_active("clean_only", True, index, 4) for index in range(4)],
            [False, False, False, False])
        self.assertFalse(fixed_grid_memory_active("latest_2", False, clean_pass=True))
        self.assertTrue(fixed_grid_memory_active("latest_2", True, clean_pass=True))
        self.assertTrue(fixed_grid_memory_active("clean_only", True, clean_pass=True))

    def test_selective_pack_preserves_source_indices_slots_and_three_target_frames(self):
        source_6 = [0] * 1560
        source_6[1] = source_6[3] = 1
        source_7 = [0] * 1560
        source_7[2] = 1
        target = [0] * 1560
        target[4] = target[5] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({
                "height": 30, "width": 52,
                "source_masks": {"6": source_6, "7": source_7},
                "target_subject_mask": target,
            }, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        masks = FixedGridMemoryMasks.from_json(handle.name)
        store = MemoryStore(
            frame_tokens=1560, descriptor_layers=[0], injection_layers=[0, 2], memory_budget=10)
        frame_6 = torch.arange(1560, dtype=torch.float32).view(1, 1560, 1, 1)
        frame_7 = frame_6 + 10000
        layer = {"k": torch.cat([frame_6, frame_7], dim=1),
                 "v": torch.cat([frame_6 + 100, frame_7 + 100], dim=1)}
        store.add_clean_block(0, [6, 7], [layer, layer, layer])

        packed = pack_fixed_grid_selective_memory(
            store, masks, "subject_to_subject", current_frames=3, num_layers=3)

        self.assertIsNone(packed[1])
        group = packed[0][0]
        self.assertEqual(group["source_frame_ids"], [6, 7])
        self.assertEqual(group["source_token_indices"], {6: [1, 3], 7: [2]})
        self.assertEqual(group["original_token_indices"].tolist(), [1, 3, 2])
        self.assertEqual(group["temporal_slots"].tolist(), [1, 1, 2])
        self.assertEqual(group["query_indices"].tolist(), [4, 5, 1564, 1565, 3124, 3125])
        self.assertEqual(group["historical_key"].flatten().tolist(), [1.0, 3.0, 10002.0])
        self.assertEqual(group["historical_value"].flatten().tolist(), [101.0, 103.0, 10102.0])
        self.assertEqual(packed[2][0]["temporal_slots"].tolist(), [1, 1, 2])

    def test_compact_entity_pack_mean_pools_each_subject_frame_without_spatial_indices(self):
        source_6 = [0] * 1560
        source_6[1] = source_6[3] = 1
        source_7 = [0] * 1560
        source_7[2] = 1
        target = [0] * 1560
        target[4] = target[5] = 1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump({
                "height": 30, "width": 52,
                "source_masks": {"6": source_6, "7": source_7},
                "target_subject_mask": target,
            }, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        masks = FixedGridMemoryMasks.from_json(handle.name)
        store = MemoryStore(
            frame_tokens=1560, descriptor_layers=[0], injection_layers=[0], memory_budget=10)
        frame_6 = torch.arange(1560, dtype=torch.float32).view(1, 1560, 1, 1)
        frame_7 = frame_6 + 10000
        layer = {"k": torch.cat([frame_6, frame_7], dim=1),
                 "v": torch.cat([frame_6 + 100, frame_7 + 100], dim=1)}
        store.add_clean_block(0, [6, 7], [layer])

        group = pack_fixed_grid_selective_memory(
            store, masks, "compact_entity_memory", current_frames=3, num_layers=1)[0][0]

        self.assertEqual(group["position_mode"], "temporal_only_neutral_spatial")
        self.assertEqual(group["source_token_counts"], {6: 2, 7: 1})
        self.assertEqual(group["temporal_slots"].tolist(), [1, 2])
        self.assertNotIn("source_token_indices", group)
        self.assertNotIn("original_token_indices", group)
        self.assertEqual(group["historical_key"].flatten().tolist(), [2.0, 10002.0])
        self.assertEqual(group["historical_value"].flatten().tolist(), [102.0, 10102.0])
        self.assertEqual(group["query_indices"].tolist(), [4, 5, 1564, 1565, 3124, 3125])

    def test_selective_memory_is_optional_through_every_model_interface(self):
        for function in (
                WanDiffusionWrapper.forward,
                CausalWanModel._forward_inference,
                CausalWanAttentionBlock.forward,
                CausalWanSelfAttention.forward):
            parameter = inspect.signature(function).parameters["selective_memory"]
            self.assertIsNone(parameter.default)

    def test_self_attention_ropes_raw_sparse_history_before_grouped_attention(self):
        module = CausalWanSelfAttention(
            dim=2, num_heads=1, local_attn_size=6, sink_size=1, qk_norm=False)
        for linear in (module.q, module.k, module.v, module.o):
            linear.weight.data.copy_(torch.eye(2))
            linear.bias.data.zero_()
        cache = {
            "k": torch.zeros(1, 6, 1, 2),
            "v": torch.zeros(1, 6, 1, 2),
            "global_end_index": torch.tensor([0]),
            "local_end_index": torch.tensor([0]),
        }
        group = {
            "query_indices": torch.tensor([0]),
            "historical_key": torch.tensor([[[[1.0, 0.0]]]]),
            "historical_value": torch.tensor([[[[2.0, 0.0]]]]),
            "original_token_indices": torch.tensor([0]),
            "temporal_slots": torch.tensor([1]),
        }
        calls = []

        def cpu_attention(query, key, value):
            calls.append((query.shape[1], key.detach().clone()))
            return torch.zeros_like(query)

        with patch("wan.modules.causal_model.attention", cpu_attention), \
                patch("wan.modules.causal_model.sparse_historical_rope",
                      wraps=sparse_historical_rope) as sparse_rope:
            module(
                torch.tensor([[[1.0, 0.0]]]), torch.tensor([1]),
                torch.tensor([[1, 1, 1]]),
                torch.polar(torch.ones(3, 1), torch.zeros(3, 1)), None,
                kv_cache=cache, selective_memory=[group])

        sparse_rope.assert_called_once()
        self.assertEqual([length for length, _ in calls], [1, 1])
        torch.testing.assert_close(calls[1][1], group["historical_key"])

    def test_clean_capture_uses_the_active_scene_index(self):
        store = MemoryStore(frame_tokens=1, descriptor_layers=[0], injection_layers=[0], memory_budget=10)

        capture_clean_memory_block(
            store, scene_index=2, current_start_frame=6, current_num_frames=1,
            clean_layers=[layer_kv([3], 1)])

        self.assertEqual([(entry.scene_id, entry.frame_id) for entry in store.entries], [(2, 6)])

    def test_store_keeps_clean_frames_on_cpu_and_routes_top_similarity(self):
        store = MemoryStore(frame_tokens=2, descriptor_layers=[0], memory_budget=10)
        store.add_clean_block(
            scene_id=1, frame_ids=[4, 5], layers=[layer_kv([1, 3])])

        self.assertEqual(store.frame_ids, [4, 5])
        self.assertEqual(store.entries[0].layers[0]["k"].device.type, "cpu")
        self.assertNotEqual(store.entries[0].layers[0]["k"].data_ptr(), store.entries[1].layers[0]["k"].data_ptr())

        routed = route_memory(
            store, {0: torch.full((1, 2), 3.0)}, k=1, exclude_frame_ids={4})

        self.assertEqual([item.frame_id for item in routed.entries], [5])
        self.assertEqual(len(routed.scores), 1)
        self.assertGreater(store.entries[1].utility, 0.0)

    def test_injection_layers_are_independent_from_descriptor_layers(self):
        store = MemoryStore(
            frame_tokens=1, descriptor_layers=[0], injection_layers=[1], memory_budget=10)
        store.add_clean_block(
            scene_id=1, frame_ids=[4], layers=[layer_kv([1], 1), layer_kv([2], 1), layer_kv([3], 1)])

        self.assertEqual(set(store.entries[0].descriptors), {0})
        self.assertEqual(set(store.entries[0].layers), {1})
        packed = store.pack_kv(store.entries, num_layers=3)
        self.assertIsNone(packed[0])
        self.assertIsNone(packed[2])
        torch.testing.assert_close(packed[1]["k"], torch.full((1, 1, 1, 2), 2.0))

    def test_clean_capture_clones_injection_kv_before_cpu_offload(self):
        source = layer_kv([2], 1)
        source["k"].requires_grad_()
        store = MemoryStore(frame_tokens=1, descriptor_layers=[0], injection_layers=[0], memory_budget=10)
        store.add_clean_block(scene_id=1, frame_ids=[4], layers=[source])

        captured = store.entries[0].layers[0]["k"]
        self.assertFalse(captured.requires_grad)
        self.assertNotEqual(captured.data_ptr(), source["k"].data_ptr())

    def test_manual_routing_preserves_requested_order_without_rng(self):
        store = MemoryStore(frame_tokens=1, descriptor_layers=[0], memory_budget=10)
        store.add_clean_block(scene_id=1, frame_ids=[4, 5, 6], layers=[layer_kv([1, 2, 3], 1)])

        routed = route_memory(
            store, {0: torch.ones(1, 2)}, k=2, manual_frame_ids=[6, 4])

        self.assertEqual([item.frame_id for item in routed.entries], [6, 4])
        self.assertEqual(routed.scores, [None, None])

    def test_manual_retrieval_remains_available_when_transition_auto_routing_is_disabled(self):
        self.assertEqual(
            retrieval_allowed(True, is_transition_block=True, transition_auto_retrieval=False,
                              manual_frame_ids=[4, 5], manual_target_blocks={8}, block_number=8),
            (True, "manual_override"))
        self.assertEqual(
            retrieval_allowed(True, is_transition_block=True, transition_auto_retrieval=False,
                              manual_frame_ids=None, manual_target_blocks=None, block_number=8),
            (False, "automatic_transition_disabled"))
        self.assertEqual(
            retrieval_allowed(True, is_transition_block=False, transition_auto_retrieval=False,
                              manual_frame_ids=[4, 5], manual_target_blocks={8}, block_number=9),
            (False, "manual_lifetime_expired"))

    def test_manual_retrieval_lifetime_starts_at_the_first_selected_block(self):
        common = dict(
            retrieval_enabled=True, is_transition_block=False, transition_auto_retrieval=False,
            manual_frame_ids=[4, 5], manual_target_blocks={8},
        )
        self.assertEqual(
            retrieval_allowed(**common, block_number=8, manual_retrieval_lifetime="pulse_1"),
            (True, "manual_override"))
        self.assertEqual(
            retrieval_allowed(**common, block_number=9, manual_retrieval_lifetime="pulse_1"),
            (False, "manual_lifetime_expired"))
        self.assertEqual(
            retrieval_allowed(**common, block_number=9, manual_retrieval_lifetime="pulse_2"),
            (True, "manual_override"))
        self.assertEqual(
            retrieval_allowed(**common, block_number=10, manual_retrieval_lifetime="pulse_2"),
            (False, "manual_lifetime_expired"))
        self.assertEqual(
            retrieval_allowed(**common, block_number=10, manual_retrieval_lifetime="persistent"),
            (True, "manual_override"))

    def test_archive_compresses_top_utility_frames_and_consolidation_keeps_diversity(self):
        store = MemoryStore(frame_tokens=1, descriptor_layers=[0], memory_budget=2)
        store.add_clean_block(scene_id=7, frame_ids=[1, 2, 3], layers=[layer_kv([1, 3, -1], 1)])
        store.entries[0].utility = 1.0
        store.entries[1].utility = 3.0
        store.entries[2].utility = 0.5

        archive = store.archive_scene(7, top_m=2)

        self.assertEqual(archive.scene_id, 7)
        torch.testing.assert_close(archive.layers[0]["k"], torch.full((1, 1, 1, 2), 2.5))
        action = store.consolidate(target_budget=2, diversity_threshold=0.99)
        self.assertTrue(action["performed"])
        self.assertEqual(len(store), 2)
        self.assertIn(2, store.frame_ids)
        self.assertIn(3, store.frame_ids)

    def test_context_modes_keep_sink_special_and_positions_explicit(self):
        local = torch.tensor([[[[10.]], [[20.]], [[30.]], [[40.]], [[50.]], [[60.]]]])
        retrieved = {"k": torch.tensor([[[[90.]], [[91.]]]]), "v": torch.tensor([[[[190.]], [[191.]]]])}

        key, value = assemble_memory_context(
            retrieved, local, local + 100, frame_tokens=1, current_frames=3, mode="replace_recent")
        self.assertEqual(key.flatten().tolist(), [10.0, 90.0, 91.0, 40.0, 50.0, 60.0])
        self.assertEqual(value.flatten().tolist(), [110.0, 190.0, 191.0, 140.0, 150.0, 160.0])

        key, _ = assemble_memory_context(
            {"k": retrieved["k"][:, :1], "v": retrieved["v"][:, :1]},
            local, local + 100, frame_tokens=1, current_frames=3, mode="prepend")
        self.assertEqual(key.flatten().tolist(), [10.0, 90.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(memory_context_layout(2, 2, 3, "prepend"), {
            "sink_frames": 1, "retrieved_frames": 2, "local_frames": 2,
            "current_frames": 3, "positions": [0, 1, 2, 3, 4, 5, 6, 7],
        })

    def test_matched_delayed_recall_logs_actual_rope_coordinates(self):
        self.assertEqual(
            memory_context_rope_positions(
                retrieved_frames=2, local_frames=2, current_frames=3,
                mode="replace_recent", current_start_frame=21, scene_cut=False),
            {
                "sink_key_position": "preserved",
                "history_key_positions": [1, 2],
                "local_key_positions": [],
                "current_key_positions": [3, 4, 5],
                "current_query_positions": [21, 22, 23],
            },
        )

    def test_matched_replace_recent_log_omits_replaced_local_ids(self):
        self.assertEqual(
            memory_context_order(
                sink_frame_id=18, history_frame_ids=[6, 7], local_frame_ids=[19, 20],
                current_frame_ids=[21, 22, 23], retained_local_frames=0),
            ["sink:18", "history:6", "history:7", "current:21", "current:22", "current:23"],
        )

    def test_transition_retention_decay_preserves_sink_and_resets_cross_attention(self):
        cache = {
            "k": torch.tensor([[[[10.]], [[20.]], [[30.]], [[40.]], [[50.]], [[60.]]]]),
            "v": torch.tensor([[[[110.]], [[120.]], [[130.]], [[140.]], [[150.]], [[160.]]]]),
            "local_end_index": torch.tensor([6]), "scene_cut": False,
        }
        cross = {"is_init": True}

        event = apply_memory_transition(
            [cache], [cross], frame_tokens=1, retention="sink+1", decay=True,
            decay_beta=0.3, scene_cut=True, device=torch.device("cpu"))

        self.assertEqual(cache["k"].flatten().tolist()[:2], [10.0, 18.0])
        self.assertEqual(cache["v"].flatten().tolist()[:2], [110.0, 48.0])
        self.assertEqual(cache["local_end_index"].item(), 2)
        self.assertTrue(cache["scene_cut"])
        self.assertFalse(cross["is_init"])
        self.assertEqual(event["retained_non_sink_frames"], 1)

    def test_transition_no_sink_excludes_previous_context_for_one_cut_block(self):
        cache = {
            "k": torch.tensor([[[[10.]], [[20.]], [[30.]]]]),
            "v": torch.tensor([[[[110.]], [[120.]], [[130.]]]]),
            "local_end_index": torch.tensor([3]), "scene_cut": False,
        }
        cross = {"is_init": True}

        event = apply_memory_transition(
            [cache], [cross], frame_tokens=1, retention="transition_no_sink", decay=False,
            decay_beta=0.3, scene_cut=True, device=torch.device("cpu"))

        self.assertEqual(cache["local_end_index"].item(), 0)
        self.assertTrue(event["persistent_sink_excluded"])
        self.assertFalse(cross["is_init"])
        self.assertEqual(
            transition_attention_context(current_start_frame=9, current_num_frames=3,
                                         retention="transition_no_sink", scene_cut=True),
            {"ordering": ["current:9", "current:10", "current:11"],
             "rope_temporal_positions": [45, 46, 47], "total_frames": 3, "total_tokens": 3},
        )

    def test_recent_only_no_sink_retains_two_raw_frames_without_the_sink(self):
        cache = {
            "k": torch.tensor([[[[10.]], [[20.]], [[30.]], [[40.]], [[50.]], [[60.]]]]),
            "v": torch.tensor([[[[110.]], [[120.]], [[130.]], [[140.]], [[150.]], [[160.]]]]),
            "local_end_index": torch.tensor([6]), "scene_cut": False,
        }
        cross = {"is_init": True}

        event = apply_memory_transition(
            [cache], [cross], frame_tokens=1, retention="recent_only_no_sink", decay=False,
            decay_beta=0.3, scene_cut=True, device=torch.device("cpu"))

        self.assertEqual(cache["k"].flatten().tolist()[:2], [50.0, 60.0])
        self.assertEqual(cache["v"].flatten().tolist()[:2], [150.0, 160.0])
        self.assertEqual(cache["local_end_index"].item(), 2)
        self.assertTrue(cache["recent_only_no_sink"])
        self.assertTrue(event["persistent_sink_excluded"])
        self.assertEqual(
            transition_attention_context(current_start_frame=9, current_num_frames=3,
                                         retention="recent_only_no_sink", scene_cut=True),
            {"ordering": ["local:7", "local:8", "current:9", "current:10", "current:11"],
             "rope_temporal_positions": [1, 2, 45, 46, 47], "total_frames": 5,
             "total_tokens": 5},
        )

    def test_transition_context_reports_the_new_scene_sink_frame_id(self):
        caches = [{}, {}]
        self.assertEqual(
            record_transition_sink(caches, current_start_frame=18, retention="transition_no_sink"), 18)
        self.assertEqual([cache["persistent_sink_frame_id"] for cache in caches], [18, 18])
        ordinary_cache = {}
        self.assertIsNone(record_transition_sink(
            [ordinary_cache], current_start_frame=18, retention="sink_only"))
        self.assertNotIn("persistent_sink_frame_id", ordinary_cache)
        self.assertEqual(
            transition_attention_context(current_start_frame=21, current_num_frames=3,
                                         retention="sink_only", scene_cut=False, sink_frame_id=18)["ordering"],
            ["sink:18", "current:21", "current:22", "current:23"],
        )

    def test_event_log_captures_component_configuration(self):
        with tempfile.NamedTemporaryFile() as handle:
            logger = MemoryPolicyEventLogger(handle.name)
            logger.write("config", {"k": 2, "context_mode": "prepend", "manual_target_blocks": {7}})
            logger.write("retrieval", {"frame_ids": [4, 5], "scores": [0.9, 0.8]})
            rows = logger.read_rows()

        self.assertEqual([row["event"] for row in rows], ["config", "retrieval"])
        self.assertEqual(rows[0]["manual_target_blocks"], [7])
        self.assertEqual(rows[1]["frame_ids"], [4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
