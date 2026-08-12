from typing import List, Optional, Tuple
from random import Random
import torch
import torch.nn.functional as F
import re

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline.memory_store import MemoryStore
from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks
from pipeline.content_routing import MemoryPolicyEventLogger, retrieval_allowed, route_memory
from wan.modules.causal_model import memory_context_layout, memory_context_rope_positions, recent_only_key_positions

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


def apply_memory_transition(kv_caches, crossattn_caches, frame_tokens, retention,
                            decay, decay_beta, scene_cut, device, cross_attention_reset=True,
                            scene_local_rope_epoch=False, current_start_frame=None):
    """Apply an opt-in scene-boundary local policy."""
    persistent_sink_excluded = retention in {"transition_no_sink", "recent_only_no_sink"}
    retained = {"sink_only": 0, "sink+1": 1, "sink+2": 2,
                "recent_only_no_sink": 2, "transition_no_sink": 0}.get(retention)
    if retained is None:
        raise ValueError(f"unsupported memory local retention: {retention}")
    if not 0.0 <= decay_beta <= 1.0:
        raise ValueError("memory decay beta must be in [0, 1]")
    actual_retained = 0
    epoch_active = bool(scene_local_rope_epoch and scene_cut and retention == "transition_no_sink")
    if epoch_active and current_start_frame is None:
        raise ValueError("scene-local RoPE epoch requires the current start frame")
    for cache, cross_cache in zip(kv_caches, crossattn_caches):
        local_end = cache["local_end_index"].item()
        available = max(0, local_end // frame_tokens - 1)
        keep = min(retained, available)
        actual_retained = keep
        sink_end = 0 if retention == "recent_only_no_sink" else frame_tokens
        retained_end = sink_end + keep * frame_tokens
        if keep:
            source_start = local_end - keep * frame_tokens
            for name in ("k", "v"):
                cache[name][:, sink_end:retained_end] = cache[name][:, source_start:local_end].clone()
                if decay:
                    cache[name][:, sink_end:retained_end].mul_(decay_beta)
        cache["local_end_index"] = torch.tensor(
            [retained_end if retention == "recent_only_no_sink" else
             (0 if persistent_sink_excluded else retained_end)], dtype=torch.long, device=device)
        cache["recent_only_no_sink"] = retention == "recent_only_no_sink"
        cache.pop("recent_only_no_sink_finalize", None)
        cache["scene_cut"] = scene_cut
        cache["scene_local_rope_epoch"] = epoch_active
        if epoch_active:
            cache["scene_local_rope_epoch_start_frame"] = int(current_start_frame)
        else:
            cache.pop("scene_local_rope_epoch_start_frame", None)
        if cross_attention_reset:
            cross_cache["is_init"] = False
    return {
        "retention": retention,
        "retained_non_sink_frames": actual_retained,
        "decay": decay,
        "decay_beta": decay_beta if decay else None,
        "scene_cut": scene_cut,
        "cross_attention_reset": cross_attention_reset,
        "persistent_sink_excluded": persistent_sink_excluded,
        "scene_local_rope_epoch": epoch_active,
        "scene_local_rope_epoch_start_frame": int(current_start_frame) if epoch_active else None,
    }


def transition_attention_context(current_start_frame, current_num_frames, retention, scene_cut, frame_tokens=1,
                                 sink_frame_id=0, scene_local_rope_epoch=False):
    """Describe the actual first-block context after an opt-in transition policy."""
    retained = {"sink_only": 0, "sink+1": 1, "sink+2": 2,
                "recent_only_no_sink": 2, "transition_no_sink": 0}.get(retention)
    if retained is None:
        raise ValueError(f"unsupported memory local retention: {retention}")
    ordering = ([] if retention in {"transition_no_sink", "recent_only_no_sink"} else [f"sink:{sink_frame_id}"]) + \
        [f"local:{frame_id}" for frame_id in range(current_start_frame - retained, current_start_frame)] + \
        [f"current:{frame_id}" for frame_id in range(current_start_frame, current_start_frame + current_num_frames)]
    current_positions = list(range(current_num_frames)) if scene_local_rope_epoch else \
        (list(range(45, 45 + current_num_frames)) if scene_cut else \
        list(range(len(ordering) - current_num_frames, len(ordering)))
        )
    positions = (recent_only_key_positions(len(ordering), current_num_frames, scene_cut).tolist()
                 if retention == "recent_only_no_sink" else
                 list(range(len(ordering) - current_num_frames)) + current_positions)
    return {
        "ordering": ordering,
        "rope_temporal_positions": positions,
        "total_frames": len(ordering),
        "total_tokens": len(ordering) * frame_tokens,
    }


def record_transition_sink(kv_caches, current_start_frame, retention):
    """Promote the first clean new-scene frame after a no-old-context transition."""
    if retention not in {"transition_no_sink", "recent_only_no_sink"}:
        return None
    for cache in kv_caches:
        cache["persistent_sink_frame_id"] = current_start_frame
    return current_start_frame


def memory_context_order(sink_frame_id, history_frame_ids, local_frame_ids, current_frame_ids,
                         retained_local_frames):
    """Label the assembled transient context without exposing replaced local slots."""
    retained_local_ids = local_frame_ids[-retained_local_frames:] if retained_local_frames else []
    return ([f"sink:{sink_frame_id}"] +
            [f"history:{frame_id}" for frame_id in history_frame_ids] +
            [f"local:{frame_id}" for frame_id in retained_local_ids] +
            [f"current:{frame_id}" for frame_id in current_frame_ids])


def select_history_frame_refs(
        captured_source_kv, source_blocks, retrieval_count, mode, random_seed=0,
        manual_frame_id=None, manual_frame_ids=None):
    """Select individual latent frames without perturbing the generation RNG."""
    candidates = [
        {"source_block": block, "frame_index": index, "global_frame_id": frame_id}
        for block in source_blocks
        for index, frame_id in enumerate(captured_source_kv[block]["frame_ids"])
    ]
    if retrieval_count > len(candidates):
        raise ValueError("not enough captured source frames for retrieval")
    if mode in {"same_entity_history", "wrong_entity_history"}:
        selected_ids = manual_frame_ids if manual_frame_ids is not None else [manual_frame_id]
        if len(selected_ids) != retrieval_count or len(set(selected_ids)) != retrieval_count:
            raise ValueError("oracle history requires distinct manually selected frames")
        candidates_by_id = {candidate["global_frame_id"]: candidate for candidate in candidates}
        if any(frame_id not in candidates_by_id for frame_id in selected_ids):
            raise ValueError(f"manual history frames {selected_ids} were not captured")
        return [candidates_by_id[frame_id] for frame_id in selected_ids]
    if mode == "coherent_history":
        selected = []
        for offset in range(max(len(captured_source_kv[block]["frame_ids"]) for block in source_blocks)):
            for block in reversed(source_blocks):
                frame_index = len(captured_source_kv[block]["frame_ids"]) - 1 - offset
                if frame_index >= 0:
                    selected.append({
                        "source_block": block,
                        "frame_index": frame_index,
                        "global_frame_id": captured_source_kv[block]["frame_ids"][frame_index],
                    })
                    if len(selected) == retrieval_count:
                        return sorted(selected, key=lambda frame: frame["global_frame_id"])
    if mode == "random_history":
        return sorted(Random(random_seed).sample(candidates, retrieval_count), key=lambda frame: frame["global_frame_id"])
    raise ValueError(f"unsupported non-contiguous mode: {mode}")


def capture_clean_kv_to_cpu(kv_caches):
    """Pop source clean-pass KV and retain it off-GPU without changing the cache."""
    return [
        {
            "k": cache.pop("noncontiguous_raw_k").to(device="cpu", copy=True),
            "v": cache.pop("noncontiguous_raw_v").to(device="cpu", copy=True),
        }
        for cache in kv_caches
    ]


def select_history_kv(captured_source_kv, history_refs, num_layers, frame_tokens):
    """Pack only selected CPU-resident historical frames for per-layer transfer."""
    return [
        {
            "k": torch.cat([
                captured_source_kv[frame["source_block"]]["layers"][layer]["k"][:,
                frame["frame_index"] * frame_tokens:(frame["frame_index"] + 1) * frame_tokens]
                for frame in history_refs
            ], dim=1),
            "v": torch.cat([
                captured_source_kv[frame["source_block"]]["layers"][layer]["v"][:,
                frame["frame_index"] * frame_tokens:(frame["frame_index"] + 1) * frame_tokens]
                for frame in history_refs
            ], dim=1),
        }
        for layer in range(num_layers)
    ]


def local_query_descriptors(kv_caches, descriptor_layers, frame_tokens):
    """Mean-pool the latest raw non-sink clean K from configured layers."""
    descriptors = {}
    for layer in descriptor_layers:
        cache = kv_caches[layer]
        local_end = cache["local_end_index"].item()
        if local_end <= frame_tokens:
            continue
        descriptors[layer] = cache["k"][0, local_end - frame_tokens:local_end].detach().float().mean(dim=0).cpu()
    return descriptors


def capture_clean_memory_block(memory_store, scene_index, current_start_frame, current_num_frames, clean_layers):
    """Store one clean block under the scene currently being generated."""
    return memory_store.add_clean_block(
        scene_index, list(range(current_start_frame, current_start_frame + current_num_frames)), clean_layers)


def fixed_grid_denoising_schedule(timesteps):
    """Describe the actual few-step execution order without assuming step labels."""
    values = [float(timestep) for timestep in timesteps]
    if all(left > right for left, right in zip(values, values[1:])):
        order = "high_to_low"
    elif all(left < right for left, right in zip(values, values[1:])):
        order = "low_to_high"
    else:
        order = "non_monotonic"
    return {"execution_timesteps": values, "noise_order": order}


def fixed_grid_memory_active(denoising_steps, clean_pass_enabled, index=None,
                              total_steps=None, clean_pass=False):
    """Keep fixed-grid history on the requested final denoising calls only."""
    if clean_pass:
        return clean_pass_enabled
    if index is None or total_steps is None:
        raise ValueError("denoising index and step count are required")
    retained = {"all": total_steps, "latest_1": 1, "latest_2": 2, "clean_only": 0}.get(denoising_steps)
    if retained is None:
        raise ValueError("unsupported fixed-grid denoising-step selection")
    return index >= total_steps - retained


def _lift_fixed_grid_mask(mask, latent_height, latent_width, device):
    if (latent_height, latent_width) != (60, 104):
        raise ValueError("manual subject latent oracle requires 60x104 VAE latents")
    return torch.tensor(mask, device=device, dtype=torch.bool).view(30, 52).repeat_interleave(2, 0).repeat_interleave(2, 1)


def capture_subject_latent_memory(latents, masks):
    """Keep only masked source VAE-latent content for IDs 6 and 7 on CPU."""
    if latents.ndim != 5 or latents.shape[1] <= 7:
        raise ValueError("source latent frames 6 and 7 must be available")
    height, width = latents.shape[-2:]
    memory = []
    for frame_id in (6, 7):
        support = _lift_fixed_grid_mask(masks.source_masks[frame_id], height, width, latents.device)
        content = latents[:, frame_id].detach().clone() * support
        memory.append({"content": content.to(device="cpu", copy=True), "support": support.cpu()})
    return memory


def _mask_geometry(mask):
    coordinates = mask.nonzero(as_tuple=False)
    if not len(coordinates):
        raise ValueError("subject mask must contain at least one latent cell")
    ymin, xmin = coordinates.min(dim=0).values.tolist()
    ymax, xmax = coordinates.max(dim=0).values.tolist()
    centroid_yx = coordinates.float().mean(dim=0).tolist()
    return {
        "bbox_xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)],
        "centroid_xy": [float(centroid_yx[1]), float(centroid_yx[0])],
    }


def _warp_subject_latent(content, source_support, target_support):
    """Warp one masked source tensor so its bbox matches the target bbox."""
    source = _mask_geometry(source_support)
    target = _mask_geometry(target_support)
    sx0, sy0, sx1, sy1 = source["bbox_xyxy"]
    tx0, ty0, tx1, ty1 = target["bbox_xyxy"]
    source_size_xy = [sx1 - sx0, sy1 - sy0]
    target_size_xy = [tx1 - tx0, ty1 - ty0]
    if 0 in source_size_xy:
        raise ValueError("subject source bbox must span both latent axes")
    scale_xy = [target_size_xy[0] / source_size_xy[0], target_size_xy[1] / source_size_xy[1]]
    source_center_xy = [(sx0 + sx1) / 2, (sy0 + sy1) / 2]
    target_center_xy = [(tx0 + tx1) / 2, (ty0 + ty1) / 2]
    height, width = source_support.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=content.device, dtype=torch.float32),
        torch.arange(width, device=content.device, dtype=torch.float32), indexing="ij")
    source_x = (xx - target_center_xy[0]) / scale_xy[0] + source_center_xy[0]
    source_y = (yy - target_center_xy[1]) / scale_xy[1] + source_center_xy[1]
    grid = torch.stack((source_x * 2 / (width - 1) - 1, source_y * 2 / (height - 1) - 1), dim=-1)
    grid = grid.unsqueeze(0).expand(content.shape[0], -1, -1, -1)
    warped_content = F.grid_sample(
        content.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True).to(content.dtype)
    warped_support = F.grid_sample(
        source_support.float()[None, None], grid[:1], mode="nearest",
        padding_mode="zeros", align_corners=True)[0, 0].bool()
    return warped_content, warped_support, {
        "source": source,
        "target": target,
        "scale_xy": scale_xy,
        "translation_xy": [
            target_center_xy[0] - scale_xy[0] * source_center_xy[0],
            target_center_xy[1] - scale_xy[1] * source_center_xy[1],
        ],
    }


def transplant_subject_latent_memory(baseline, memory, masks, affine_align=False):
    """Apply the fixed 6 -> 6/7 -> 7 patch map only where source masks support it."""
    if baseline.ndim != 5 or baseline.shape[1] != 3 or len(memory) != 2:
        raise ValueError("latent subject patch requires three targets and two source patches")
    target = _lift_fixed_grid_mask(masks.target_subject_mask, *baseline.shape[-2:], baseline.device)
    content = [item["content"].to(device=baseline.device, dtype=baseline.dtype) for item in memory]
    supports = [item["support"].to(device=baseline.device, dtype=torch.bool) & target for item in memory]
    affine_audit = None
    if affine_align:
        warped = [_warp_subject_latent(value, item["support"].to(device=baseline.device, dtype=torch.bool), target)
                  for value, item in zip(content, memory)]
        content = [item[0] for item in warped]
        supports = [item[1] & target for item in warped]
        affine_audit = {str(frame_id): item[2] for frame_id, item in zip((6, 7), warped)}
    plans = [(supports[0], content[0]),
             (supports[0] & supports[1], (content[0] + content[1]) * 0.5),
             (supports[1], content[1])]
    patched = baseline.clone()
    for index, (support, values) in enumerate(plans):
        patched[:, index] = torch.where(support[None, None], values, baseline[:, index])
    outside = ~target
    audit = {
        "outside_target_equal": torch.equal(patched[:, :, :, outside], baseline[:, :, :, outside]),
        "outside_target_max_abs": float((patched[:, :, :, outside] - baseline[:, :, :, outside]).abs().max()),
        "supported_token_counts": [int(support.sum().item() // 4) for support, _ in plans],
        "supported_latent_cell_counts": [int(support.sum().item()) for support, _ in plans],
    }
    if affine_audit is not None:
        audit["source_to_target_affines"] = affine_audit
    return patched, audit


def latent_patch_cache_write_mask(memory, masks, baseline, erode_steps):
    """Return the source-supported target core used only for the block-8 cache write."""
    if erode_steps not in (0, 1, 2):
        raise ValueError("cache write erosion must be 0, 1, or 2")
    mode = "subject_to_subject" if erode_steps == 0 else f"subject_erode{erode_steps}"
    target = _lift_fixed_grid_mask(
        masks.target_mask_for_mode(mode), *baseline.shape[-2:], baseline.device)
    supports = [item["support"].to(device=baseline.device, dtype=torch.bool) & target for item in memory]
    plans = torch.stack((supports[0], supports[0] & supports[1], supports[1]))
    return plans, {
        "mode": mode,
        "token_counts": [int(plan.sum().item() // 4) for plan in plans],
        "latent_cell_counts": [int(plan.sum().item()) for plan in plans],
    }


def latent_patch_clean_cache_input(baseline, patched, persistent, block_number, cache_write_mask=None):
    """Allow only the requested block-8 patch to become autoregressive cache state."""
    if not persistent or block_number != 8:
        return baseline
    if cache_write_mask is None:
        return patched
    if cache_write_mask.shape != patched.shape[1:2] + patched.shape[-2:]:
        raise ValueError("cache write mask must have shape [target_frames, latent_height, latent_width]")
    return torch.where(cache_write_mask[None, :, None], patched, baseline)


def pack_fixed_grid_selective_memory(memory_store, masks, mode, current_frames, num_layers, alpha=1.0):
    """Pack raw masked K/V as spatial tokens or one pooled entity token per frame."""
    entries = memory_store.get_entries([6, 7])
    compact = mode == "compact_entity_memory"
    if compact:
        source_indices = {frame_id: masks.history_token_indices(frame_id) for frame_id in (6, 7)}
        target_indices = masks.subject_query_indices()
    elif mode == "background_to_background":
        source_indices = {
            frame_id: masks.history_background_token_indices(frame_id) for frame_id in (6, 7)}
        target_indices = masks.background_query_indices()
    else:
        source_indices = {
            frame_id: masks.history_token_indices_for_mode(mode, frame_id) for frame_id in (6, 7)}
        target_indices = masks.target_query_indices_for_mode(mode)

    query_indices = torch.tensor([
        frame * memory_store.frame_tokens + index
        for frame in range(current_frames) for index in target_indices
    ], dtype=torch.long)
    original_indices = None if compact else torch.tensor(
        source_indices[6] + source_indices[7], dtype=torch.long)
    temporal_slots = torch.tensor([1, 2], dtype=torch.long) if compact else torch.tensor(
        [1] * len(source_indices[6]) + [2] * len(source_indices[7]), dtype=torch.long)
    packed = [None] * num_layers
    for layer in memory_store.injection_layers:
        if layer >= num_layers:
            continue
        selected = []
        for entry in entries:
            indices = torch.tensor(source_indices[entry.frame_id], dtype=torch.long)
            selected_kv = {
                name: entry.layers[layer][name].index_select(1, indices)
                for name in ("k", "v")
            }
            selected.append({name: value.mean(dim=1, keepdim=True) for name, value in selected_kv.items()}
                            if compact else selected_kv)
        group = {
            "mode": mode,
            "alpha": alpha,
            "source_frame_ids": [6, 7],
            "source_token_counts": {frame_id: len(indices) for frame_id, indices in source_indices.items()},
            "temporal_slots": temporal_slots,
            "query_indices": query_indices,
            "historical_key": torch.cat([item["k"] for item in selected], dim=1),
            "historical_value": torch.cat([item["v"] for item in selected], dim=1),
        }
        if compact:
            group["position_mode"] = "temporal_only_neutral_spatial"
            group["source_memory_token_counts"] = {6: 1, 7: 1}
        else:
            group["source_token_indices"] = source_indices
            group["original_token_indices"] = original_indices
        packed[layer] = [group]
    return packed


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        
        # Default FPS for scene duration calculations
        # Formula: blocks = (seconds * fps) / (4 * num_frame_per_block)
        # Where 4 is the upsampling factor from latent frames to actual frames
        self.default_fps = getattr(args, "fps", 16)
        self.default_blocks_per_scene = 14  # Default: 14 blocks = 10.5 seconds at 16 fps

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        noncontiguous_source_blocks: Optional[List[int]] = None,
        noncontiguous_target_block: Optional[int] = None,
        noncontiguous_mode: str = "baseline",
        noncontiguous_retrieval_count: int = 1,
        noncontiguous_random_seed: int = 0,
        noncontiguous_manual_frame_id: Optional[int] = None,
        noncontiguous_manual_frame_ids: Optional[List[int]] = None,
        memory_policy_config: Optional[dict] = None,
        clean_pass_callback=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        source_blocks = sorted(set(noncontiguous_source_blocks or []))
        noncontiguous_enabled = bool(source_blocks or noncontiguous_target_block is not None)
        memory_policy = memory_policy_config or {}
        memory_enabled = bool(memory_policy.get("enabled", False))
        if memory_enabled and noncontiguous_enabled:
            raise ValueError("attention-memory policy and Phase 1 non-contiguous KV are separate experiments")
        if memory_enabled and memory_policy["context_mode"] == "replace_recent" and memory_policy["k"] > 2:
            print("Attention-memory replace_recent supports at most two retained local slots; unavailable blocks are logged and skipped")
        if noncontiguous_enabled:
            if not source_blocks or noncontiguous_target_block is None:
                raise ValueError("non-contiguous KV requires source blocks and a target block")
            if source_blocks[0] < 1 or source_blocks[-1] >= noncontiguous_target_block:
                raise ValueError("source blocks must be positive and precede the target block")
            if noncontiguous_target_block > num_blocks:
                raise ValueError("non-contiguous target block exceeds generated blocks")
            if source_blocks[-1] > num_blocks:
                raise ValueError("non-contiguous source block exceeds generated blocks")
            if noncontiguous_mode not in {
                    "baseline", "coherent_history", "random_history",
                    "same_entity_history", "wrong_entity_history"}:
                raise ValueError("unsupported non-contiguous mode")
            if noncontiguous_retrieval_count not in {1, 2}:
                raise ValueError("non-contiguous retrieval count must be one or two")
            if self.local_attn_size != 6:
                raise ValueError("matched non-contiguous retrieval requires local_attn_size=6")
        
        # ================================
        # Interactive Video Generation
        # ================================
        # Parse scene durations from prompts
        scene_prompts, scene_block_counts, scene_cut_flags = self._parse_scene_durations(text_prompts[0])
        conditional_dict_list = [self.text_encoder(text_prompts=[tp]) for tp in scene_prompts]
        
        # Calculate cumulative block indices for scene transitions
        # scene_block_boundaries[i] is the block index where scene i+1 starts
        scene_block_boundaries = []
        scene_cut_boundaries = []  # Boundaries that should have scene cuts
        cumulative_blocks = 0
        for i, block_count in enumerate(scene_block_counts[:-1]):  # Exclude last scene
            cumulative_blocks += block_count
            scene_block_boundaries.append(cumulative_blocks)
            # scene_cut_flags[i] indicates if scene i+1 should start with a scene cut
            if scene_cut_flags[i]:
                scene_cut_boundaries.append(cumulative_blocks)
        
        print(f"Scene configuration:")
        for i, (prompt, blocks, has_cut) in enumerate(zip(scene_prompts, scene_block_counts, scene_cut_flags)):
            duration_seconds = (blocks * 4 * self.num_frame_per_block) / self.default_fps
            cut_indicator = " [SCENE CUT]" if has_cut else ""
            print(f"  Scene {i+1}: {blocks} blocks ({duration_seconds:.2f}s){cut_indicator} - '{prompt[:50]}...'")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["scene_cut"] = False

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            # Use the first scene's conditional dict for initial latent processing
            initial_conditional_dict = conditional_dict_list[0]
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=initial_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=initial_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop @hidir: Inference  enters here
        all_num_frames = [self.num_frame_per_block] * num_blocks 
        # all_num_frames = [self.num_frame_per_block * num_blocks]
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        # ------------------------------------------------------------ #
        def kv_flush(scene_cut_needed, device):
            """Flush KV cache at scene boundaries by rolling cache and resetting cross-attention."""
            n_layers = len(self.crossattn_cache)
            for i in range(n_layers):
                self.crossattn_cache[i]['is_init'] = False
                self.kv_cache1[i]['k'][:, 1560:4680] = self.kv_cache1[i]['k'][:, -3120:]
                self.kv_cache1[i]['v'][:, 1560:4680] = self.kv_cache1[i]['v'][:, -3120:]
                self.kv_cache1[i]['local_end_index'] = torch.tensor([4680], dtype=torch.long, device=device)
                self.kv_cache1[i]['scene_cut'] = scene_cut_needed
        # ------------------------------------------------------------ #
        captured_source_kv = {}
        memory_store = None
        memory_logger = MemoryPolicyEventLogger(None)
        last_query_descriptors = {}
        fixed_grid_config = memory_policy.get("fixed_grid")
        fixed_grid_masks = FixedGridMemoryMasks.from_json(
            fixed_grid_config["mask_path"]) if fixed_grid_config else None
        latent_cache_erode_steps = {
            "latent_subject_patch_persistent_cache_erode1": 1,
            "latent_subject_patch_persistent_cache_erode2": 2,
        }.get(fixed_grid_config["mode"] if fixed_grid_config else None, 0)
        latent_subject_patch = bool(fixed_grid_config and fixed_grid_config["mode"] in {
            "latent_subject_patch", "affine_aligned_latent_subject_patch", "latent_subject_patch_persistent",
            "latent_subject_patch_persistent_cache_erode1",
            "latent_subject_patch_persistent_cache_erode2"})
        affine_aligned_latent_subject_patch = bool(
            fixed_grid_config and fixed_grid_config["mode"] == "affine_aligned_latent_subject_patch")
        latent_subject_patch_persistent = bool(
            fixed_grid_config and fixed_grid_config["mode"] == "latent_subject_patch_persistent") or \
            bool(latent_cache_erode_steps)
        source_latent_memory = None
        if memory_enabled:
            memory_store = MemoryStore(
                frame_tokens=self.frame_seq_length,
                descriptor_layers=memory_policy["descriptor_layers"],
                memory_budget=memory_policy["target_budget"],
                injection_layers=memory_policy["injection_layers"],
            )
            memory_logger = MemoryPolicyEventLogger(memory_policy.get("log_path"))
            memory_logger.write("config", {
                key: value for key, value in memory_policy.items() if key != "manual_frame_ids"
            } | {"manual_frame_ids": memory_policy.get("manual_frame_ids"),
                 "frame_tokens": self.frame_seq_length,
                 "num_transformer_layers": self.num_transformer_blocks,
                 "local_attn_size": self.local_attn_size,
                 "sink_size": 1})
            if fixed_grid_config:
                memory_logger.write("fixed_grid_dmd_schedule", {
                    **fixed_grid_denoising_schedule(self.denoising_step_list.tolist()),
                    "clean_pass_timestep": float(self.args.context_noise),
                    "denoising_steps": fixed_grid_config.get("denoising_steps", "all"),
                    "clean_pass_enabled": fixed_grid_config.get("clean_pass", True),
                })
        for current_block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()
            # Determine which scene this block belongs to
            scene_index = 0
            for boundary in scene_block_boundaries:
                if current_block_index < boundary:
                    break
                scene_index += 1
            conditional_dict = conditional_dict_list[scene_index]
            
            # Check if we need to flush KV cache (at scene boundaries)
            # Flush when we transition to a new scene (except at the start)
            scene_cut_needed = current_block_index in scene_cut_boundaries
            if current_block_index in scene_block_boundaries:
                if memory_enabled:
                    pre_transition_query = local_query_descriptors(
                        self.kv_cache1, memory_policy["descriptor_layers"], self.frame_seq_length)
                    previous_scene_id = scene_index - 1
                    archive = memory_store.archive_scene(previous_scene_id, memory_policy["archive_top_m"]) \
                        if memory_policy["archive"] else None
                    trimmed_archives = memory_store.trim_archive(
                        memory_policy["archive_recent_scenes"], memory_policy["archive_high_utility"]) \
                        if memory_policy["archive"] else []
                    transition = apply_memory_transition(
                        self.kv_cache1, self.crossattn_cache, self.frame_seq_length,
                        memory_policy["local_retention"], memory_policy["decay"],
                        memory_policy["decay_beta"], scene_cut_needed, noise.device,
                        memory_policy["cross_attention_reset"],
                        memory_policy.get("scene_local_rope_epoch", False), current_start_frame)
                    last_query_descriptors = pre_transition_query or last_query_descriptors
                    memory_logger.write("transition", {
                        "from_scene_id": previous_scene_id, "to_scene_id": scene_index,
                        "archive_frame_ids": None if archive is None else archive.frame_ids,
                        "archive_utility": None if archive is None else archive.utility,
                        "archive_size": len(memory_store.archive), "trimmed_archive_scene_ids": trimmed_archives,
                        "memory_size": len(memory_store), **transition,
                    })
                else:
                    kv_flush(scene_cut_needed, noise.device)
            else:
                # Reset scene_cut flag (it should only be True for the first block of a scene with cut)
                n_layers = len(self.crossattn_cache)
                for i in range(n_layers):
                    self.kv_cache1[i]['scene_cut'] = False
            # ---------------------------------------------------------------- #
            current_block_number = current_block_index + 1
            retrieved_kv = None
            selective_memory = None
            history_refs = []
            memory_route = None
            memory_query = local_query_descriptors(
                self.kv_cache1, memory_policy.get("descriptor_layers", ()), self.frame_seq_length) \
                if memory_enabled else {}
            memory_query_source = "local_raw"
            if memory_enabled and not memory_query:
                memory_query = last_query_descriptors
                memory_query_source = "pre_transition_raw" if memory_query else "unavailable"
            local_non_sink_frames = max(0, self.kv_cache1[0]["local_end_index"].item() // self.frame_seq_length - 1)
            context_non_sink_frames = max(
                0, min(self.local_attn_size, local_non_sink_frames + 1 + current_num_frames) - current_num_frames - 1)
            local_frame_ids = list(range(max(1, current_start_frame - local_non_sink_frames), current_start_frame))
            context_local_frame_ids = local_frame_ids[-context_non_sink_frames:]
            is_transition_block = current_block_index in scene_block_boundaries
            if memory_enabled and is_transition_block:
                memory_logger.write("attention_context", {
                    "block": current_block_number, "scene_id": scene_index,
                    "retention": memory_policy["local_retention"],
                    **transition_attention_context(
                        current_start_frame, current_num_frames, memory_policy["local_retention"],
                        scene_cut_needed, self.frame_seq_length,
                        self.kv_cache1[0].get("persistent_sink_frame_id", 0),
                        transition["scene_local_rope_epoch"]),
                })
            retrieval_allowed_now, retrieval_reason = retrieval_allowed(
                memory_policy.get("retrieval", False), is_transition_block,
                memory_policy.get("transition_auto_retrieval", False),
                memory_policy.get("manual_frame_ids"), memory_policy.get("manual_target_blocks"), current_block_number,
                memory_policy.get("retrieval_lifetime", "pulse_1"))
            manual_retrieval = bool(memory_policy.get("manual_frame_ids"))
            if memory_enabled and retrieval_allowed_now and (memory_query or manual_retrieval) and len(memory_store):
                retrieval_available = memory_policy["context_mode"] != "replace_recent" or \
                    memory_policy["k"] <= context_non_sink_frames
                if retrieval_available:
                    memory_route = route_memory(
                        memory_store, memory_query, memory_policy["k"], exclude_frame_ids={0, *local_frame_ids},
                        manual_frame_ids=memory_policy.get("manual_frame_ids"))
                    if memory_route.entries:
                        if fixed_grid_masks is not None:
                            if current_block_number == 8:
                                alpha = fixed_grid_config.get("alpha", 1.0)
                                selective_memory = None if latent_subject_patch or alpha == 0.0 else pack_fixed_grid_selective_memory(
                                    memory_store, fixed_grid_masks, fixed_grid_config["mode"],
                                    current_num_frames, self.num_transformer_blocks, alpha=alpha)
                                compact = fixed_grid_config["mode"] == "compact_entity_memory"
                                if compact or latent_subject_patch:
                                    target_indices = fixed_grid_masks.subject_query_indices()
                                    source_indices = {
                                        frame_id: fixed_grid_masks.history_token_indices(frame_id)
                                        for frame_id in (6, 7)}
                                elif fixed_grid_config["mode"] == "background_to_background":
                                    target_indices = fixed_grid_masks.background_query_indices()
                                    source_indices = {
                                        frame_id: fixed_grid_masks.history_background_token_indices(frame_id)
                                        for frame_id in (6, 7)}
                                else:
                                    target_indices = fixed_grid_masks.target_query_indices_for_mode(
                                        fixed_grid_config["mode"])
                                    source_indices = {
                                        frame_id: fixed_grid_masks.history_token_indices_for_mode(
                                            fixed_grid_config["mode"], frame_id)
                                        for frame_id in (6, 7)}
                                base_sink_frame = current_start_frame - context_non_sink_frames - 1
                                base_ordering = (
                                    [f"sink:{base_sink_frame}"] +
                                    [f"local:{frame_id}" for frame_id in range(
                                        current_start_frame - context_non_sink_frames, current_start_frame)] +
                                    [f"current:{frame_id}" for frame_id in range(
                                        current_start_frame, current_start_frame + current_num_frames)]
                                )
                                memory_logger.write("fixed_grid_selective_memory", {
                                    "block": current_block_number,
                                    "mode": fixed_grid_config["mode"],
                                    "alpha": alpha,
                                    "historical_branch_bypassed": latent_subject_patch or alpha == 0.0,
                                    "source_frame_ids": [6, 7],
                                    "source_token_counts": {
                                        frame_id: len(indices) for frame_id, indices in source_indices.items()},
                                    "source_temporal_slots": {6: 1, 7: 2},
                                    "representation": ("manual_subject_latent_patch" if latent_subject_patch else
                                                       "mean_pooled_entity" if compact else "raw_sparse_spatial_kv"),
                                    "pooled_memory_token_counts": {6: 1, 7: 1} if compact else None,
                                    "historical_position_mode": (
                                        "vae_latent_2x2_mask_lift" if latent_subject_patch else
                                        "temporal_only_neutral_spatial" if compact else "original_source_spatial"),
                                    "source_spatial_coordinates_applied": not (compact or latent_subject_patch),
                                    "source_token_indices": source_indices,
                                    "source_row_col_coordinates": {
                                        frame_id: [divmod(index, fixed_grid_masks.width) for index in indices]
                                        for frame_id, indices in source_indices.items()},
                                    "target_block": 8,
                                    "target_query_frame_ids": list(range(
                                        current_start_frame, current_start_frame + current_num_frames)),
                                    "target_per_frame_query_indices": target_indices,
                                    "target_query_count": current_num_frames * len(target_indices),
                                    "base_ordering": base_ordering,
                                    "base_order_derived_from": {
                                        "current_start_frame": current_start_frame,
                                        "context_non_sink_frames": context_non_sink_frames,
                                        "current_num_frames": current_num_frames,
                                    },
                                    "base_context_unchanged": True,
                                })
                                memory_logger.write("fixed_grid_dmd_timestep_gate", {
                                    **fixed_grid_denoising_schedule(self.denoising_step_list.tolist()),
                                    "block": current_block_number,
                                    "denoising_steps": fixed_grid_config.get("denoising_steps", "all"),
                                    "denoising_active": [
                                        fixed_grid_memory_active(
                                            fixed_grid_config.get("denoising_steps", "all"),
                                            fixed_grid_config.get("clean_pass", True), index,
                                            len(self.denoising_step_list))
                                        for index in range(len(self.denoising_step_list))],
                                    "clean_pass_timestep": float(self.args.context_noise),
                                    "clean_pass_active": fixed_grid_memory_active(
                                        fixed_grid_config.get("denoising_steps", "all"),
                                        fixed_grid_config.get("clean_pass", True), clean_pass=True),
                                })
                        else:
                            retrieved_kv = memory_store.pack_kv(
                                memory_route.entries, self.num_transformer_blocks)
                            history_refs = [{"source_block": None, "frame_index": None,
                                             "global_frame_id": entry.frame_id, "scene_id": entry.scene_id}
                                            for entry in memory_route.entries]
                memory_logger.write("retrieval", {
                    "block": current_block_number, "scene_id": scene_index,
                    "query_layers": sorted(memory_query), "query_source": memory_query_source,
                    "requested_k": memory_policy["k"],
                    "retrieved_frame_ids": [entry.frame_id for entry in memory_route.entries] if memory_route else [],
                    "retrieved_scene_ids": [entry.scene_id for entry in memory_route.entries] if memory_route else [],
                    "scores": [] if memory_route is None else memory_route.scores,
                    "excluded_frame_ids": [0, *local_frame_ids], "memory_size": len(memory_store),
                    "context_mode": memory_policy["context_mode"],
                    "retrieval_lifetime": memory_policy.get("retrieval_lifetime", "pulse_1"),
                    "routing_mode": retrieval_reason,
                })
            elif memory_enabled:
                memory_logger.write("retrieval", {
                    "block": current_block_number, "scene_id": scene_index,
                    "query_layers": sorted(memory_query), "query_source": memory_query_source,
                    "requested_k": memory_policy["k"],
                    "retrieved_frame_ids": [], "retrieved_scene_ids": [], "scores": [],
                    "excluded_frame_ids": [0, *local_frame_ids], "memory_size": len(memory_store),
                    "context_mode": memory_policy["context_mode"],
                    "retrieval_lifetime": memory_policy.get("retrieval_lifetime", "pulse_1"),
                    "routing_mode": retrieval_reason,
                    "skipped": "no_query_or_memory" if not (memory_query or manual_retrieval) or not len(memory_store) else
                    ("replace_recent_local_slots" if retrieval_allowed_now else retrieval_reason),
                })
            if noncontiguous_enabled and current_block_number == noncontiguous_target_block:
                if current_num_frames != 3:
                    raise ValueError("matched non-contiguous retrieval requires three current latent frames")
                if noncontiguous_mode != "baseline":
                    missing_sources = [block for block in source_blocks if block not in captured_source_kv]
                    if missing_sources:
                        raise RuntimeError(f"missing clean KV for source blocks {missing_sources}")
                    history_refs = select_history_frame_refs(
                        captured_source_kv, source_blocks, noncontiguous_retrieval_count,
                        noncontiguous_mode, noncontiguous_random_seed,
                        noncontiguous_manual_frame_id, noncontiguous_manual_frame_ids)
                    retrieved_kv = select_history_kv(
                        captured_source_kv, history_refs, self.num_transformer_blocks,
                        self.frame_seq_length)
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                denoising_selective_memory = selective_memory
                if fixed_grid_config is not None and selective_memory is not None:
                    denoising_selective_memory = selective_memory if fixed_grid_memory_active(
                        fixed_grid_config.get("denoising_steps", "all"),
                        fixed_grid_config.get("clean_pass", True), index,
                        len(self.denoising_step_list)) else None

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieved_kv=retrieved_kv,
                        memory_context_mode=memory_policy.get("context_mode", "replace_recent"),
                        selective_memory=denoising_selective_memory,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieved_kv=retrieved_kv,
                        memory_context_mode=memory_policy.get("context_mode", "replace_recent"),
                        selective_memory=denoising_selective_memory,
                    )

            baseline_denoised_pred = denoised_pred
            output_denoised_pred = denoised_pred
            if latent_subject_patch and current_block_number == 8:
                if source_latent_memory is None:
                    raise RuntimeError("latent subject source frames were not captured before block 8")
                output_denoised_pred, patch_audit = transplant_subject_latent_memory(
                    baseline_denoised_pred, source_latent_memory, fixed_grid_masks,
                    affine_align=affine_aligned_latent_subject_patch)
                cache_write_mask, cache_write_audit = (None, None)
                if latent_cache_erode_steps:
                    cache_write_mask, cache_write_audit = latent_patch_cache_write_mask(
                        source_latent_memory, fixed_grid_masks, baseline_denoised_pred,
                        latent_cache_erode_steps)
                clean_cache_denoised_pred = latent_patch_clean_cache_input(
                    denoised_pred, output_denoised_pred, latent_subject_patch_persistent,
                    current_block_number, cache_write_mask)
                memory_logger.write("latent_subject_patch", {
                    "block": current_block_number,
                    "source_frame_ids": [6, 7],
                    "target_frame_ids": list(range(current_start_frame, current_start_frame + current_num_frames)),
                    "temporal_mapping": ["source_6", "0.5_source_6+0.5_source_7", "source_7"],
                    "latent_shape": list(output_denoised_pred.shape),
                    "target_mask_lift": "30x52_to_60x104_exact_2x2",
                    "spatial_registration": "bbox_affine" if affine_aligned_latent_subject_patch else "none",
                    "clean_cache_input": (
                        f"patched_subject_latent_cache_erode{latent_cache_erode_steps}"
                        if latent_cache_erode_steps else
                        "patched_subject_latent" if latent_subject_patch_persistent else
                        "baseline_denoised_pred"),
                    "clean_cache_input_equals_baseline": torch.equal(clean_cache_denoised_pred, denoised_pred),
                    **patch_audit,
                    **({"cache_write_mask": cache_write_audit} if cache_write_audit else {}),
                })
            else:
                clean_cache_denoised_pred = denoised_pred

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = output_denoised_pred
            if latent_subject_patch and source_latent_memory is None and \
                    current_start_frame + current_num_frames > 7:
                source_latent_memory = capture_subject_latent_memory(output, fixed_grid_masks)
                memory_logger.write("latent_subject_memory_capture", {
                    "source_frame_ids": [6, 7],
                    "latent_shape_per_frame": list(source_latent_memory[0]["content"].shape),
                    "source_token_counts": {
                        frame_id: len(fixed_grid_masks.history_token_indices(frame_id)) for frame_id in (6, 7)},
                    "mask_lift": "30x52_to_60x104_exact_2x2",
                    "stored_content": "masked_subject_only_cpu",
                })

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            clean_selective_memory = selective_memory
            if fixed_grid_config is not None and selective_memory is not None:
                clean_selective_memory = selective_memory if fixed_grid_memory_active(
                    fixed_grid_config.get("denoising_steps", "all"),
                    fixed_grid_config.get("clean_pass", True), clean_pass=True) else None
            if memory_enabled and memory_policy["local_retention"] == "recent_only_no_sink":
                for cache in self.kv_cache1:
                    cache["recent_only_no_sink_finalize"] = True
            self.generator(
                noisy_image_or_video=clean_cache_denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                retrieved_kv=retrieved_kv,
                memory_context_mode=memory_policy.get("context_mode", "replace_recent"),
                selective_memory=clean_selective_memory,
                capture_kv=(memory_enabled or (noncontiguous_enabled and current_block_number in source_blocks)),
            )
            if clean_pass_callback is not None:
                clean_pass_callback(current_block_number, denoised_pred, self.kv_cache1)

            if memory_enabled and is_transition_block:
                record_transition_sink(
                    self.kv_cache1, current_start_frame, memory_policy["local_retention"])

            if noncontiguous_enabled and current_block_number in source_blocks:
                captured_source_kv[current_block_number] = {
                    "frame_ids": list(range(current_start_frame, current_start_frame + current_num_frames)),
                    "layers": capture_clean_kv_to_cpu(self.kv_cache1),
                }
            elif memory_enabled:
                clean_layers = capture_clean_kv_to_cpu(self.kv_cache1)
                capture_clean_memory_block(
                    memory_store, scene_index, current_start_frame, current_num_frames, clean_layers)
                last_query_descriptors = local_query_descriptors(
                    self.kv_cache1, memory_policy["descriptor_layers"], self.frame_seq_length)
                consolidation = {"performed": False, "before": len(memory_store), "after": len(memory_store), "removed": []}
                if memory_policy["consolidation"] and len(memory_store) > memory_policy["consolidate_n_max"]:
                    consolidation = memory_store.consolidate(
                        memory_policy["target_budget"], memory_policy["diversity_threshold"])
                memory_logger.write("write", {
                    "block": current_block_number, "scene_id": scene_index,
                    "frame_ids": list(range(current_start_frame, current_start_frame + current_num_frames)),
                    "memory_size": len(memory_store), "archive_size": len(memory_store.archive),
                    "consolidation": consolidation,
                })
            if memory_enabled and (history_refs or not is_transition_block):
                retrieved_count = len(history_refs)
                layout = memory_context_layout(
                    retrieved_count, context_non_sink_frames, current_num_frames, memory_policy["context_mode"])
                context_order = memory_context_order(
                    self.kv_cache1[0].get("persistent_sink_frame_id", 0),
                    [frame["global_frame_id"] for frame in history_refs], context_local_frame_ids,
                    list(range(current_start_frame, current_start_frame + current_num_frames)),
                    layout["local_frames"])
                memory_logger.write("context", {
                    "block": current_block_number, "scene_id": scene_index,
                    "ordering": context_order, "positions": layout["positions"],
                    "rope_positions": memory_context_rope_positions(
                        retrieved_count, context_non_sink_frames, current_num_frames,
                        memory_policy["context_mode"], current_start_frame, scene_cut_needed),
                    "retrieved_frames": retrieved_count, "local_frames": layout["local_frames"],
                    "current_frames": current_num_frames, "total_frames": len(context_order),
                    "total_tokens": len(context_order) * self.frame_seq_length,
                })
            if noncontiguous_enabled and current_block_number == noncontiguous_target_block:
                history_ids = [frame["global_frame_id"] for frame in history_refs]
                retained_recent = 2 - len(history_refs)
                recent_ids = list(range(current_start_frame - retained_recent, current_start_frame))
                current_ids = list(range(current_start_frame, current_start_frame + current_num_frames))
                context_order = ["sink:0"] + [f"history:{frame_id}" for frame_id in history_ids] + \
                    [f"recent:{frame_id}" for frame_id in recent_ids] + [f"current:{frame_id}" for frame_id in current_ids]
                context_frames = len(context_order)
                context_tokens = context_frames * self.frame_seq_length
                print(
                    "Non-contiguous KV context "
                    f"(mode={noncontiguous_mode}, block={current_block_number}, source_blocks={source_blocks}, "
                    f"history_global_frame_ids={history_ids}, ordering={context_order}, "
                    f"rope_positions={list(range(context_frames))}, retrieved={len(history_refs)} frames/"
                    f"{len(history_refs) * self.frame_seq_length} tokens, recent={retained_recent} frames/"
                    f"{retained_recent * self.frame_seq_length} tokens, current={current_num_frames} frames/"
                    f"{current_num_frames * self.frame_seq_length} tokens, total={context_frames} frames/"
                    f"{context_tokens} tokens, baseline_total=6 frames/{6 * self.frame_seq_length} tokens)"
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    # def _initialize_compressed_kv_cache(self, batch_size, dtype, device):
    #     """
    #     Initialize a Per-GPU compressed KV cache for the Wan model.
    #     """
    #     kv_cache1 = []
    #     if self.local_attn_size != -1:
    #         # Use the local attention size to compute the KV cache size
    #         kv_cache_size = self.local_attn_size * self.frame_seq_length
    #     else:
    #         # Use the default KV cache size
    #         kv_cache_size = 32760

    #     for _ in range(self.num_transformer_blocks):
    #         kv_cache1.append({
    #             "compressed_kv": torch.zeros([batch_size, kv_cache_size, 1088], dtype=dtype, device=device),
    #             "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
    #             "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
    #         })

    #     self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "scene_cut": False
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _parse_scene_durations(self, prompt: str) -> Tuple[List[str], List[int], List[bool]]:
        """
        Parse scene prompts with optional durations and scene cut indicators.
        
        Format: "prompt1[5s] | prompt2[15s#] | prompt3"
        - If duration is specified (e.g., [5s]), use that duration
        - If no duration is specified, use the default (14 blocks = 10.5 seconds)
        - If # is present after duration (e.g., [15s#]), mark that scene transition for scene cut
        
        Returns:
            Tuple of (prompt_texts, block_counts_per_scene, scene_cut_flags)
            scene_cut_flags[i] is True if scene i+1 should start with a scene cut
        """
        # Split by | to get individual scene prompts
        scene_parts = [part.strip() for part in prompt.split('|')]
        prompt_texts = []
        block_counts = []
        scene_cut_flags = []
        
        for scene_part in scene_parts:
            # Check if duration is specified: [Xs] or [X.5s] or [Xs#] etc.
            duration_match = re.search(r'\[(\d+\.?\d*)\s*s#?\]', scene_part)
            has_scene_cut = '#' in scene_part
            
            if duration_match:
                # Extract duration in seconds
                duration_seconds = float(duration_match.group(1))
                # Remove the duration from the prompt text (including # if present)
                prompt_text = re.sub(r'\[\d+\.?\d*\s*s#?\]', '', scene_part).strip()
                
                # Convert seconds to blocks
                # Formula: blocks = (seconds * fps) / (4 * num_frame_per_block)
                # Where 4 is the upsampling factor from latent to actual frames
                blocks = int((duration_seconds * self.default_fps) / (4 * self.num_frame_per_block))
                # Ensure at least 1 block
                blocks = max(1, blocks)
            else:
                # No duration specified, use default
                prompt_text = scene_part
                blocks = self.default_blocks_per_scene
            
            prompt_texts.append(prompt_text)
            block_counts.append(blocks)
            scene_cut_flags.append(has_scene_cut)
        
        return prompt_texts, block_counts, scene_cut_flags
