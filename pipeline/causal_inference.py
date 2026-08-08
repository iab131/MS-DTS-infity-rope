from typing import List, Optional, Tuple
from random import Random
import torch
import re

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


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
                kv_flush(scene_cut_needed, noise.device)
            else:
                # Reset scene_cut flag (it should only be True for the first block of a scene with cut)
                n_layers = len(self.crossattn_cache)
                for i in range(n_layers):
                    self.kv_cache1[i]['scene_cut'] = False
            # ---------------------------------------------------------------- #
            current_block_number = current_block_index + 1
            retrieved_kv = None
            history_refs = []
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

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieved_kv=retrieved_kv,
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
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                retrieved_kv=retrieved_kv,
                capture_kv=(noncontiguous_enabled and current_block_number in source_blocks),
            )
            if clean_pass_callback is not None:
                clean_pass_callback(current_block_number, denoised_pred, self.kv_cache1)

            if noncontiguous_enabled and current_block_number in source_blocks:
                captured_source_kv[current_block_number] = {
                    "frame_ids": list(range(current_start_frame, current_start_frame + current_num_frames)),
                    "layers": capture_clean_kv_to_cpu(self.kv_cache1),
                }
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
