import argparse
import torch
import os
import re
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from utils.interactive import add_subtitles
from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks, validate_fixed_grid_options

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

def sanitize_filename(text, max_length=100):
    """Remove or replace invalid filename characters."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Truncate to max_length
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized


def write_video_file(path, frames, fps):
    """Use torchvision when available, otherwise the repository's imageio path."""
    try:
        from torchvision.io import write_video
    except ImportError:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames.detach().cpu().clamp(0, 255).to(torch.uint8).numpy():
                writer.append_data(frame)
    else:
        write_video(path, frames, fps=fps)

def parse_durations_from_prompt(prompt_text):
    """
    Parse durations from prompt in the format: "action 1"[5s] | "action 2"[15s] | ...
    Returns total duration in seconds if all actions have durations, None otherwise.
    """
    # Split by ';' to get the part before subtitles
    prompt_part = prompt_text.split(';')[0].strip()
    
    # Split by '|' to get individual actions
    scene_parts = [part.strip() for part in prompt_part.split('|')]
    
    total_duration = 0.0
    for scene_part in scene_parts:
        # Look for duration pattern: [5s], [15s#], [10.5s], etc.
        duration_match = re.search(r'\[(\d+\.?\d*)\s*s[#]?\]', scene_part)
        if duration_match:
            duration_seconds = float(duration_match.group(1))
            total_duration += duration_seconds
        else:
            # If any action doesn't have a duration, return None
            return None
    
    return total_duration if total_duration > 0 else None

def calculate_latent_frames_from_duration(total_duration_seconds, fps, temporal_compression, 
                                          num_frame_per_block, independent_first_frame, 
                                          has_initial_latent):
    """
    Calculate the number of latent frames needed based on total duration.
    
    Args:
        total_duration_seconds: Total duration in seconds
        fps: Frames per second (16)
        temporal_compression: Compression factor from latent to actual frames (4)
        num_frame_per_block: Number of frames per block (must be multiple)
        independent_first_frame: Whether first frame is independent (from config)
        has_initial_latent: Whether initial_latent is provided (i2v case)
    
    Returns:
        Number of latent frames for the noise tensor
    """
    import math
    
    # Calculate total output frames
    total_output_frames = int(total_duration_seconds * fps)
    
    # Determine which pipeline constraint applies
    # Pipeline checks: if not independent_first_frame or (independent_first_frame and initial_latent is not None):
    #   -> num_frames % num_frame_per_block == 0
    # else (independent_first_frame and initial_latent is None):
    #   -> (num_frames - 1) % num_frame_per_block == 0
    
    if has_initial_latent:
        # For image-to-video: first frame is provided as initial_latent
        # Total output frames = 1 (from initial) + num_latent_frames * temporal_compression
        # So: num_latent_frames = (total_output_frames - 1) / temporal_compression
        base_latent_frames = (total_output_frames - 1) // temporal_compression
        # Must be multiple of num_frame_per_block (because initial_latent is provided)
        latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
        # Ensure at least num_frame_per_block frames
        latent_frames = max(latent_frames, num_frame_per_block)
    else:
        # For text-to-video
        if independent_first_frame:
            # Pipeline expects: (num_frames - 1) % num_frame_per_block == 0
            # Total output frames = 1 (independent first) + num_latent_frames * temporal_compression
            # So: num_latent_frames = (total_output_frames - 1) / temporal_compression
            base_latent_frames = (total_output_frames - 1) // temporal_compression
            # (latent_frames - 1) must be multiple of num_frame_per_block
            # So latent_frames = 1 + k * num_frame_per_block for some k >= 0
            if base_latent_frames == 0:
                latent_frames = 1
            else:
                k = math.ceil(base_latent_frames / num_frame_per_block)
                latent_frames = 1 + k * num_frame_per_block
        else:
            # Pipeline expects: num_frames % num_frame_per_block == 0
            # Total output frames = num_latent_frames * temporal_compression
            # So: num_latent_frames = total_output_frames / temporal_compression
            base_latent_frames = total_output_frames // temporal_compression
            # Must be multiple of num_frame_per_block
            latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
            # Ensure at least num_frame_per_block frames
            latent_frames = max(latent_frames, num_frame_per_block)
    
    return latent_frames

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=None,
                    help="Number of output frames. Required if prompt does not contain duration information.")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--output_index", type=int, default=None,
                    help="Override the index in output filename (default: uses seed_idx from num_samples loop)")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--noncontiguous-kv", action="store_true",
                    help="Enable Phase 1 historical clean-KV injection.")
parser.add_argument("--noncontiguous-source-blocks", type=str, default=None,
                    help="Comma-separated one-based clean-pass source block numbers.")
parser.add_argument("--noncontiguous-target-block", type=int, default=None,
                    help="One-based block number that receives the historical KV prefix.")
parser.add_argument("--noncontiguous-kv-mode", choices=[
                    "baseline", "coherent_history", "random_history",
                    "same_entity_history", "wrong_entity_history"],
                    default="baseline", help="Matched-context Phase 1 mode.")
parser.add_argument("--noncontiguous-retrieval-count", type=int, choices=[1, 2], default=1,
                    help="Historical latent frames replacing recent non-sink context frames.")
parser.add_argument("--noncontiguous-history-frame-id", type=int, default=None,
                    help="Manually selected global latent frame for an oracle history mode.")
parser.add_argument("--noncontiguous-history-frame-ids", type=str, default=None,
                    help="Comma-separated manual global latent frames for an oracle history mode.")
parser.add_argument("--save-clean-latent-blocks", type=str, default=None,
                    help="Comma-separated one-based clean-latent block numbers to save.")
parser.add_argument("--save-raw-decoded", action="store_true",
                    help="Save the decoded tensor before MP4 conversion.")
parser.add_argument("--attention-memory-policy", action="store_true",
                    help="Enable the experimental clean-pass attention-memory policy.")
parser.add_argument("--memory-retrieval", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable content-routed historical K/V retrieval when the policy is enabled.")
parser.add_argument("--memory-context-mode", choices=["replace_recent", "prepend"], default="prepend",
                    help="Replace local frames at matched budget or prepend retrieved historical frames.")
parser.add_argument("--memory-k", type=int, default=5,
                    help="Number of historical latent frames selected by content routing.")
parser.add_argument("--memory-descriptor-layers", type=str, default="0,1,5,14,16",
                    help="Comma-separated clean-K layers used for mean-pooled routing descriptors.")
parser.add_argument("--memory-injection-layers", type=str, default="0,1,5,14,16",
                    help="Comma-separated layers that retain historical K/V and receive transient injection.")
parser.add_argument("--memory-manual-frame-ids", type=str, default=None,
                    help="Comma-separated MemoryStore frame IDs overriding descriptor routing in order.")
parser.add_argument("--memory-manual-target-blocks", type=str, default=None,
                    help="Optional comma-separated one-based blocks where manual memory injection is allowed.")
parser.add_argument("--memory-retrieval-lifetime", choices=["pulse_1", "pulse_2", "persistent"], default="pulse_1",
                    help="Manual retrieval duration from the first selected block; pulse_1 preserves the oracle default.")
parser.add_argument("--memory-transition-auto-retrieval", action=argparse.BooleanOptionalAction, default=False,
                    help="Allow automatic descriptor routing on the first block after a scene transition.")
parser.add_argument("--memory-local-retention", choices=["sink_only", "sink+1", "sink+2", "transition_no_sink"], default="sink_only",
                    help="Local cache retained at policy-managed scene transitions.")
parser.add_argument("--memory-decay", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply fixed beta decay to retained non-sink local K/V at policy transitions.")
parser.add_argument("--memory-decay-beta", type=float, default=0.3,
                    help="Fixed retained non-sink K/V multiplier at policy transitions.")
parser.add_argument("--memory-crossattn-reset", action=argparse.BooleanOptionalAction, default=True,
                    help="Reset cross-attention cache at policy-managed scene transitions.")
parser.add_argument("--memory-archive", action=argparse.BooleanOptionalAction, default=True,
                    help="Archive a utility-weighted scene summary at policy-managed scene transitions.")
parser.add_argument("--memory-archive-top-m", type=int, default=3,
                    help="Top utility frames compressed into each SceneArchive entry.")
parser.add_argument("--memory-archive-recent-scenes", type=int, default=10,
                    help="Recent SceneArchive entries retained after archive trimming.")
parser.add_argument("--memory-archive-high-utility", type=int, default=5,
                    help="High-utility SceneArchive entries retained after archive trimming.")
parser.add_argument("--memory-consolidation", action=argparse.BooleanOptionalAction, default=True,
                    help="Consolidate MemoryStore when its configurable frame threshold is exceeded.")
parser.add_argument("--memory-consolidate-n-max", type=int, default=200,
                    help="MemoryStore frame count that triggers consolidation.")
parser.add_argument("--memory-target-budget", type=int, default=150,
                    help="MemoryStore frame count retained by consolidation.")
parser.add_argument("--memory-diversity-threshold", type=float, default=0.9,
                    help="Cosine similarity threshold above which consolidation treats entries as redundant.")
parser.add_argument("--memory-policy-log", type=str, default=None,
                    help="JSONL event log path; defaults under output_folder when policy is enabled.")
parser.add_argument("--memory-fixed-grid-mask-path", type=str, default=None,
                    help="Opt-in fixed 30x52 source/target mask JSON for the manual recall oracle.")
parser.add_argument("--memory-fixed-grid-mode", choices=[
                    "subject_to_subject", "subject_erode1", "subject_erode2",
                    "subject_boundary_only", "background_to_background", "compact_entity_memory",
                    "latent_subject_patch"], default=None,
                    help="Apply only the matching fixed-grid historical-memory arm.")
parser.add_argument("--memory-fixed-grid-alpha", type=float, default=1.0,
                    help="Interpolate fixed-grid historical output from baseline (0) to full oracle (1).")
parser.add_argument("--memory-fixed-grid-denoising-steps", choices=["all", "latest_1", "latest_2", "clean_only"],
                    default="all", help="Fixed-grid history on DMD calls, or only on the clean cache pass.")
parser.add_argument("--memory-fixed-grid-clean-pass", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply fixed-grid history during the timestep-zero clean cache pass.")
args = parser.parse_args()

if args.noncontiguous_kv:
    try:
        noncontiguous_source_blocks = sorted({
            int(block.strip()) for block in (args.noncontiguous_source_blocks or "").split(",") if block.strip()
        })
    except ValueError:
        parser.error("--noncontiguous-source-blocks must be comma-separated integers")
    if not noncontiguous_source_blocks or args.noncontiguous_target_block is None:
        parser.error("--noncontiguous-kv requires --noncontiguous-source-blocks and --noncontiguous-target-block")
    try:
        noncontiguous_history_frame_ids = [
            int(frame.strip()) for frame in (args.noncontiguous_history_frame_ids or "").split(",") if frame.strip()
        ]
    except ValueError:
        parser.error("--noncontiguous-history-frame-ids must be comma-separated integers")
    if args.noncontiguous_history_frame_id is not None:
        noncontiguous_history_frame_ids = [args.noncontiguous_history_frame_id]
    if args.noncontiguous_kv_mode in {"same_entity_history", "wrong_entity_history"} and \
            len(noncontiguous_history_frame_ids) != args.noncontiguous_retrieval_count:
        parser.error("oracle history modes require one distinct manual frame per retrieval slot")
else:
    noncontiguous_source_blocks = None
    noncontiguous_history_frame_ids = None

if bool(args.memory_fixed_grid_mask_path) != bool(args.memory_fixed_grid_mode):
    parser.error("--memory-fixed-grid-mask-path and --memory-fixed-grid-mode must be provided together")
if args.memory_fixed_grid_mask_path and not args.attention_memory_policy:
    parser.error("fixed-grid recall requires --attention-memory-policy")
if not 0.0 <= args.memory_fixed_grid_alpha <= 1.0:
    parser.error("--memory-fixed-grid-alpha must be between 0 and 1")
if args.memory_fixed_grid_alpha != 1.0 and not args.memory_fixed_grid_mask_path:
    parser.error("--memory-fixed-grid-alpha requires fixed-grid recall")
if args.memory_fixed_grid_denoising_steps == "clean_only" and not args.memory_fixed_grid_clean_pass:
    parser.error("clean_only requires --memory-fixed-grid-clean-pass")

if args.attention_memory_policy:
    if args.noncontiguous_kv:
        parser.error("--attention-memory-policy and --noncontiguous-kv are separate experiments and cannot be combined")
    try:
        memory_descriptor_layers = sorted({
            int(layer.strip()) for layer in args.memory_descriptor_layers.split(",") if layer.strip()
        })
        memory_manual_frame_ids = [
            int(frame.strip()) for frame in (args.memory_manual_frame_ids or "").split(",") if frame.strip()
        ]
        memory_injection_layers = sorted({
            int(layer.strip()) for layer in args.memory_injection_layers.split(",") if layer.strip()
        })
        memory_manual_target_blocks = {
            int(block.strip()) for block in (args.memory_manual_target_blocks or "").split(",") if block.strip()
        }
    except ValueError:
        parser.error("memory layer and manual frame lists must be comma-separated integers")
    if not memory_descriptor_layers or not memory_injection_layers or any(
            layer < 0 or layer >= 30 for layer in memory_descriptor_layers + memory_injection_layers):
        parser.error("memory descriptor and injection layers must select transformer layers 0 through 29")
    if any(block < 1 for block in memory_manual_target_blocks):
        parser.error("--memory-manual-target-blocks must contain positive block numbers")
    if args.memory_k < 0 or args.memory_archive_top_m < 0 or args.memory_archive_recent_scenes < 0 or \
            args.memory_archive_high_utility < 0 or args.memory_target_budget <= 0 or \
            args.memory_consolidate_n_max <= 0 or not 0.0 <= args.memory_decay_beta <= 1.0 or \
            not -1.0 <= args.memory_diversity_threshold <= 1.0:
        parser.error("memory policy numeric settings are out of range")
    if memory_manual_frame_ids and (len(memory_manual_frame_ids) != args.memory_k or
                                    len(set(memory_manual_frame_ids)) != len(memory_manual_frame_ids)):
        parser.error("--memory-manual-frame-ids requires exactly --memory-k distinct frame IDs")
    try:
        fixed_grid_config = validate_fixed_grid_options(
            args.memory_fixed_grid_mask_path, args.memory_fixed_grid_mode,
            args.attention_memory_policy, memory_manual_frame_ids or None,
            memory_manual_target_blocks or None,
            local_retention=args.memory_local_retention,
            context_mode=args.memory_context_mode)
        if fixed_grid_config:
            FixedGridMemoryMasks.from_json(fixed_grid_config["mask_path"])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    memory_policy_config = {
        "enabled": True,
        "retrieval": args.memory_retrieval,
        "context_mode": args.memory_context_mode,
        "k": args.memory_k,
        "descriptor_layers": memory_descriptor_layers,
        "injection_layers": memory_injection_layers,
        "manual_frame_ids": memory_manual_frame_ids or None,
        "manual_target_blocks": memory_manual_target_blocks or None,
        "retrieval_lifetime": args.memory_retrieval_lifetime,
        "transition_auto_retrieval": args.memory_transition_auto_retrieval,
        "local_retention": args.memory_local_retention,
        "decay": args.memory_decay,
        "decay_beta": args.memory_decay_beta,
        "cross_attention_reset": args.memory_crossattn_reset,
        "archive": args.memory_archive,
        "archive_top_m": args.memory_archive_top_m,
        "archive_recent_scenes": args.memory_archive_recent_scenes,
        "archive_high_utility": args.memory_archive_high_utility,
        "consolidation": args.memory_consolidation,
        "consolidate_n_max": args.memory_consolidate_n_max,
        "target_budget": args.memory_target_budget,
        "diversity_threshold": args.memory_diversity_threshold,
        "log_path": args.memory_policy_log or os.path.join(args.output_folder, "memory_policy.jsonl"),
    }
    if fixed_grid_config:
        fixed_grid_config["alpha"] = args.memory_fixed_grid_alpha
        fixed_grid_config["denoising_steps"] = args.memory_fixed_grid_denoising_steps
        fixed_grid_config["clean_pass"] = args.memory_fixed_grid_clean_pass
        memory_policy_config["fixed_grid"] = fixed_grid_config
else:
    memory_policy_config = None

try:
    clean_latent_snapshot_blocks = {
        int(block.strip()) for block in (args.save_clean_latent_blocks or "").split(",") if block.strip()
    }
except ValueError:
    parser.error("--save-clean-latent-blocks must be comma-separated integers")
if any(block < 1 for block in clean_latent_snapshot_blocks):
    parser.error("--save-clean-latent-blocks must contain positive block numbers")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    generator_state_dict = state_dict['generator' if not args.use_ema else 'generator_ema']
    
    # Fix FSDP checkpoint loading by removing _fsdp_wrapped_module prefix
    def rename_param(name):
        return name.replace("_fsdp_wrapped_module.", "")
    
    # Create a new state dict with renamed parameters
    renamed_state_dict = {}
    for name, param in generator_state_dict.items():
        renamed_name = rename_param(name)
        renamed_state_dict[renamed_name] = param
    
    pipeline.generator.load_state_dict(renamed_state_dict)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

subtitles = ''
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt_and_subtitles = batch['prompts'][0]
        # Ensure ';' exists for subtitle parsing (add if missing)
        if ';' not in prompt_and_subtitles:
            prompt_and_subtitles = prompt_and_subtitles + ';'
        prompt = prompt_and_subtitles.split(';')[0]  # Get caption from batch
        subtitles = prompt_and_subtitles.split(';')[1]  # Get subtitles from batch (empty string if no subtitles)
        print(prompt)
        prompts = [prompt] * args.num_samples
        extended_prompt = None  # i2v doesn't use extended prompts
        prompt_for_duration = prompt

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
        has_initial_latent = True
    else:
        # For text-to-video, batch is just the text prompt
        prompt_and_subtitles = batch['prompts'][0]
        # Ensure ';' exists for subtitle parsing (add if missing)
        if ';' not in prompt_and_subtitles:
            prompt_and_subtitles = prompt_and_subtitles + ';'
        prompt = prompt_and_subtitles.split(';')[0]  # Get caption from batch
        subtitles = prompt_and_subtitles.split(';')[1]  # Get subtitles from batch (empty string if no subtitles)
        print(prompt)
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
            prompt_for_duration = extended_prompt
        else:
            prompts = [prompt] * args.num_samples
            prompt_for_duration = prompt
        initial_latent = None
        has_initial_latent = False

    # Determine number of output frames based on duration or use provided value
    total_duration = parse_durations_from_prompt(prompt_for_duration)
    
    if total_duration is not None:
        # Mode 1: Calculate frames from duration
        fps = 16.0
        temporal_compression = 4
        num_frame_per_block = getattr(pipeline, 'num_frame_per_block', getattr(config, 'num_frame_per_block', 1))
        independent_first_frame = getattr(pipeline, 'independent_first_frame', getattr(config, 'independent_first_frame', False))
        
        num_latent_frames = calculate_latent_frames_from_duration(
            total_duration, fps, temporal_compression, num_frame_per_block,
            independent_first_frame, has_initial_latent
        )
        
        print(f"Duration-based frame calculation: {total_duration}s -> {num_latent_frames} latent frames")
        if args.num_output_frames is not None:
            print(f"Warning: --num_output_frames ({args.num_output_frames}) is ignored when durations are specified in prompt")
    else:
        # Mode 2: Require --num_output_frames
        if args.num_output_frames is None:
            raise ValueError("--num_output_frames must be provided when prompt does not contain duration information")
        num_latent_frames = args.num_output_frames
        if has_initial_latent:
            # For i2v, subtract 1 because first frame is provided
            num_latent_frames = args.num_output_frames - 1

    # Create noise tensor with calculated number of frames
    if has_initial_latent:
        sampled_noise = torch.randn(
            [args.num_samples, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        sampled_noise = torch.randn(
            [args.num_samples, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    def save_clean_latent(block, clean_latents, _cache):
        if block in clean_latent_snapshot_blocks:
            torch.save(
                clean_latents.detach().cpu().clone(),
                os.path.join(args.output_folder, f"{idx}_clean_latents_block_{block:02d}.pt"))

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
        noncontiguous_source_blocks=noncontiguous_source_blocks,
        noncontiguous_target_block=args.noncontiguous_target_block if args.noncontiguous_kv else None,
        noncontiguous_mode=args.noncontiguous_kv_mode,
        noncontiguous_retrieval_count=args.noncontiguous_retrieval_count,
        noncontiguous_random_seed=args.seed,
        noncontiguous_manual_frame_id=args.noncontiguous_history_frame_id,
        noncontiguous_manual_frame_ids=noncontiguous_history_frame_ids,
        memory_policy_config=memory_policy_config,
        clean_pass_callback=save_clean_latent if clean_latent_snapshot_blocks else None,
    )
    if args.save_raw_decoded:
        torch.save(
            video.detach().cpu().clone(),
            os.path.join(args.output_folder, f"{idx}_raw_decoded_before_mp4.pt"))
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Parse time durations from actions (before ';') for subtitle alignment
    prompt_for_timing = extended_prompt if extended_prompt is not None else prompt
    action_durations = None
    if prompt_for_timing:
        # Parse durations from prompt actions (format: "text[5s] | text[10s]")
        scene_parts = [part.strip() for part in prompt_for_timing.split('|')]
        action_durations = []
        for scene_part in scene_parts:
            # Look for duration pattern: [5s], [15s#], [10.5s], etc.
            duration_match = re.search(r'\[(\d+\.?\d*)\s*s[#]?\]', scene_part)
            if duration_match:
                duration_seconds = float(duration_match.group(1))
                action_durations.append(duration_seconds)
            else:
                # If any action doesn't have a duration, don't use durations
                action_durations = None
                break
    
    # Parse subtitles (after ';') and align with action durations
    subtitle_list = []
    if subtitles and subtitles.strip():
        # Split subtitles by '|' and strip whitespace
        subtitle_list = [s.strip() for s in subtitles.split('|')]
    else:
        # No subtitles provided, create empty list
        subtitle_list = []
    
    # Align subtitles with action durations (one subtitle per action)
    if action_durations is not None:
        # Align subtitles with durations: one subtitle per action duration
        # If fewer subtitles than actions, pad with empty strings
        # If more subtitles than actions, truncate to match actions
        if len(subtitle_list) < len(action_durations):
            subtitle_list.extend([""] * (len(action_durations) - len(subtitle_list)))
        elif len(subtitle_list) > len(action_durations):
            subtitle_list = subtitle_list[:len(action_durations)]
        
        # Use durations for subtitle timing
        time_durations = action_durations
    else:
        # No durations available, use None (will fall back to equal division)
        time_durations = None
    
    # Only add subtitles if we have at least one non-empty subtitle
    if subtitle_list and any(s.strip() for s in subtitle_list):
        video = add_subtitles(video, subtitle_list, fps=16.0, time_durations=time_durations)

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # Use output_index if provided, otherwise use seed value, otherwise use seed_idx
            if args.output_index is not None:
                file_idx = args.output_index
            elif args.num_samples == 1:
                # When generating single sample, use seed value in filename
                file_idx = args.seed
            else:
                file_idx = seed_idx
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{file_idx}_{model}.mp4')
            else:
                safe_prompt = sanitize_filename(prompt, max_length=100)
                output_path = os.path.join(args.output_folder, f'{safe_prompt}-{file_idx}.mp4')
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video_file(output_path, video[seed_idx], fps=16)
