#!/usr/bin/env python3
"""Validate and run the registered hard-cut and same-scene transition matrices."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path):
    """Load the checked-in declarative benchmark manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    phase0_arms = {"live_kv_flush", "sink_plus1", "sink_only", "transition_no_sink"}
    phase2b_arms = {"transition_no_sink", "transition_no_sink_scene_local_rope_epoch"}
    phase3a_arms = {"live_kv_flush", "transition_no_sink"}
    phase3b_arms = {"live_kv_flush", "sink_only", "recent_only_no_sink", "transition_no_sink"}
    phase3c_arms = {"live_kv_flush", "always_reset", "boundary_conditioned"}
    phase5_arms = {"live_infinity_rope", "always_reset", "native_state_rebinding"}
    if manifest.get("benchmark_id") == "hard_cut_transition_phase0_20260810":
        if len(manifest.get("pairs", [])) != 4 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase0_arms:
            raise ValueError("Phase-0 benchmark requires four pairs, two seeds, and four live arms")
    elif manifest.get("benchmark_id") == "hard_cut_scene_local_rope_epoch_phase2b_20260810":
        if len(manifest.get("pairs", [])) != 2 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase2b_arms:
            raise ValueError("Phase-2B benchmark requires two pairs, two seeds, and two no-sink arms")
    elif manifest.get("benchmark_id") == "same_scene_action_transition_phase3a_20260811":
        if len(manifest.get("pairs", [])) != 2 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase3a_arms or \
                manifest.get("uses_hard_cut", True):
            raise ValueError("Phase-3A requires two pairs, two seeds, live/no-sink arms, and normal boundaries")
    elif manifest.get("benchmark_id") in {
            "hard_cut_state_retention_factorial_phase3b_20260811",
            "same_scene_state_retention_factorial_phase3b_20260811"}:
        if len(manifest.get("pairs", [])) != 2 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase3b_arms:
            raise ValueError("Phase-3B requires two pairs, two seeds, and the four retention arms")
    elif manifest.get("benchmark_id") == "mixed_boundary_state_lifetime_phase3c_20260812":
        if len(manifest.get("pairs", [])) != 2 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase3c_arms:
            raise ValueError("Phase-3C requires two mixed-boundary scenarios, two seeds, and three arms")
        if any(len(pair.get("segments", [])) != 4 for pair in manifest["pairs"]):
            raise ValueError("Phase-3C requires A1 | A2 # B1 | B2 segments")
    elif manifest.get("benchmark_id") == "phase5_generalization_checkpoint_20260813":
        if len(manifest.get("categories", [])) != 7 or len(manifest.get("pairs", [])) != 7 or \
                manifest.get("seeds") != [101, 202, 303] or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase5_arms:
            raise ValueError("Phase-5 checkpoint requires seven categories, seven storyboards, three fixed seeds, and three main arms")
        for pair in manifest["pairs"]:
            if len(pair.get("segments", [])) != 6 or pair.get("boundary_after") != ["|", "#", "|", "#", "|"]:
                raise ValueError("Phase-5 requires A1 | A2 # B1 | B2 # C1 | C2")
            if pair.get("transition_blocks") != [5, 9, 13, 17, 21] or \
                    pair.get("transition_raw_frames") != [46, 94, 142, 190, 238]:
                raise ValueError("Phase-5 schedule must use four blocks per 3.0-second scene")
            rendered = _render_segments(pair, manifest["duration_seconds_per_scene"])
            prompt_path = ROOT / pair["prompt_path"]
            if not prompt_path.is_file() or prompt_path.read_text(encoding="utf-8").strip() != rendered:
                raise ValueError(f"Phase-5 prompt file mismatch: {prompt_path}")
    else:
        raise ValueError("unsupported hard-cut benchmark manifest")
    prompts = (segment for pair in manifest["pairs"] for segment in pair.get("segments", [pair.get("a", ""), pair.get("b", "")]))
    if any("#" in prompt or "|" in prompt for prompt in prompts):
        raise ValueError("pair prompts must not include scheduling syntax")
    return manifest


def _render_segments(pair, duration):
    boundaries = pair.get("boundary_after")
    if boundaries is None:
        boundaries = ["|", "#", "|"]
    if len(boundaries) != len(pair["segments"]) - 1:
        raise ValueError("segment boundary count must be one less than segment count")
    rendered = []
    for index, segment in enumerate(pair["segments"]):
        marker = "#" if index < len(boundaries) and boundaries[index] == "#" else ""
        rendered.append(f"{segment}[{duration}s{marker}]")
    return " | ".join(rendered)


def _command(manifest, pair, seed, arm):
    settings = manifest["matched_settings"]
    output_root = arm.get("reuse_output_root", settings["output_root"])
    output = f'{output_root}/{pair["id"]}/seed_{seed}/{arm["id"]}'
    if "segments" in pair:
        prompt = _render_segments(pair, manifest["duration_seconds_per_scene"])
    else:
        cut_marker = "#" if manifest.get("uses_hard_cut", True) else ""
        prompt = f'{pair["a"]}[{manifest["duration_seconds_per_scene"]}s{cut_marker}] | {pair["b"]}[{manifest["duration_seconds_per_scene"]}s]'
    command = [
        "python", "inference.py", "--config_path", settings["config_path"],
        "--checkpoint_path", settings["checkpoint_path"],
        "--data_path", pair["prompt_path"],
        "--output_folder", output, "--seed", str(seed), "--num_samples", str(settings["num_samples"]),
        "--output_index", str(settings["output_index"]),
        "--save-clean-latent-blocks", ",".join(map(str, settings["save_clean_latent_blocks"])),
    ]
    if settings["use_ema"]:
        command.append("--use_ema")
    if settings["save_with_index"]:
        command.append("--save_with_index")
    if settings["save_raw_decoded"]:
        command.append("--save-raw-decoded")
    if arm["attention_memory_policy"]:
        command.extend([
            "--attention-memory-policy", "--memory-context-mode", settings["memory_context_mode"],
            "--memory-k", str(settings["memory_k"]),
            "--memory-descriptor-layers", settings["memory_descriptor_layers"],
            "--memory-injection-layers", settings["memory_injection_layers"],
            "--memory-local-retention", arm["local_retention"],
        ])
        for setting, flag in (
            ("memory_retrieval", "--no-memory-retrieval"),
            ("memory_decay", "--no-memory-decay"),
            ("memory_archive", "--no-memory-archive"),
            ("memory_consolidation", "--no-memory-consolidation"),
            ("memory_transition_auto_retrieval", "--no-memory-transition-auto-retrieval"),
        ):
            if not settings[setting]:
                command.append(flag)
        command.append("--memory-crossattn-reset" if settings["memory_crossattn_reset"]
                       else "--no-memory-crossattn-reset")
        if arm.get("scene_local_rope_epoch", False):
            command.append("--scene-local-rope-epoch")
    if arm.get("boundary_conditioned_ar_state", False):
        command.append("--boundary-conditioned-ar-state")
    return prompt, output, command


def build_run_rows(manifest):
    """Expand four pairs × two seeds × four arms into non-executing run records."""
    rows = []
    for pair in manifest["pairs"]:
        for seed in manifest["seeds"]:
            for arm in manifest["arms"]:
                prompt, output, command = _command(manifest, pair, seed, arm)
                transitions = pair.get("transition_blocks", [4, 7, 10] if "segments" in pair else [4])
                rows.append({
                    "run_id": f'{pair["id"]}__seed{seed}__{arm["id"]}',
                    "pair_id": pair["id"], "seed": seed, "arm_id": arm["id"],
                    "prompt": prompt, "output_folder": output, "command": command,
                "first_b_block": transitions[0], "first_b_raw_frame":
                    pair.get("transition_raw_frames", [33])[0],
                "transition_blocks": transitions,
                "transition_raw_frames": pair.get("transition_raw_frames", [33]),
                    "status": "planned_not_run", "review_fields": manifest["review_fields"],
                } | ({"reuse_ledger": arm["reuse_ledger"]} if "reuse_ledger" in arm else {}))
    return rows


def _gpu_memory_for_pid(pid):
    """Return this direct inference process's current VRAM use, if available."""
    probe = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if probe.returncode:
        return None
    for line in probe.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def execute_rows(rows):
    """Run planned rows serially and record runtime, direct-process VRAM, and outputs."""
    for row in rows:
        if "reuse_ledger" in row:
            ledger_path = ROOT / row["reuse_ledger"]
            prior = json.loads(ledger_path.read_text(encoding="utf-8"))
            prior_row = next((candidate for candidate in prior["runs"]
                              if candidate["run_id"] == row["run_id"]), None)
            if prior_row is None or prior_row.get("status") != "completed" or \
                    prior_row.get("command") != row["command"]:
                raise ValueError(f"reuse provenance mismatch for {row['run_id']}")
            output_folder = ROOT / row["output_folder"]
            if not (output_folder / "0_raw_decoded_before_mp4.pt").is_file() or \
                    not (output_folder / "0-0_ema.mp4").is_file():
                raise FileNotFoundError(f"reused artifact incomplete: {output_folder}")
            row.update({
                "status": "reused_exact_provenance", "returncode": prior_row.get("returncode"),
                "runtime_seconds": prior_row.get("runtime_seconds"),
                "peak_vram_mib": prior_row.get("peak_vram_mib"),
                "output_metadata": prior_row.get("output_metadata"),
                "reused_from_ledger": row["reuse_ledger"],
            })
            continue
        started = time.monotonic()
        process = subprocess.Popen(row["command"], cwd=ROOT)
        peak_vram_mib = 0
        sampled_vram = False
        while process.poll() is None:
            current_vram_mib = _gpu_memory_for_pid(process.pid)
            if current_vram_mib is not None:
                peak_vram_mib = max(peak_vram_mib, current_vram_mib)
                sampled_vram = True
            time.sleep(0.5)
        output_folder = ROOT / row["output_folder"]
        row.update({
            "status": "completed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_vram_mib": peak_vram_mib if sampled_vram else None,
            "output_metadata": {"output_folder": str(output_folder),
                                "exists": output_folder.exists()},
        })
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate_artifacts(rows, root=ROOT, git_commit=None):
    """Attach mechanical provenance and live-reference RGB divergence to completed rows."""
    if git_commit is None:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True)
        git_commit = result.stdout.strip()
    by_case = {}
    for row in rows:
        if row.get("status") not in {"completed", "reused_exact_provenance"}:
            continue
        output = Path(root) / row["output_folder"]
        raw, mp4 = output / "0_raw_decoded_before_mp4.pt", output / "0-0_ema.mp4"
        if not raw.is_file() or not mp4.is_file():
            raise FileNotFoundError(f"missing completed artifacts under {output}")
        row.update({"git_commit": git_commit, "raw_output_sha256": _sha256(raw),
                    "mp4_sha256": _sha256(mp4)})
        by_case.setdefault((row["pair_id"], row["seed"]), []).append((row, raw))
    for case_rows in by_case.values():
        reference = next((raw for row, raw in case_rows if row["arm_id"] in {
            "live_kv_flush", "live_infinity_rope"}), None)
        if reference is None:
            raise ValueError("artifact divergence requires a live reference arm")
        reference_video = torch.load(reference, map_location="cpu", weights_only=True)
        for row, raw in case_rows:
            video = torch.load(raw, map_location="cpu", weights_only=True)
            if tuple(video.shape) != tuple(reference_video.shape):
                raise ValueError(f"raw video shape mismatch for {row['run_id']}")
            different = (video != reference_video).reshape(video.shape[0], video.shape[1], -1).any(dim=2).any(dim=0)
            indices = different.nonzero(as_tuple=False).flatten().tolist()
            row["raw_first_divergence_from_live_rgb_frame"] = None if not indices else indices[0] + 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional JSON path for planned rows; never executes inference.")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the planned rows serially (requires explicit later approval).")
    args = parser.parse_args()
    rows = build_run_rows(load_manifest(args.manifest))
    if args.execute:
        rows = execute_rows(rows)
        rows = annotate_artifacts(rows)
    payload = {"status": "completed" if args.execute else "planned_not_run",
               "run_count": len(rows), "runs": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
