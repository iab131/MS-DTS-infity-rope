#!/usr/bin/env python3
"""Validate and enumerate, but never execute, the Phase-0 hard-cut matrix."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path):
    """Load the checked-in declarative benchmark manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    phase0_arms = {"live_kv_flush", "sink_plus1", "sink_only", "transition_no_sink"}
    phase2b_arms = {"transition_no_sink", "transition_no_sink_scene_local_rope_epoch"}
    if manifest.get("benchmark_id") == "hard_cut_transition_phase0_20260810":
        if len(manifest.get("pairs", [])) != 4 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase0_arms:
            raise ValueError("Phase-0 benchmark requires four pairs, two seeds, and four live arms")
    elif manifest.get("benchmark_id") == "hard_cut_scene_local_rope_epoch_phase2b_20260810":
        if len(manifest.get("pairs", [])) != 2 or len(manifest.get("seeds", [])) != 2 or \
                {arm["id"] for arm in manifest.get("arms", [])} != phase2b_arms:
            raise ValueError("Phase-2B benchmark requires two pairs, two seeds, and two no-sink arms")
    else:
        raise ValueError("unsupported hard-cut benchmark manifest")
    if any("#" in pair["a"] or "|" in pair["a"] or "#" in pair["b"] or "|" in pair["b"]
           for pair in manifest["pairs"]):
        raise ValueError("pair prompts must not include scheduling syntax")
    return manifest


def _command(manifest, pair, seed, arm):
    settings = manifest["matched_settings"]
    output = f'{settings["output_root"]}/{pair["id"]}/seed_{seed}/{arm["id"]}'
    prompt = f'{pair["a"]}[{manifest["duration_seconds_per_scene"]}s#] | {pair["b"]}[{manifest["duration_seconds_per_scene"]}s]'
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
    return prompt, output, command


def build_run_rows(manifest):
    """Expand four pairs × two seeds × four arms into non-executing run records."""
    rows = []
    for pair in manifest["pairs"]:
        for seed in manifest["seeds"]:
            for arm in manifest["arms"]:
                prompt, output, command = _command(manifest, pair, seed, arm)
                rows.append({
                    "run_id": f'{pair["id"]}__seed{seed}__{arm["id"]}',
                    "pair_id": pair["id"], "seed": seed, "arm_id": arm["id"],
                    "prompt": prompt, "output_folder": output, "command": command,
                    "first_b_block": 4, "first_b_raw_frame": 33,
                    "status": "planned_not_run", "review_fields": manifest["review_fields"],
                })
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
    payload = {"status": "completed" if args.execute else "planned_not_run",
               "run_count": len(rows), "runs": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
