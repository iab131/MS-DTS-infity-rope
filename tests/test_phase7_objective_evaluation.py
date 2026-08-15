"""Small deterministic checks for Phase-7 metric definitions."""

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "phase7_objective_evaluation",
    Path(__file__).parents[1] / "scripts" / "phase7_objective_evaluation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_hard_cut_metrics_detect_source_to_target_change():
    import numpy as np

    # Eight source frames, then a five-frame transition into a stable target.
    features = np.vstack(
        [np.tile([1.0, 0.0], (8, 1)),
         np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8], [0.0, 1.0]]),
         np.tile([0.0, 1.0], (8, 1))]
    )
    result = MODULE.hard_cut_metrics(
        features, cut_frame=8, source_count=8, stable_start=5, stable_stop=13,
        required_stable_frames=2,
    )

    assert result["source_similarity"][0] > result["source_similarity"][-1]
    assert result["target_similarity"][0] < result["target_similarity"][-1]
    assert result["transition_latency_frames"] == 4


def test_continuity_metrics_flag_chromatic_temporal_noise():
    import numpy as np

    rng = np.random.default_rng(7)
    stable = np.full((10, 8, 8, 3), 0.45, dtype=np.float32)
    noisy = rng.random((8, 8, 3), dtype=np.float32)
    frames = np.concatenate([stable, noisy[None]], axis=0)
    features = np.vstack([np.tile([1.0, 0.0], (10, 1)), [[0.0, 1.0]]])
    result = MODULE.continuity_metrics(features, frames, boundary_frame=10, pre_count=8)

    assert result["rgb_step_ratio"] > 10.0
    assert result["feature_step_ratio"] > 10.0
    assert result["post_colorfulness"] > result["pre_colorfulness"]


def test_review_reader_strips_copied_csv_header_whitespace(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("case_id   , score_1_to_5, notes\ncase, 4, clean\n")

    rows = MODULE._read_csv_rows(path)

    assert rows == [{"case_id": "case", "score_1_to_5": "4", "notes": "clean"}]
