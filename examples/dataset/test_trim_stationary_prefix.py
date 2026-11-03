import importlib.util
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "trim_prefix", Path(__file__).with_name("trim_stationary_prefix.py")
)
trim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trim)


def test_removes_hold_and_preserves_two_frames_of_context():
    actions = np.zeros((15, 8))
    actions[6:, 0] = 0.1
    assert trim.trim_index(actions, fps=10) == 4


def test_isolated_motion_spike_does_not_trigger_start():
    actions = np.zeros((30, 8))
    actions[2, 0] = 0.1
    actions[15:, 0] = 0.1
    assert trim.trim_index(actions, fps=30) == 9


def test_gripper_motion_can_start_an_episode():
    actions = np.zeros((30, 8))
    actions[12:, 7] = 0.01
    assert trim.trim_index(actions, fps=30) == 6


def test_stationary_or_immediately_active_episode_is_kept():
    assert trim.trim_index(np.zeros((30, 8)), 30) == 0
    actions = np.zeros((30, 8))
    actions[1:, 0] = 0.1
    assert trim.trim_index(actions, 30) == 0


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        trim.trim_index(np.full((30, 8), np.nan), 30)


def test_video_frame_cut_has_exact_count_and_zero_timestamp(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=10, options={"crf": "0", "g": "1"})
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), 40 + index * 10, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    stats = trim.trim_video(source, output, cut=3, original_length=10, fps=10)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 7
    assert frames[0].time == 0
    assert frames[-1].time == pytest.approx(0.6)
    assert frames[0].to_ndarray(format="rgb24").mean() == pytest.approx(70, abs=3)
    assert stats["count"].tolist() == [7]
