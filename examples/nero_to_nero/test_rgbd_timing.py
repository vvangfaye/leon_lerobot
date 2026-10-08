"""零值/停滞/回退时钟与旧报告回放，不访问相机或 CAN。"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import smoke_femto
from nero_platform import femto, rgbd_timing, timing
from nero_platform.review import LocalReview


def sample(capture: float, device=(0, 0), global_=(0, 0), index: int = 1) -> dict:
    return {
        "capture_timestamp": capture,
        "age_ms": 1,
        "rgb_timestamp_us": device[0],
        "depth_timestamp_us": device[1],
        "rgb_global_timestamp_us": global_[0],
        "depth_global_timestamp_us": global_[1],
        "rgb_frame_index": index,
        "depth_frame_index": index,
        "invalid_fraction": 0,
        "above_range_fraction": 0,
    }


@pytest.mark.parametrize(
    "device,global_",
    [
        ((0, 0), (0, 0)),
        ((None, None), (None, None)),
        ((float("nan"), 100), (0, 0)),
        ((-1, 100), (0, 0)),
        ((100, 0), (0, 100)),
        ((True, 100), (0, 0)),
    ],
)
def test_invalid_timestamps_never_become_zero_skew(device, global_):
    tracker = rgbd_timing.TimestampTracker()
    for capture in (1, 2):
        result = tracker.assess(sample(capture, device, global_))
        assert result["timestamp_status"] == "unavailable"
        assert result["rgb_depth_skew_ms"] is None
        assert result["timestamp_source"] is None


def test_progress_stalls_resets_and_recovers_without_mixing_clock_domains():
    tracker = rgbd_timing.TimestampTracker()
    assert tracker.assess(sample(1, (100, 200)))["timestamp_status"] == "warming_up"
    assert tracker.assess(sample(2, (200, 300)))["rgb_depth_skew_ms"] == 0.1
    assert tracker.assess(sample(3, (200, 300), index=3))["timestamp_status"] == "stalled"
    assert tracker.assess(sample(4, (100, 200), index=4))["timestamp_status"] == "regressed"
    result = tracker.assess(sample(5, (200, 300), index=5))
    assert result["timestamp_status"] == "valid"
    assert result["timestamp_source"] == "device"
    assert tracker.assess(sample(5, (200, 300), index=5)) == result, "相同缓存帧不能误报时钟停滞"
    tracker.assess(sample(6, (0, 300), (1_000_000, 1_000_100)))
    result = tracker.assess(sample(7, (0, 400), (1_033_333, 1_033_433)))
    assert result["timestamp_source"] == "global"
    assert result["rgb_depth_skew_ms"] == 0.1


def test_background_publishes_images_with_missing_metadata_and_preserves_duplicate_age():
    camera = femto.FemtoCamera()
    camera._publish(sample(1, index=0))
    camera._publish(sample(2, index=0))
    assert camera._latest["capture_timestamp"] == 2
    assert camera._latest["timestamp_status"] == "unavailable"
    camera._publish(sample(3, (100, 100), index=1))
    camera._publish(sample(4, (200, 200), index=2))
    assert camera._latest["timestamp_status"] == "valid"
    camera._publish(sample(5, (200, 200), index=2))
    assert camera._latest["capture_timestamp"] == 4, "同一 SDK 帧不能刷新帧龄"
    camera._publish(sample(6, (200, 200), index=3))
    assert camera._latest["timestamp_status"] == "stalled"


def test_metadata_api_failure_is_nonfatal():
    frame = SimpleNamespace(get_timestamp_us=Mock(side_effect=RuntimeError("metadata unavailable")))
    assert femto.frame_number(frame, "get_timestamp_us") is None
    assert femto.frame_number(frame, "get_global_timestamp_us") is None


def test_receive_gap_is_a_separate_backlog_check_in_report_and_review():
    cadence = timing.EpisodeTiming(30, 0)
    entry = sample(1)
    entry.update(rgb_system_timestamp_us=1_000_000, depth_system_timestamp_us=1_400_000)
    cadence.finish(0, 0, 0.001, 0, None, {"depth": entry})
    summary = cadence.summary(1 / 30)
    stats = summary["cameras"]["depth"]
    assert stats["max_rgb_depth_receive_skew_ms"] == 400
    assert stats["timestamp_checked_frames"] == 0
    assert stats["max_rgb_depth_skew_ms"] is None
    assert any("旧帧积压" in message for message in summary["warnings"])
    events = LocalReview.events({"samples": cadence.samples}, [0], 30)
    assert "主机接收时差 400.0 ms" in events[0]["message"]
    assert rgbd_timing.receive_skew_ms(sample(1)) is None
    assert rgbd_timing.receive_skew_limit_ms(15) == 200


def test_report_counts_only_checked_pairs_and_localizes_unknown_transition():
    cadence = timing.EpisodeTiming(30, 0)
    data = [sample(1), sample(2, (100, 100)), sample(3, (200, 50_200)), sample(4, (200, 50_200), index=4)]
    for i, entry in enumerate(data):
        cadence.finish(i, i / 30, i / 30 + 0.001, i, None, {"depth": entry})
    summary = cadence.summary(4 / 30)
    stats = summary["cameras"]["depth"]
    assert stats["timestamp_checked_frames"] == 1
    assert stats["timestamp_unverified_frames"] == 3
    assert stats["max_rgb_depth_skew_ms"] == 50
    assert any("有效时间戳差超过" in message for message in summary["warnings"])
    assert any("时间同步未验证" in message for message in summary["warnings"])
    events = LocalReview.events({"samples": cadence.samples}, [i / 30 for i in range(4)], 30)
    assert [event["frame"] for event in events] == [0, 1, 2, 3]
    assert "不可用" in events[0]["message"]
    assert "未推进" in events[-1]["message"]


@pytest.mark.parametrize("raw_value", [0, None])
def test_legacy_zero_summary_is_recomputed_without_overwriting_report(tmp_path, raw_value):
    root = tmp_path / "dataset"
    path = timing.report_path(root, 0)
    path.parent.mkdir(parents=True)
    cadence = timing.EpisodeTiming(30, 0)
    for i in range(3):
        cadence.finish(
            i, i / 30, i / 30 + 0.001, i, None, {"depth": sample(i + 1, (raw_value, raw_value), index=i)}
        )
    report = {
        "schema_version": 2,
        "episode_index": 0,
        "summary": cadence.summary(0.1),
        "samples": cadence.samples,
    }
    for entry in report["samples"]:
        for key in rgbd_timing.CHECK_FIELDS:
            entry["cameras"]["depth"].pop(key, None)
    stats = report["summary"]["cameras"]["depth"]
    for key in list(stats):
        if key.startswith("timestamp_"):
            del stats[key]
    stats["max_rgb_depth_skew_ms"] = 0
    report["summary"]["warnings"] = []
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    quality, normalized = LocalReview(tmp_path).report(root, {"episode_index": 0, "length": 3}, 30)
    assert normalized is not None, quality
    corrected = quality["summary"]["cameras"]["depth"]
    assert corrected["max_rgb_depth_skew_ms"] is None
    assert corrected["timestamp_checked_frames"] == 0
    assert corrected["timestamp_unverified_frames"] == 3
    assert path.read_bytes() == original
    events = LocalReview.events(normalized, [0, 1 / 30, 2 / 30], 30)
    assert len(events) == 1, "连续未知状态无需为每帧产生同一条告警"


def test_smoke_requires_explicit_run_and_closes_camera_after_error():
    with patch.object(smoke_femto.femto, "FemtoCamera") as constructor:
        assert smoke_femto.main([]) == 0
    constructor.assert_not_called()
    camera = SimpleNamespace(
        connect=Mock(side_effect=RuntimeError("模拟打开失败")), disconnect=Mock(), fps=30
    )
    report = smoke_femto.run_smoke(camera, 1)
    assert report["result"] == "failed"
    assert report["errors"] == ["模拟打开失败"]
    camera.disconnect.assert_called_once()


def test_smoke_checks_live_frames_and_valid_timestamps():
    data = [sample(1, (100, 100), index=1), sample(2, (200, 200), index=2), sample(3, (300, 300), index=3)]
    tracker = rgbd_timing.TimestampTracker()
    for entry in data:
        entry.update(tracker.assess(entry))
        entry["rgb"] = np.zeros((2, 2, 3), np.uint8)
        entry["depth"] = np.ones((2, 2, 1), np.float32)
    camera = SimpleNamespace(
        connect=Mock(),
        disconnect=Mock(),
        fps=30,
        output_size=(2, 2),
        metadata={},
        read_snapshot=Mock(side_effect=copy.deepcopy(data)),
    )
    with patch.object(smoke_femto.time, "perf_counter", side_effect=[0, 0.1, 0.2, 0.3, 1.1, 1.1]):
        report = smoke_femto.run_smoke(camera, 1)
    assert report["result"] == "passed", report
    assert report["timestamp_stats"]["timestamp_checked_frames"] == 2
    camera.disconnect.assert_called_once()
