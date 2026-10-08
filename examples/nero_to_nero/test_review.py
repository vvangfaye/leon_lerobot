"""回放只读契约：真实 Parquet、索引偏移、安全路径及真实 LeRobot 保存结果。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nero_platform import camera_io, cameras, devices, recorder, timing
from nero_platform.app import create_app
from nero_platform.review import LocalReview
from nero_platform.runtime import Runtime

from lerobot.datasets import LeRobotDataset


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="nero-review-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runtime = Runtime(dataset_root=self.root)
        self.client = create_app(self.runtime).test_client()
        self.reader = LocalReview(self.root)
        for guard in (
            patch.object(devices.AgxArmFactory, "create_arm", side_effect=AssertionError("禁止访问真机")),
            patch.object(cameras, "detect_cameras", side_effect=AssertionError("回放不得扫描相机")),
            patch.object(camera_io, "RecordingCamera", side_effect=AssertionError("禁止打开相机")),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.fixture()

    def fixture(self, repo: str = "nero/review", mode: str = "dual") -> Path:
        root = self.root / repo
        for path in (
            "meta/episodes/chunk-000",
            "meta/nero_timing",
            "data/chunk-000",
            "videos/observation.images.right/chunk-000",
        ):
            (root / path).mkdir(parents=True, exist_ok=True)
        features = recorder.make_features({"right": {"type": "rgb", "path": "/fake/right"}}, mode)
        features["observation.images.right"]["info"] = {"video.codec": "av1"}
        info = {
            "codebase_version": "v3.0",
            "fps": 10,
            "total_episodes": 2,
            "total_frames": 6,
            "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        self.write_json(root / "meta/info.json", info)
        dimensions = 16 if mode == "dual" else 8
        data, rows = [], []
        for episode in range(2):
            row = {
                "episode_index": episode,
                "length": 3,
                "tasks": ["拿起 <杯子>"],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/observation.images.right/chunk_index": 0,
                "videos/observation.images.right/file_index": 0,
                "videos/observation.images.right/from_timestamp": episode * 0.3,
                "videos/observation.images.right/to_timestamp": (episode + 1) * 0.3,
            }
            rows.append(row)
            for frame in range(3):
                values = [episode + frame / 10 + i / 100 for i in range(dimensions)]
                data.append(
                    {
                        "episode_index": episode,
                        "frame_index": frame,
                        "timestamp": frame / 10,
                        "action": values,
                        "observation.state": [v - 0.01 for v in values],
                    }
                )
            report = timing.EpisodeTiming(10, 0)
            for frame in range(3):
                report.finish(
                    frame,
                    frame / 10,
                    frame / 10 + 0.001,
                    frame,
                    None,
                    {"right": {"capture_timestamp": frame / 10, "age_ms": 2}},
                )
            self.write_json(
                timing.report_path(root, episode),
                {
                    "schema_version": 2,
                    "episode_index": episode,
                    "summary": report.summary(0.3),
                    "samples": report.samples,
                },
            )
        pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(data), root / "data/chunk-000/file-000.parquet")
        (root / "videos/observation.images.right/chunk-000/file-000.mp4").write_bytes(b"0123456789abcdef")
        return root

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def get(self, endpoint: str = "episode", **params: object):
        return self.client.get(
            f"/api/review/{endpoint}",
            query_string={
                "repo_id": "nero/review",
                "episode_index": 1,
                **params,
            },
        )

    def test_listing_pagination_and_raw_dual_values(self) -> None:
        listing = self.get("episodes", offset=1, limit=1)
        self.assertEqual(listing.status_code, 200)
        self.assertEqual([e["index"] for e in listing.json["episodes"]], [0])
        episode = self.get().json
        self.assertEqual(episode["frames"], 3)
        self.assertEqual(episode["pair_mode"], "dual")
        self.assertEqual(episode["names"][8], "right.joint1.pos")
        np.testing.assert_allclose(episode["action"][0], [1 + i / 100 for i in range(16)])
        self.assertEqual(episode["timestamps"], [0, 0.1, 0.2])
        self.assertEqual(episode["quality"]["state"], "ok")
        self.assertEqual(episode["tasks"], ["拿起 <杯子>"])

    def test_single_and_shared_mp4_segment_offsets(self) -> None:
        self.fixture("nero/single", "single")
        episode = self.get(repo_id="nero/single").json
        self.assertEqual(len(episode["action"][0]), 8)
        self.assertEqual(episode["videos"][0]["start"], 0.3)
        self.assertEqual(episode["videos"][0]["end"], 0.6)
        response = self.client.get(
            "/api/review/video",
            query_string={
                "repo_id": "nero/single",
                "episode_index": 1,
                "camera": "right",
            },
            headers={"Range": "bytes=4-7"},
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"4567")
        self.assertEqual(response.headers["Content-Range"], "bytes 4-7/16")
        response.close()

    def test_report_events_locate_failed_sample_to_adjacent_frame(self) -> None:
        path = timing.report_path(self.root / "nero/review", 1)
        report = json.loads(path.read_text())
        report["samples"][1]["cameras"]["right"]["repeated"] = True
        report["samples"].append(
            {
                "frame_index": None,
                "started_seconds": 0.5,
                "completed_seconds": 0.55,
                "error": "反馈无效",
                "cameras": {},
            }
        )
        report["summary"]["failed_samples"] = 1
        report["summary"]["warnings"] = ["存在失败采样"]
        self.write_json(path, report)
        episode = self.get().json
        self.assertEqual(episode["quality"]["state"], "review")
        self.assertEqual([e["frame"] for e in episode["events"]], [1, 2])
        self.assertEqual(episode["events"][1]["time"], 0.2)
        self.assertEqual(episode["events"][1]["host_time"], 0.5)
        self.assertIn("未写入", episode["events"][1]["message"])

    def test_control_gap_still_needs_review_when_data_sampling_is_regular(self) -> None:
        path = timing.report_path(self.root / "nero/review", 1)
        report = json.loads(path.read_text())
        report["control"] = {"summary": {"target_hz": 50, "warnings": ["错过 4 个计划控制时刻"]}}
        self.write_json(path, report)
        quality = self.get().json["quality"]
        self.assertEqual(quality["state"], "review")
        self.assertIn("错过 4 个计划控制时刻", quality["warnings"])
        self.assertEqual(quality["control_timing"]["target_hz"], 50)

    def test_missing_legacy_and_corrupt_reports_keep_curves_available(self) -> None:
        path = timing.report_path(self.root / "nero/review", 1)
        original = json.loads(path.read_text())
        original["schema_version"] = 1
        self.write_json(path, original)
        self.assertEqual(self.get().json["quality"]["state"], "ok")
        path.unlink()
        self.assertEqual(self.get().json["quality"]["state"], "unknown")
        invalid = ["{broken", '{"summary":NaN}']
        for mutation in ("completed", "cameras", "pairs", "frame_index", "summary"):
            report = json.loads(json.dumps(original))
            if mutation == "completed":
                del report["samples"][0]["completed_seconds"]
            elif mutation == "cameras":
                report["samples"][0]["cameras"] = []
            elif mutation == "pairs":
                report["pairs"] = "bad"
            elif mutation == "frame_index":
                report["samples"][0]["frame_index"] = 0.5
            else:
                report["summary"]["cameras"] = {"right": {}}
            invalid.append(json.dumps(report))
        for content in invalid:
            with self.subTest(content=content[:80]):
                path.write_text(content)
                response = self.get()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["quality"]["state"], "review")
                self.assertEqual(response.json["events"], [])
                self.assertEqual(len(response.json["action"]), 3)

    def test_missing_video_keeps_curves_and_reports_error(self) -> None:
        (self.root / "nero/review/videos/observation.images.right/chunk-000/file-000.mp4").unlink()
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertIn("缺失", response.json["videos"][0]["error"])
        self.assertEqual(self.get("video", camera="right").status_code, 404)

    def test_active_writer_is_not_finalized_or_read(self) -> None:
        dataset = Mock()
        self.runtime.dataset = dataset
        self.runtime.state.update(repo_id="nero/review", connected=True)
        with patch.object(LocalReview, "episode", side_effect=AssertionError("不得读取活动数据")):
            for route in ("episode", "episodes", "video"):
                response = self.get(route, camera="right")
                self.assertEqual(response.status_code, 409)
                self.assertIn("无需失能", response.json["msg"])
        self.assertEqual(dataset.mock_calls, [])
        self.assertTrue(self.get("datasets").json["datasets"][0]["blocked"])
        self.fixture("nero/other", "single")
        self.assertEqual(self.get(repo_id="nero/other").status_code, 200)
        self.runtime.dataset = None
        self.runtime.state["connected"] = False
        self.assertEqual(self.get().status_code, 200)

    def test_paths_parameters_and_unknown_camera_are_rejected(self) -> None:
        for repo in ("../review", "nero/..", "/tmp/data", "nero/review/extra", "nero\\review"):
            self.assertEqual(self.get(repo_id=repo).status_code, 400)
        for value in (-1, "a", "1.1"):
            self.assertEqual(self.get(episode_index=value).status_code, 400)
        self.assertEqual(self.get(episode_index=5).status_code, 404)
        self.assertEqual(self.get("episodes", limit=1000).status_code, 400)
        self.assertEqual(self.get("video", camera="../../info").status_code, 404)

    def test_symlink_and_metadata_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            root = self.root / "nero/review"
            path = root / "videos/observation.images.right/chunk-000/file-000.mp4"
            secret = Path(outside) / "secret.mp4"
            secret.write_bytes(b"private")
            path.unlink()
            path.symlink_to(secret)
            self.assertEqual(self.get("video", camera="right").status_code, 400)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["data_path"] = str(secret)
            self.write_json(info_path, info)
            self.assertEqual(self.get().status_code, 400)
            (self.root / "nero/escape").symlink_to(outside)
            self.assertEqual(self.get(repo_id="nero/escape").status_code, 400)

    def test_corrupt_or_unfinished_parquet_returns_json_error(self) -> None:
        path = self.root / "nero/review/meta/episodes/chunk-000/file-000.parquet"
        path.write_bytes(b"PAR1 unfinished")
        response = self.get()
        self.assertEqual(response.status_code, 422)
        self.assertIn("读取失败", response.json["msg"])
        path.unlink()
        self.assertEqual(self.get().status_code, 409)

    def test_bad_frame_indices_or_nan_refuse_misleading_alignment(self) -> None:
        path = self.root / "nero/review/data/chunk-000/file-000.parquet"
        original = pq.read_table(path).to_pylist()
        for invalid in ("duplicate", "nan", "time"):
            rows = json.loads(json.dumps(original))
            if invalid == "duplicate":
                rows[-1]["frame_index"] = 1
            elif invalid == "nan":
                rows[-1]["action"][0] = float("nan")
            else:
                rows[-1]["timestamp"] = 0.4
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertEqual(self.get().status_code, 400)

    def test_all_review_routes_leave_files_and_runtime_untouched(self) -> None:
        def hashes() -> dict:
            return {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*")
                if p.is_file()
            }

        before, state = hashes(), dict(self.runtime.state)
        for route in ("/", "/review", "/static/review.js"):
            self.assertEqual(self.client.get(route).status_code, 200)
        for route in ("datasets", "episodes", "episode", "video"):
            response = self.get(route, camera="right")
            self.assertEqual(response.status_code, 200)
            response.close()
            self.assertEqual(self.client.post(f"/api/review/{route}").status_code, 405)
        self.assertEqual(hashes(), before)
        self.assertEqual(self.runtime.state, state)
        self.assertIsNone(self.runtime.dataset)

    def test_real_lerobot_two_episode_video_roundtrip(self) -> None:
        root = self.root / "nero/real"
        dataset = LeRobotDataset.create(
            "nero/real",
            root=root,
            fps=10,
            robot_type="nero",
            features=recorder.make_features({"right": {"type": "rgb", "path": "/fake/right"}}),
            image_writer_threads=1,
            concatenate_videos=False,
        )
        try:
            for episode in range(2):
                for frame in range(3):
                    dataset.add_frame(
                        {
                            "action": np.full(8, episode + frame / 10, dtype=np.float32),
                            "observation.state": np.full(8, episode - frame / 10, dtype=np.float32),
                            "observation.images.right": np.full((480, 640, 3), episode * 100, dtype=np.uint8),
                            "task": "回放测试",
                        }
                    )
                dataset.save_episode()
        finally:
            dataset.finalize()
        for index in range(2):
            response = self.get(repo_id="nero/real", episode_index=index)
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.json["videos"][0]["start"], 0)
            self.assertEqual(response.json["videos"][0]["error"], None)
            np.testing.assert_allclose(response.json["action"][0], [index] * 8)
            path, _, _ = self.reader.video("nero/real", index, "right")
            self.assertIn(b"ftyp", path.read_bytes()[:16])
        self.assertEqual(len(list((root / "videos").rglob("*.mp4"))), 2)


if __name__ == "__main__":
    unittest.main()
