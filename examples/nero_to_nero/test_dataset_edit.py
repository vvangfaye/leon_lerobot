"""真实 LeRobot/Parquet/视频的离线删除测试。每次删除前独立复制备份，绝不使用用户数据集。"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from nero_platform import dataset_edit, dataset_transaction, recorder
from nero_platform.app import create_app
from nero_platform.review import LocalReview, ReviewError
from nero_platform.runtime import JOINT_KEYS, Runtime

from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import load_stats


class DatasetEditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="nero-edit-tests-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "datasets/review/test"
        self.runtime = Runtime(dataset_root=self.base / "datasets")
        self.runtime.state["repo_id"] = "review/test"
        self.client = create_app(self.runtime).test_client()
        self.backups = 0

    def make(
        self,
        episodes: int = 3,
        *,
        shards: bool = True,
        videos: bool = False,
        shared: bool = False,
        depth: bool = False,
    ) -> None:
        features = {
            key: {"dtype": "float32", "shape": (8,), "names": JOINT_KEYS}
            for key in ("action", "observation.state")
        }
        if videos:
            features["observation.images.left"] = {
                "dtype": "video",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            }
        if depth:
            features["observation.images.left_depth"] = {
                "dtype": "video",
                "shape": (64, 64, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
        options = {
            "root": self.root,
            "concatenate_videos": shared,
            "rgb_encoder": RGBEncoderConfig(vcodec="h264", preset="ultrafast"),
            "depth_encoder": DepthEncoderConfig(depth_min=0.0, depth_max=10.0),
            "encoder_threads": 1,
        }
        for ep in range(episodes):
            if ep == 0:
                dataset = LeRobotDataset.create("review/test", 30, features=features, **options)
            elif shards:
                dataset = LeRobotDataset.resume("review/test", **options)
            for _ in range(3):
                frame = {
                    key: np.full(8, 10 + ep, dtype=np.float32) for key in ("action", "observation.state")
                }
                frame["task"] = "test"
                if videos:
                    frame["observation.images.left"] = np.full((64, 64, 3), 40 * ep, dtype=np.uint8)
                if depth:
                    frame["observation.images.left_depth"] = np.full((64, 64, 1), 1.0 + ep, dtype=np.float32)
                dataset.add_frame(frame)
            dataset.save_episode()
            if shards:
                dataset.finalize()
            report = self.root / "meta/nero_timing" / f"episode_{ep:06d}.json"
            report.parent.mkdir(exist_ok=True)
            report.write_text(json.dumps({"episode_index": ep}))
        if not shards:
            dataset.finalize()
        self.runtime.state["episode_count"] = episodes

    def backup(self) -> Path:
        self.backups += 1
        dest = self.base / f"backup-{self.backups}"
        shutil.copytree(self.root, dest)  # 独立副本，不能使用链接冒充备份。
        return dest

    @staticmethod
    def contents(root: Path) -> dict[str, bytes]:
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def target(self) -> dict:
        result = self.client.post("/api/record/delete-last/preview")
        self.assertEqual(result.status_code, 200, result.json)
        return result.json["target"]

    def delete(self, target: dict | None = None):
        target = target if target is not None else self.target()
        self.backup()
        return self.client.post("/api/record/delete-last", json=target)

    def test_shards_preserved_review_and_resume_work(self) -> None:
        self.make()
        untouched = {
            str(p.relative_to(self.root)): p.read_bytes()
            for folder in ("data", "meta/episodes")
            for p in (self.root / folder).glob("chunk-*/file-00[01].parquet")
        }
        self.assertEqual(self.delete().status_code, 200)
        for path, content in untouched.items():
            self.assertEqual((self.root / path).read_bytes(), content)
        LocalReview(self.root.parents[1]).episode("review/test", 1)
        dataset = LeRobotDataset.resume("review/test", root=self.root)
        dataset.add_frame(
            {
                "action": np.ones(8, dtype=np.float32),
                "observation.state": np.ones(8, dtype=np.float32),
                "task": "test",
            }
        )
        dataset.save_episode()
        dataset.finalize()
        self.assertEqual(dataset.meta.total_episodes, 3)
        self.assertEqual(dataset.meta.total_frames, 7)
        LocalReview(self.root.parents[1]).episode("review/test", 2)

    def test_same_parquet_keeps_prior_rows(self) -> None:
        self.make(shards=False)
        self.assertEqual(self.delete().status_code, 200)
        data = pq.read_table(self.root / "data/chunk-000/file-000.parquet")
        self.assertEqual(data["episode_index"].to_pylist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(data["index"].to_pylist(), list(range(6)))

    def test_shared_video_rejected_before_any_change(self) -> None:
        self.make(2, shards=False, videos=True, shared=True)
        backup = self.backup()
        response = self.client.post("/api/record/delete-last/preview")
        self.assertEqual(response.status_code, 409)
        self.assertIn("共用视频", response.json["msg"])
        response = self.client.post("/api/record/delete-last", json={"repo_id": "review/test"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_video_and_depth_stats_and_files_match_remaining_episode(self) -> None:
        self.make(2, videos=True, depth=True)
        row = pq.read_table(self.root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
        kept_videos = {p: p.read_bytes() for p in (self.root / "videos").rglob("file-000.mp4")}
        self.assertEqual(self.delete().status_code, 200)
        self.assertEqual(len(list((self.root / "videos").rglob("*.mp4"))), 2)
        self.assertFalse((self.root / "meta/nero_timing/episode_000001.json").exists())
        for path, content in kept_videos.items():
            self.assertEqual(path.read_bytes(), content)
        for feature, stats in load_stats(self.root).items():
            for name, value in stats.items():
                np.testing.assert_allclose(value, np.atleast_1d(row[f"stats/{feature}/{name}"]))

    def test_delete_to_zero_resumes_without_old_stats(self) -> None:
        self.make(1)
        self.assertEqual(self.delete().status_code, 200)
        self.assertIsNone(load_stats(self.root))
        dataset = LeRobotDataset.resume("review/test", root=self.root)
        dataset.add_frame(
            {
                "action": np.full(8, 99, dtype=np.float32),
                "observation.state": np.full(8, 99, dtype=np.float32),
                "task": "test",
            }
        )
        dataset.save_episode()
        dataset.finalize()
        self.assertEqual(dataset.meta.total_frames, 1)
        np.testing.assert_allclose(dataset.meta.stats["action"]["mean"], np.full(8, 99))
        np.testing.assert_array_equal(dataset.meta.stats["action"]["count"], [1])

    def test_failed_stats_leave_original_untouched(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        with patch.object(dataset_edit, "_remaining_stats", side_effect=OSError("统计测试失败")):
            self.assertEqual(self.delete(target).status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))
        self.assertEqual(self.delete(target).status_code, 200)
        self.assertEqual(recorder.disk_episode_count(self.runtime), 2)

    def test_mid_commit_failure_rolls_back_and_retry_deletes_only_target(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        original = dataset_transaction.os.replace
        failed = False

        def fail_once(src, dst):
            nonlocal failed
            if not failed and Path(dst) == self.root / "meta/info.json":
                failed = True
                raise OSError("模拟磁盘错误")
            return original(src, dst)

        with patch.object(dataset_transaction.os, "replace", side_effect=fail_once):
            response = self.delete(target)
        self.assertEqual(response.status_code, 409)
        self.assertIn("已恢复原数据", response.json["msg"])
        self.assertEqual(self.contents(self.root), self.contents(backup))
        self.assertEqual(self.delete(target).status_code, 200)
        self.assertEqual(recorder.disk_episode_count(self.runtime), 2)
        self.assertEqual(self.delete(target).status_code, 409)
        self.assertEqual(recorder.disk_episode_count(self.runtime), 2)

    def test_restart_recovers_interrupted_commit_before_confirmation(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        original = dataset_transaction.os.replace

        def interrupt(src, dst):
            if Path(dst) == self.root / "meta/info.json":
                raise KeyboardInterrupt("模拟进程中断")
            return original(src, dst)

        with (
            patch.object(dataset_transaction.os, "replace", side_effect=interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.delete(target)
        self.assertTrue(dataset_transaction.pending(self.root))
        with self.assertRaises(ReviewError):
            LocalReview(self.root.parents[1]).episode("review/test", 0)
        self.runtime = Runtime(dataset_root=self.base / "datasets")
        self.runtime.state["repo_id"] = "review/test"
        self.client = create_app(self.runtime).test_client()
        self.assertEqual(self.target(), target)
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_failure_after_video_removal_restores_video_bytes(self) -> None:
        self.make(2, videos=True)
        target = self.target()
        backup = self.backup()
        original = Path.unlink

        def fail_report(path, *args, **kwargs):
            if path == self.root / "meta/nero_timing/episode_000001.json":
                raise OSError("模拟报告删除失败")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_report):
            self.assertEqual(self.delete(target).status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_committed_marker_error_does_not_report_false_failure(self) -> None:
        self.make()
        target = self.target()
        original = dataset_transaction._mark

        def fail_after_mark(work, manifest):
            original(work, manifest)
            if manifest["state"] == "committed":
                raise OSError("模拟提交标记后目录同步失败")

        with patch.object(dataset_transaction, "_mark", side_effect=fail_after_mark):
            self.assertEqual(self.delete(target).status_code, 200)
        self.assertEqual(recorder.disk_episode_count(self.runtime), 2)
        self.assertEqual(self.delete(target).status_code, 409)

    def test_failed_rollback_preserves_journal_then_resume_recovers(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        original = dataset_transaction.os.replace

        def fail_writes(src, dst):
            if Path(dst) == self.root / "meta/info.json" or Path(src).name == "restore.tmp":
                raise OSError("磁盘暂时不可写")
            return original(src, dst)

        with patch.object(dataset_transaction.os, "replace", side_effect=fail_writes):
            response = self.delete(target)
            self.assertIn("恢复失败", response.json["msg"])
            self.assertTrue(dataset_transaction.pending(self.root))
            self.assertEqual(self.client.post("/api/record/delete-last/preview").status_code, 409)
        info = json.loads((backup / "meta/info.json").read_text())
        recorder.validate_resume(
            self.root, 30, {key: info["features"][key] for key in ("action", "observation.state")}
        )
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_missing_episode_stats_rejected_without_mutation(self) -> None:
        self.make()
        path = self.root / "meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        pq.write_table(table.drop(["stats/action/mean"]), path)
        backup = self.backup()
        self.assertEqual(self.delete().status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_missing_confirmation_and_dataset_switch_rejected(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        self.assertEqual(self.client.post("/api/record/delete-last").status_code, 409)
        self.runtime.state["repo_id"] = "review/other"
        self.assertEqual(self.delete(target).status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))

    def test_busy_or_open_writer_rejects_preview_and_delete(self) -> None:
        self.make()
        target = self.target()
        backup = self.backup()
        for state in ("connected", "recording", "ending", "homing", "deleting", "shutting_down"):
            with self.subTest(state=state):
                self.runtime.state[state] = True
                self.assertEqual(self.client.post("/api/record/delete-last/preview").status_code, 409)
                self.assertEqual(self.client.post("/api/record/delete-last", json=target).status_code, 409)
                self.runtime.state[state] = False
        self.runtime.dataset = object()
        self.assertEqual(self.client.post("/api/record/delete-last/preview").status_code, 409)
        self.assertEqual(self.contents(self.root), self.contents(backup))


if __name__ == "__main__":
    unittest.main()
