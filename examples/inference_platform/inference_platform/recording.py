"""Record with the same LeRobotDataset writer used by the collection platform."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "inference"


def output_path(settings: dict) -> tuple[str, Path]:
    name = settings.get("datasetName", "").strip() or "nero/inference"
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", name):
        raise ValueError("数据集名称请填写 owner/name，只使用字母、数字、下划线或连字符")
    root = Path(settings.get("outputDir", "").strip() or DEFAULT_ROOT).expanduser().resolve()
    return name, root.joinpath(name, datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])


class Recording:
    def __init__(self, settings: dict, contract: object, capture: object, command: object) -> None:
        from lerobot.datasets import LeRobotDataset

        self.repo, self.path = output_path(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.contract, self.capture, self.command = contract, capture, command
        first = capture()
        features = {
            "observation.state": {"dtype": "float32", "shape": (contract.dim,), "names": contract.names},
            "action": {"dtype": "float32", "shape": (contract.dim,), "names": contract.names},
            "action.commanded": {"dtype": "float32", "shape": (1,), "names": ["command_sent"]},
            "observation.elapsed": {"dtype": "float32", "shape": (1,), "names": ["seconds"]},
        }
        for key, image in first["images"].items():
            features[key] = {"dtype": "video", "shape": image.shape, "names": ["height", "width", "channels"]}
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo,
            root=self.path,
            fps=30,
            features=features,
            robot_type=f"{contract.config['robot']}_{contract.config['arms']}_{contract.config['space']}",
            use_videos=True,
            image_writer_threads=4,
            concatenate_videos=False,
        )
        self.task = settings["task"]
        self.stop = threading.Event()
        self.error = None
        self.frames = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name="inference-recorder")
        self.manifest = {
            "checkpoint": str(contract.path),
            "config": contract.config,
            "task": self.task,
            "fps": 30,
            "status": "recording",
            "chunks": [],
            "action": "实际发送的关节/EEF目标；action.commanded=0 表示尚未发送或暂停保持",
        }
        self._manifest()
        self.thread.start()

    def _manifest(self) -> None:
        tmp = self.path / "inference.json.tmp"
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2))
        tmp.replace(self.path / "inference.json")

    def chunk(self, actions: np.ndarray, elapsed_ms: float) -> None:
        self.manifest["chunks"].append(
            {
                "at": time.monotonic() - self.started,
                "inference_ms": elapsed_ms,
                "predicted_actions": actions.tolist(),
            }
        )
        self._manifest()

    def _run(self) -> None:
        due = time.monotonic()
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                if now - due > 0.5:
                    raise ValueError("录制写入超过 0.5 秒未跟上，已停止执行")
                obs = self.capture()
                action, commanded = self.command()
                frame = {
                    "observation.state": obs["state"].copy(),
                    "action": obs["state"].copy() if action is None else action.copy(),
                    "action.commanded": np.asarray([float(commanded)], dtype=np.float32),
                    "observation.elapsed": np.asarray([now - self.started], dtype=np.float32),
                    "task": self.task,
                    **{key: image.copy() for key, image in obs["images"].items()},
                }
                self.dataset.add_frame(frame)
                self.frames += 1
                due += 1 / 30
                self.stop.wait(max(0, due - time.monotonic()))
        except Exception as exc:
            self.error = str(exc)

    def finish(self, status: str, message: str) -> None:
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise ValueError(f"录制线程未退出，原始数据保留在 {self.path}")
        errors = []
        try:
            if self.frames:
                self.dataset.save_episode()
        except Exception as exc:
            errors.append(str(exc))
        finally:
            try:
                self.dataset.finalize()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            self.manifest.update(status="save_error", message="；".join(errors), frames=self.frames)
            self._manifest()
            raise ValueError("；".join(errors))
        self.manifest.update(status=status, message=message, frames=self.frames, recording_error=self.error)
        self._manifest()
