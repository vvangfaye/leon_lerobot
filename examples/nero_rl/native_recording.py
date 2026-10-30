"""Image caching and RL transition assembly outside the VLA control loop."""

import threading
import time
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Any

import cv2
import numpy as np

from .config import TaskConfig
from .environment import NeroEnv
from .hardware import Snapshot


class CameraCache:
    def __init__(self, backend: Any):
        self.backend = backend
        self.latest: tuple[dict, dict] = ({}, {})
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="nero-rl-images", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                frames, stamps = {}, {}
                for name, camera in self.backend.cameras.items():
                    frame, stamp, _ = camera.read_latest_with_timestamp(max_age_ms=500)
                    frames[name], stamps[name] = frame, stamp
                self.latest = frames, stamps
                self.backend.last_frames = frames
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(1 / 60)

    def ready(self, names: dict) -> bool:
        frames, stamps = self.latest
        return set(frames) == set(names) and all(
            0 <= time.perf_counter() - stamp <= 0.5 for stamp in stamps.values()
        )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1)


@dataclass
class NativeTick:
    timestamp: float
    joints: np.ndarray
    width: float
    previous_command: np.ndarray  # Actual SDK target before this command: q7 + width.
    command: np.ndarray
    requested: np.ndarray
    source: str
    images: dict[str, np.ndarray]
    camera_stamps: dict[str, float]
    info: dict = field(default_factory=dict)


class NativeRecorder:
    """One transition per sent command, joined to the next feedback snapshot.

    The control thread submits immutable numeric/image references without
    waiting. A full queue invalidates recording, never clips a human command.
    """

    def __init__(self, cfg: TaskConfig, backend: Any, *, mock: bool = False, capacity: int = 256):
        self.cfg = cfg
        self.env = NeroEnv(cfg, backend, mock=mock)
        self.jobs: Queue = Queue(maxsize=capacity)
        self.rows: list[dict] = []
        self.previous: tuple[NativeTick, dict] | None = None
        self.error: str | None = None
        self.latest_obs: dict | None = None
        self.cached_images: dict[str, tuple[float, np.ndarray]] = {}
        self.thread = threading.Thread(target=self._run, name="nero-rl-recorder", daemon=True)
        self.thread.start()

    def _observation(self, tick: NativeTick) -> dict:
        if set(tick.images) != set(self.cfg.cameras):
            raise ValueError("Incomplete camera snapshot; recording interrupted")
        self.env.command_joints = tick.previous_command[:7].copy()
        self.env.command_width = float(tick.previous_command[7])
        pose = self.env.backend.forward(tick.joints)
        obs = self.env.observation(Snapshot(tick.joints, tick.width, pose, {}))
        for name, frame in tick.images.items():
            stamp = tick.camera_stamps[name]
            if not -0.001 <= tick.timestamp - stamp <= 0.5:
                raise ValueError(f"Stale image {name}; recording interrupted")
            cached = self.cached_images.get(name)
            if cached is None or cached[0] != stamp:
                small = cv2.resize(
                    frame, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA
                )
                cached = stamp, small.transpose(2, 0, 1).copy()
                self.cached_images[name] = cached
            obs[f"observation.images.{name}"] = cached[1]
        return obs

    def _consume(self, tick: NativeTick) -> None:
        obs = self._observation(tick)
        if self.previous is not None:
            previous, before = self.previous
            action = self.cfg.encode_joint_target(previous.command[:7], float(previous.command[7]))
            requested = self.cfg.encode_joint_target(previous.requested[:7], float(previous.requested[7]))
            if np.any(np.abs(action) > 1.000001):
                raise ValueError("SDK target cannot be represented by the absolute action contract")
            self.rows.append(
                {
                    "state": before,
                    "next_state": obs,
                    "action": action,
                    "requested_action": requested,
                    "reward": 0.0,
                    "done": False,
                    "truncated": False,
                    "intervention": previous.source in ("right_master_vla", "right_master_relative"),
                    "elapsed_s": tick.timestamp - previous.timestamp,
                    "safety_rejection": previous.info.get("safety_rejection") is not None,
                    "action_source": previous.source,
                    "control_timestamp": previous.timestamp,
                    "camera_timestamps": [previous.camera_stamps[name] for name in self.cfg.cameras],
                }
            )
        self.previous, self.latest_obs = (tick, obs), obs

    def _run(self) -> None:
        while True:
            kind, payload, reply = self.jobs.get()
            try:
                if kind == "close":
                    return
                if kind == "start":
                    self.rows, self.previous, self.error = [], None, None
                elif kind == "tick" and self.error is None:
                    self._consume(payload)
                elif kind in ("segment", "finish"):
                    tick, outcome, truncated = payload
                    if tick is not None and self.error is None:
                        self._consume(tick)
                    self.previous = None
                    if kind == "finish" and self.rows:
                        failed = self.error is not None
                        self.rows[-1].update(
                            reward=float(outcome == "success" and not failed),
                            done=outcome is not None and not failed,
                            truncated=bool(truncated or failed),
                        )
            except Exception as exc:
                self.error, self.previous = str(exc), None
                if self.rows and kind == "finish":
                    self.rows[-1]["truncated"] = True
            finally:
                if reply is not None:
                    reply.set()
                self.jobs.task_done()

    def _sync(self, kind: str, payload: Any = None) -> None:
        done = threading.Event()
        self.jobs.put((kind, payload, done), timeout=5)
        if not done.wait(5):
            raise TimeoutError("RL recorder did not flush")

    def start(self) -> None:
        self._sync("start")

    def submit(self, tick: NativeTick) -> bool:
        if self.error is not None:
            return False
        try:
            self.jobs.put_nowait(("tick", tick, None))
            return True
        except Full:
            self.error = "RL recording queue overflow; human following remains active"
            return False

    def segment(self, boundary: NativeTick | None) -> None:
        self._sync("segment", (boundary, None, False))

    def finish(self, boundary: NativeTick | None, outcome: str | None, *, truncated: bool) -> list[dict]:
        self._sync("finish", (boundary, outcome, truncated))
        return self.rows

    def close(self) -> None:
        self._sync("close")
        self.thread.join(timeout=1)
