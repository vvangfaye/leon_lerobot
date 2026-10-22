"""Device lifecycle and explicit identification jog for step 2."""

from __future__ import annotations

import copy
import fcntl
import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any
from uuid import uuid4

from .discovery import attachment_present
from .hardware import discover, make_reader
from .jog import JogJob, JogStopped

ARM_ROLES = ("left_leader", "right_leader", "left_follower", "right_follower")

# 推理平台角色 → 采集平台角色标签（限位文件按标签命名）。
_ROLE_LABELS = {
    "left_leader": "主左", "right_leader": "主右",
    "left_follower": "从左", "right_follower": "从右",
}


def _saved_limit_channel(label: str) -> str | None:
    try:
        from .hardware import camera_module

        saved = camera_module("joint_limits").load_limits(label)
        return saved.get("channel") if saved else None
    except Exception:
        return None


class DeviceWorker:
    """One owner per device; HTTP clients only consume cached samples."""

    def __init__(self, info: dict, robot: str, factory: Callable, present: Callable) -> None:
        self.info, self.robot, self.factory, self.present = info, robot, factory, present
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.phase, self.message = "connecting", "正在建立连接…"
        self.sample: dict = {}
        self.jpeg: bytes | None = None
        self.rgb = None
        self.captured = 0.0
        self.commands = queue.Queue()
        self.updated = 0.0
        self.closed = False
        self.faults = 0
        self.pending_jog: JogJob | None = None
        self.reader: Any = None
        self.pending_measure = False
        self.measure_stop = threading.Event()
        self.measure_report: dict | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"preview-{info['id']}")

    def _run(self) -> None:
        reader = None
        try:
            reader = self.factory(self.info, self.robot)
            reader.connect()
            self.reader = reader
            while True:
                with self.lock:
                    job, self.pending_jog = self.pending_jog, None
                    measure = self.pending_measure
                    self.pending_measure = False
                if job is not None:
                    self._jog(reader, job)
                if measure:
                    self._measure(reader)
                if self.stop.is_set():
                    break
                try:
                    future, operation, value, check = self.commands.get_nowait()
                except queue.Empty:
                    future = None
                if future is not None and future.set_running_or_notify_cancel():
                    with self.lock:
                        self.phase = "controlling"
                    try:
                        check()
                        result = reader.control(operation, value, check)
                        # Publish fresh feedback before completing the RPC.
                        sample, jpeg = reader.sample()
                        with self.lock:
                            self.sample, self.jpeg, self.updated = sample, jpeg, time.monotonic()
                            self.phase = "ready" if sample["ready"] else "waiting"
                            self.message = sample["message"]
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                    continue
                if not self.present(self.info):
                    raise ValueError("设备已拔插或 CAN 已关闭，请断开后重新检测")
                try:
                    sample, jpeg = reader.sample()
                    with self.lock:
                        if self.phase == "ready" and (
                            not sample["ready"] or time.monotonic() - self.updated >= 0.5
                        ):
                            self.faults += 1
                        self.rgb = getattr(reader, "rgb", None)
                        self.captured = getattr(reader, "captured", 0.0)
                        self.sample, self.jpeg = sample, jpeg
                        self.phase = "ready" if sample["ready"] else "waiting"
                        self.message, self.updated = sample["message"], time.monotonic()
                except Exception as exc:
                    with self.lock:
                        if self.phase == "ready":
                            self.faults += 1
                        self.phase, self.message, self.sample, self.jpeg = "error", str(exc), {}, None
                self.stop.wait(0.005 if self.info["kind"] == "arm" else 0.025)
        except Exception as exc:
            with self.lock:
                self.phase, self.message = "error", str(exc)
                self.sample, self.jpeg = {}, None
        finally:
            while not self.commands.empty():
                future, *_ = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(ValueError("设备已断开"))
            with self.lock:
                if self.pending_jog is not None:
                    self.pending_jog.finish("error", "设备连接中断，未执行微动")
                    self.pending_jog = None
            try:
                if reader is not None:
                    reader.close()
                with self.lock:
                    self.closed = True
                    self.jpeg, self.sample = None, {}
                    if self.stop.is_set():
                        self.phase, self.message = "closed", "已断开"
            except Exception as exc:
                with self.lock:
                    self.phase, self.message = "release_failed", f"释放失败，请退出平台后重试：{exc}"
                    self.jpeg, self.sample = None, {}

    def _jog(self, reader: Any, job: JogJob) -> None:
        def check() -> None:
            job.check()
            if self.stop.is_set():
                raise JogStopped("设备正在断开")
            if not self.present(self.info):
                raise ValueError("CAN 设备已断开或发生拔插")

        with self.lock:
            self.phase, self.message = "jogging", "正在微动测试"
            self.sample, self.jpeg = {}, None
        try:
            job.wait_for_browser()
            check()
            result = reader.jog(check, job.progress)
            job.finish(*result)
        except JogStopped as exc:
            job.finish("stopped", f"{exc}；未开始运动")
        except Exception as exc:
            # Real adapter catches motion exceptions and attempts an emergency
            # stop before returning. An unexpected adapter failure stays locked.
            job.finish("error", f"微动适配异常：{exc}；请检查设备后重启平台", recovery=True)

    def _measure(self, reader: Any) -> None:
        with self.lock:
            self.phase, self.message = "measuring", "拖动测量中"
        try:
            report = reader.measure_limits(self.measure_stop, lambda _message: None)
            label = self.info.get("label")
            if not label:
                self.measure_report = {"error": "请先分配用途（左/右主从臂），再测量"}
                return
            from .hardware import camera_module

            path = camera_module("joint_limits").save_limits(label, report)
            self.measure_report = {
                "ok": True, "label": label, "path": str(path),
                "samples": report.get("valid_samples", 0),
            }
        except Exception as exc:
            self.measure_report = {"error": str(exc)}
        finally:
            with self.lock:
                self.phase = "ready"

    def queue_measure(self) -> None:
        with self.lock:
            if self.stop.is_set() or not self.thread.is_alive() or self.phase != "ready" \
                    or time.monotonic() - self.updated >= 0.5 or self.pending_measure:
                raise ValueError("机械臂尚无有效普通关节反馈，请先连接")
            self.pending_measure = True
            self.measure_stop = threading.Event()
            self.measure_report = None

    def stop_measure(self) -> None:
        self.measure_stop.set()

    def command(self, operation: str, value: Any, check: Callable) -> Future:
        future = Future()
        if self.stop.is_set() or not self.thread.is_alive():
            future.set_exception(ValueError("设备已断开"))
        else:
            self.commands.put((future, operation, value, check))
        return future

    def capture(self) -> tuple:
        with self.lock:
            if self.stop.is_set() or self.phase not in ("ready", "controlling") or time.monotonic() - self.updated >= .5:
                raise ValueError(f"{self.info['id']} 无新反馈")
            if self.info["kind"] == "camera" and (self.rgb is None or time.monotonic() - self.captured >= .5):
                raise ValueError(f"{self.info['id']} 图像过期")
            return copy.deepcopy(self.sample), self.rgb, self.captured

    def queue_jog(self, job: JogJob) -> None:
        with self.lock:
            if self.stop.is_set() or not self.thread.is_alive() or self.phase != "ready" \
                    or time.monotonic() - self.updated >= 0.5 or self.pending_jog is not None:
                raise ValueError("机械臂尚无有效普通关节反馈，请先连接")
            self.pending_jog = job

    def snapshot(self) -> dict:
        with self.lock:
            phase, message = self.phase, self.message
            if self.stop.is_set() and self.thread.is_alive():
                phase, message = "closing", "正在释放设备…"
            elif self.stop.is_set() and self.closed:
                phase, message = "closed", "已断开"
            if phase == "ready" and time.monotonic() - self.updated >= 0.5:
                phase, message = "error", "反馈超过 0.5 秒未更新"
            measure = {
                "measuring": self.pending_measure or self.phase == "measuring",
                "report": self.measure_report,
            }
            return {"phase": phase, "message": message, "sample": copy.deepcopy(self.sample),
                    "faults": self.faults, "measure": measure}

    def frame(self) -> bytes | None:
        with self.lock:
            if self.stop.is_set() or self.phase != "ready" or time.monotonic() - self.updated >= 0.5:
                return None
            return self.jpeg


class DeviceManager:
    def __init__(
        self, *, scanner: Callable = discover, factory: Callable = make_reader,
        present: Callable = attachment_present, lease_path: Path | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.scanner, self.factory, self.present = scanner, factory, present
        self.lease_path = lease_path or Path(f"/tmp/nero-inference-devices-{os.getuid()}.lock")
        self.lease: Any = None
        self.devices: dict = {}
        self.workers: dict[str, DeviceWorker] = {}
        self.roles: dict[str, str] = {}
        self.detection: dict = {}
        self.config: dict | None = None
        self.revision = uuid4().hex
        self.confirmed = False
        self.confirmed_faults: dict[str, int] = {}
        self.scanning = False
        self.shutdown = False
        self.message = "请检测设备，再打开反馈和画面"
        self.scan_thread: threading.Thread | None = None
        self.jog_job: JogJob | None = None
        self.motion_job: JogJob | None = None

    def _jog_idle(self) -> None:
        if self.motion_job is not None:
            status = self.motion_job.snapshot()
            if status["active"] or status["recovery"]:
                raise ValueError("模型执行/回位中，请先停止并等待停稳" if status["active"] else "停止未确认，请现场检查后重启平台")
        if self.jog_job is not None and self.jog_job.snapshot()["active"]:
            raise ValueError("微动测试中，请先停止并等待停稳")

    def _active(self) -> bool:
        return any(worker.thread.is_alive() or not worker.closed for worker in self.workers.values())

    def _release_lease(self) -> None:
        if self.lease is not None and not self._active():
            self.lease.close()
            self.lease = None

    def _require_revision(self, revision: str) -> None:
        if self.shutdown or self.scanning or revision != self.revision:
            raise ValueError("设备列表已变化，请刷新后重试")

    @staticmethod
    def _validate_config(config: dict) -> dict:
        if not isinstance(config, dict) or any(
            config.get(key) not in values for key, values in {
                "robot": ("nero", "piper"), "arms": ("single", "dual"), "space": ("joint", "eef"),
            }.items()
        ):
            raise ValueError("请选择 NERO / Piper、单臂 / 双臂及动作空间")
        return {key: config[key] for key in ("robot", "arms", "space")}

    def configure(self, config: dict, revision: str) -> None:
        config = self._validate_config(config)
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            if config == self.config:
                return
            robot_changed = self.config is not None and config["robot"] != self.config["robot"]
            if robot_changed:
                for identity, worker in self.workers.items():
                    if self.devices[identity]["kind"] == "arm":
                        worker.stop.set()
            self.config = config
            self.confirmed, self.confirmed_faults = False, {}
            self.revision = uuid4().hex
            if self.jog_job is not None and not self.jog_job.snapshot()["recovery"]:
                self.jog_job = None
            self.message = (
                "机械臂类型已修改，正在释放原机械臂连接；请按新类型重新连接。用途分配和相机预览已保留。"
                if robot_changed else "使用方式 / 动作空间已更新，设备连接和用途分配已保留，请重新核对用途。"
            )

    def scan(self, config: dict) -> None:
        config = self._validate_config(config)
        with self.lock:
            self._jog_idle()
            if self.shutdown or self.scanning or self._active():
                raise ValueError("请等待检测完成，或先断开全部设备")
            self._release_lease()
            self.confirmed = False
            self.roles, self.devices, self.workers, self.detection = {}, {}, {}, {}
            if self.jog_job is not None and not self.jog_job.snapshot()["recovery"]:
                self.jog_job = None
            self.config = config
            self.revision, self.scanning, self.message = uuid4().hex, True, "正在检测 CAN 与相机…"
            self.scan_thread = threading.Thread(target=self._scan, daemon=True, name="device-scan")
            self.scan_thread.start()

    def _scan(self) -> None:
        try:
            devices, detection = self.scanner()
            with self.lock:
                if not self.shutdown:
                    self.devices, self.detection = devices, detection
                    self.message = "检测完成，请打开反馈/预览并分配用途" if devices else "未发现设备，请检查连接"
        except Exception as exc:
            with self.lock:
                self.message = f"检测失败：{exc}"
        finally:
            with self.lock:
                self.scanning = False

    def open(self, identity: str, revision: str) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            info = self.devices.get(identity)
            if info is None or not self.present(info):
                raise ValueError("设备不存在、已拔插或 CAN 未启动，请重新检测")
            previous = self.workers.get(identity)
            if previous is not None and (previous.thread.is_alive() or not previous.closed):
                raise ValueError("设备正在连接或尚未释放，请先断开")
            if self.lease is None:
                stream = self.lease_path.open("a")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    stream.close()
                    raise ValueError("另一推理平台正在使用设备，请先断开它的连接") from exc
                self.lease = stream
            self.confirmed = False
            self.revision = uuid4().hex
            worker = DeviceWorker(info, self.config["robot"], self.factory, self.present)
            self.workers[identity] = worker
            worker.thread.start()
            self.message = "连接中；查看反馈和画面后确认用途"

    def disconnect(self, revision: str, identity: str | None = None) -> None:
        with self.lock:
            self._require_revision(revision)
            if self.motion_job is not None and self.motion_job.snapshot()["active"]:
                raise ValueError("请先停止模型执行/回位并等待停稳，再断开设备")
            if identity is not None and identity not in self.devices:
                raise ValueError("设备不存在")
            self.confirmed = False
            self.revision = uuid4().hex
            for key, worker in self.workers.items():
                if identity is None or key == identity:
                    if self.jog_job is not None and self.jog_job.identity == key:
                        self.jog_job.cancel.set()
                    worker.stop.set()
            self.message = "已请求断开；正在微动的机械臂先停止，再释放通信"

    def assign(self, identity: str, role: str, revision: str) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            info = self.devices.get(identity)
            if info is None:
                raise ValueError("设备不存在")
            allowed = ARM_ROLES if info["kind"] == "arm" else ("head", "left", "right")
            if role not in ("", *allowed):
                raise ValueError("用途与当前设备配置不符")
            if role and any(
                key != identity and value == role and self.devices[key]["kind"] == info["kind"]
                for key, value in self.roles.items()
            ):
                raise ValueError("这个用途已分配给另一台设备，请先取消原分配")
            self.confirmed = False
            self.revision = uuid4().hex
            info["label"] = _ROLE_LABELS.get(role)
            self.roles[identity] = role
            # 换臂提醒：该角色标签下已保存的限位若来自另一通道，提示重新测量。
            label = info["label"]
            if label:
                saved_channel = _saved_limit_channel(label)
                if saved_channel and saved_channel != identity:
                    self.message = (
                        f"警告：{label} 已保存的关节限位来自 {saved_channel}，当前是 {identity}；"
                        "若换了机械臂请重新测量，否则限位可能是错的"
                    )
                else:
                    self.message = "用途已更新，请按实物重新核对后确认"
            else:
                self.message = "用途已更新，请按实物重新核对后确认"

    def _mapping_ready(self) -> bool:
        selected = {key: role for key, role in self.roles.items() if role}
        followers = {
            role for key, role in selected.items()
            if self.devices[key]["kind"] == "arm" and role.endswith("_follower")
        }
        cameras = [key for key in selected if self.devices[key]["kind"] == "camera"]
        required = len(followers) == 1 if self.config and self.config["arms"] == "single" \
            else followers == {"left_follower", "right_follower"}
        return required and bool(cameras) and all(
            key in self.workers and self.workers[key].snapshot()["phase"] == "ready"
            and self.present(self.devices[key]) for key in selected
        )

    def confirm(self, revision: str, acknowledged: bool) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            if acknowledged is not True:
                raise ValueError("请先按实物核对设备用途并勾选确认")
            if not self._mapping_ready():
                raise ValueError("单臂需恰好一个左/右从臂，双臂需左右从臂；主臂可选，至少一台相机，所选设备须有新反馈")
            self.confirmed = True
            self.confirmed_faults = {
                key: self.workers[key].snapshot()["faults"] for key, role in self.roles.items() if role
            }
            self.revision = uuid4().hex
            self.message = "设备用途已确认，实时预览中；可加载模型或回到初始位置"

    def jog(self, identity: str, revision: str, acknowledged: bool) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            if self.jog_job is not None and self.jog_job.snapshot()["recovery"]:
                raise ValueError("上次微动异常，需要检查实物并处理故障后重启平台")
            if acknowledged is not True:
                raise ValueError("请确认微动通道及周围空间")
            if self.devices.get(identity, {}).get("kind") != "arm" or identity not in self.workers:
                raise ValueError("请先连接要测试的机械臂")
            if not self.present(self.devices[identity]):
                raise ValueError("CAN 设备已断开或发生拔插")
            job = JogJob(identity)
            self.workers[identity].queue_jog(job)
            self.jog_job = job
            self.confirmed, self.revision = False, uuid4().hex
            self.message = f"{identity} 微动测试；结束后按实物分配用途"

    def stop_jog(self) -> None:
        # Stop never waits for the SDK, and remains valid from a stale browser.
        # Rotating revision also rejects a delayed start submitted before stop.
        with self.lock:
            self.revision, self.confirmed = uuid4().hex, False
            if self.jog_job is not None:
                self.jog_job.cancel.set()

    def heartbeat_jog(self, job_id: str) -> None:
        with self.lock:
            if self.jog_job is not None and self.jog_job.id == job_id:
                self.jog_job.heartbeat()

    def disable_arm(self, identity: str, revision: str, acknowledged: bool) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            if acknowledged is not True:
                raise ValueError("请确认失能操作")
            info = self.devices.get(identity)
            if info is None or info["kind"] != "arm" or identity not in self.workers:
                raise ValueError("请先连接要失能的机械臂")
            if not self.present(info):
                raise ValueError("CAN 设备已断开或发生拔插")
            worker = self.workers[identity]
        future = worker.command("disable", None, lambda: None)
        try:
            future.result(timeout=6)
        except FutureTimeout:
            raise ValueError("失能超时，请检查设备")
        with self.lock:
            self.confirmed = False
            self.revision = uuid4().hex
            self.message = f"{identity} 已失能（关节与夹爪）；再次微动/执行会重新使能"

    def measure(self, identity: str, revision: str) -> None:
        with self.lock:
            self._require_revision(revision)
            self._jog_idle()
            info = self.devices.get(identity)
            if info is None or info["kind"] != "arm" or identity not in self.workers:
                raise ValueError("请先连接要测量的机械臂")
            if not self.present(info):
                raise ValueError("CAN 设备已断开或发生拔插")
            if not self.roles.get(identity):
                raise ValueError("请先分配用途（左/右主从臂），再测量")
            self.workers[identity].queue_measure()
            self.confirmed = False
            self.revision = uuid4().hex
            self.message = f"{identity} 拖动测量中；完成后停止并保存"

    def stop_measure(self, identity: str, revision: str) -> None:
        with self.lock:
            self._require_revision(revision)
            if identity not in self.workers:
                raise ValueError("设备未连接")
            self.workers[identity].stop_measure()
            self.revision = uuid4().hex

    def status(self) -> dict:
        with self.lock:
            self._release_lease()
            moving = self.motion_job is not None and self.motion_job.snapshot()["active"]
            if self.confirmed and not moving and (not self._mapping_ready() or any(
                self.workers[key].snapshot()["faults"] != faults for key, faults in self.confirmed_faults.items()
            )):
                self.confirmed = False
                self.revision = uuid4().hex
                self.message = "设备反馈中断或连接发生变化，请重新核对用途"
            rows = []
            for identity, info in self.devices.items():
                worker = self.workers.get(identity)
                state = worker.snapshot() if worker else {"phase": "closed", "message": "未连接", "sample": {}}
                rows.append({**info, **state, "role": self.roles.get(identity, "")})
            return copy.deepcopy({
                "schema_version": 4,
                "revision": self.revision, "config": self.config, "devices": rows, "detection": self.detection,
                "scanning": self.scanning, "active": self._active(), "confirmed": self.confirmed,
                "message": self.message,
                "motion_active": moving,
                "jog": self.jog_job.snapshot() if self.jog_job is not None else None,
            })

    def frame(self, identity: str, revision: str) -> bytes | None:
        with self.lock:
            if revision != self.revision or identity not in self.workers:
                return None
            return self.workers[identity].frame()

    def close(self) -> None:
        with self.lock:
            self.shutdown, self.confirmed = True, False
            if self.motion_job is not None:
                self.motion_job.cancel.set()
            if self.jog_job is not None:
                self.jog_job.cancel.set()
            workers = list(self.workers.values())
            for worker in workers:
                worker.stop.set()
        deadline = time.monotonic() + 5
        for worker in workers:
            worker.thread.join(timeout=max(0, deadline - time.monotonic()))
        with self.lock:
            self._release_lease()
