"""π0.5 lifecycle, closed-loop execution and cancellation, independent of HTTP."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeout
from uuid import uuid4

import numpy as np

from .contracts import Contract
from .jog import JogJob, JogStopped
from .motion import home_target, plan
from .policy import PolicyProcess
from .recording import Recording, output_path


class InferenceRuntime:
    def __init__(
        self,
        manager: object,
        policy_factory: Callable = PolicyProcess,
        recorder_factory: Callable = Recording,
    ) -> None:
        self.manager, self.policy_factory, self.recorder_factory = manager, policy_factory, recorder_factory
        self.lock = threading.RLock()
        self.revision = uuid4().hex
        self.policy = None
        self.contract = None
        self.job = None
        self.thread = None
        self.load_cancel = threading.Event()
        self.model_state, self.message = "empty", "选择 checkpoint 后即可加载 π0.5；执行前确认本次使用的设备"
        self.device, self.record_path = "", ""
        self.inference_ms, self.chunks, self.started = None, 0, 0.0
        self.elapsed = 0.0
        self.last_action, self.commanded = None, False
        self.recorder = None
        self.arms, self.cameras = {}, {}
        self.closed = False

    def status(self) -> dict:
        with self.lock:
            job = self.job.snapshot() if self.job else None
            return {
                "version": 1,
                "revision": self.revision,
                "model_state": self.model_state,
                "checkpoint": str(self.contract.path) if self.contract else "",
                "config": self.contract.config if self.contract else None,
                "device": self.device,
                "message": job["message"] if job and job["active"] else self.message,
                "job": job,
                "chunks": self.chunks,
                "inference_ms": self.inference_ms,
                "elapsed": round(time.monotonic() - self.started, 1)
                if job and job["active"]
                else self.elapsed,
                "record_path": self.record_path,
                "record_frames": self.recorder.frames if self.recorder else 0,
            }

    def _idle(self, revision: str) -> None:
        if self.closed or revision != self.revision:
            raise ValueError("运行状态已变化，请刷新后重试")
        if self.thread and self.thread.is_alive():
            raise ValueError("正在加载、执行或保存，请等待完成或停止")
        if self.job and self.job.snapshot()["recovery"]:
            raise ValueError("上次停止未确认，请现场处理后重启平台")

    def load(self, settings: dict, revision: str) -> None:
        with self.lock:
            self._idle(revision)
            contract = Contract(settings["path"], settings)
            with self.manager.lock:
                self.manager._jog_idle()
            # Loading weights neither opens hardware nor sends commands.
            # Bind only at execution, where the current roles and feedback matter.
            self.contract = contract
            self.revision, self.model_state = uuid4().hex, "loading"
            self.message = "正在本机加载 π0.5 权重与 checkpoint 预处理器…"
            self.load_cancel.clear()
            self.thread = threading.Thread(target=self._load, daemon=True, name="pi05-load")
            self.thread.start()

    def _load(self) -> None:
        def check() -> None:
            if self.load_cancel.is_set():
                raise JogStopped("模型加载已取消")

        try:
            if self.policy:
                self.policy.close()
            self.policy = self.policy_factory(str(self.contract.path))
            result = self.policy.load(check)
            check()
            with self.lock:
                self.model_state, self.device = "ready", result["device"]
                self.message = "π0.5 已加载；单集测试按时长运行，持续执行直到手动停止"
        except Exception as exc:
            if self.policy:
                self.policy.close()
                self.policy = None
            with self.lock:
                self.model_state, self.message = "error", str(exc)
        finally:
            with self.lock:
                self.revision = uuid4().hex

    def unload(self, revision: str) -> None:
        with self.lock:
            self._idle(revision)
            if self.policy:
                self.policy.close()
            self.policy, self.contract = None, None
            self.model_state, self.message = "empty", "模型已卸载，显存已释放"
            self.revision = uuid4().hex

    def start(self, kind: str, settings: dict, revision: str) -> None:
        if kind not in ("single", "continuous", "home"):
            raise ValueError("未知执行模式")
        with self.lock:
            self._idle(revision)
            with self.manager.lock:
                self.manager._jog_idle()
                if self.manager.jog_job and self.manager.jog_job.snapshot()["recovery"]:
                    raise ValueError("微动异常未恢复，请检查实物后重启")
                status = self.manager.status()
                if not status["confirmed"]:
                    raise ValueError("请先核对并确认设备用途")
                if kind != "home":
                    if self.model_state != "ready" or self.policy is None:
                        raise ValueError("请先加载 π0.5")
                    if settings.get("path") != str(self.contract.path) or any(
                        settings.get(k) != v for k, v in self.contract.config.items()
                    ):
                        raise ValueError("页面配置与已加载模型不一致，请重新加载")
                    if not isinstance(settings.get("task"), str) or not settings["task"].strip():
                        raise ValueError("请填写任务指令")
                    if kind == "single":
                        try:
                            duration = float(settings.get("duration", 60.0))
                        except (TypeError, ValueError):
                            duration = 60.0
                        if not 1.0 <= duration <= 600.0:
                            raise ValueError("运行时长需在 1-600 秒之间")
                        settings["duration"] = duration
                    self.arms, self.cameras = self.contract.bind(status)
                    if settings.get("record"):
                        output_path(settings)
                else:
                    self.arms = {
                        d["role"].removesuffix("_follower"): d["id"]
                        for d in status["devices"]
                        if d["kind"] == "arm" and d["role"].endswith("_follower")
                    }
                    self.cameras = {}
                self.job = JogJob(kind)
                self.job.message = "等待页面确认执行请求…"
                self.manager.motion_job = self.job
                self.revision = uuid4().hex
                self.inference_ms, self.chunks, self.started = None, 0, time.monotonic()
                self.last_action, self.commanded = None, False
                self.recorder, self.record_path = None, ""
                self.message = "等待页面确认执行请求…"
                self.thread = threading.Thread(
                    target=self._run,
                    args=(kind, copy.deepcopy(settings), status["config"]["robot"]),
                    daemon=True,
                    name="pi05-execution",
                )
                self.thread.start()

    def heartbeat(self, identity: str) -> None:
        with self.lock:
            if self.job and self.job.id == identity:
                self.job.heartbeat()

    def stop(self) -> None:
        with self.lock:
            self.revision = uuid4().hex  # Reject delayed requests submitted before Stop.
            self.load_cancel.set()
            if self.job and self.job.snapshot()["active"]:
                self.job.cancel.set()
                self.message = "已请求停止，正在等待机械臂停稳并保存数据…"

    def _check(self) -> None:
        self.job.check()
        if self.recorder and self.recorder.error:
            raise ValueError(f"录制异常：{self.recorder.error}")
        for identity in [*self.arms.values(), *self.cameras.values()]:
            worker = self.manager.workers[identity]
            if worker.stop.is_set() or not self.manager.present(worker.info):
                raise ValueError("设备已断开或拔插")
        for identity in self.cameras.values():
            self.manager.workers[identity].capture()

    def _check_policy(self) -> None:
        # 预测期间不响应取消（让 RPC 跑完，避免残留响应与半截状态），只检测设备/录制异常。
        # 取消会在预测之后由 _check 捕获，模型因此始终保留，停止后可再次运行。
        if self.recorder and self.recorder.error:
            raise ValueError(f"录制异常：{self.recorder.error}")
        for identity in [*self.arms.values(), *self.cameras.values()]:
            worker = self.manager.workers[identity]
            if worker.stop.is_set() or not self.manager.present(worker.info):
                raise ValueError("设备已断开或拔插")
        for identity in self.cameras.values():
            self.manager.workers[identity].capture()

    def _commands(self, operation: str, values: dict | None = None, stopping: bool = False) -> dict:
        check = (lambda: None) if stopping else self._check
        futures = {
            side: self.manager.workers[identity].command(operation, (values or {}).get(side), check)
            for side, identity in self.arms.items()
        }
        result, errors = {}, []
        deadline = time.monotonic() + (8 if operation == "prepare" else 3)
        for side, future in futures.items():
            try:
                while True:
                    if not stopping:
                        check()
                    try:
                        result[side] = future.result(timeout=0.02)
                        break
                    except FutureTimeout:
                        if time.monotonic() >= deadline:
                            raise ValueError(f"{side} 控制线程响应超时") from None
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            if not stopping:
                self.job.check()
            raise ValueError("；".join(errors))
        return result

    def _capture(self) -> dict:
        samples, images = {}, {}
        for identity in self.arms.values():
            samples[identity] = self.manager.workers[identity].capture()[0]
        for key, identity in self.cameras.items():
            images[key] = self.manager.workers[identity].capture()[1]
        return {
            "state": self.contract.state(samples, self.arms),
            "images": images,
            "captured": time.monotonic(),
            "points": np.asarray(
                [[*samples[i]["joint_radians"], samples[i]["gripper_m"]] for i in self.arms.values()]
            ),
        }

    def _record_command(self) -> tuple:
        with self.lock:
            return self.last_action, self.commanded

    def _set_command(self, points: np.ndarray) -> None:
        samples = {
            identity: {"joint_radians": point[:-1].tolist(), "gripper_m": float(point[-1])}
            for identity, point in zip(self.arms.values(), points, strict=True)
        }
        action = self.contract.state(samples, self.arms)
        with self.lock:
            self.last_action, self.commanded = action, True

    def _wait_target(self, targets: np.ndarray, timeout: float, stream: bool = False) -> None:
        deadline, stable, previous, stamps = time.monotonic() + timeout, 0, None, None
        while time.monotonic() < deadline:
            self._check()
            if stream:
                self._commands("send", dict(zip(self.arms, targets.tolist(), strict=True)))
            values = self._commands("read")
            points = np.asarray([value["point"] for value in values.values()])
            current = [value["stamp"] for value in values.values()]
            near = np.max(np.abs(points[..., :-1] - targets[..., :-1])) <= np.radians(0.8)
            if stamps is None or all(a != b for a, b in zip(stamps, current, strict=True)):
                still = previous is not None and np.max(
                    np.abs(points[..., :-1] - previous[..., :-1])
                ) <= np.radians(0.05)
                stable = stable + 1 if near and still else 0
                if stable >= 3:
                    return
                previous, stamps = points, current
            self.job.cancel.wait(0.025)
        raise ValueError("目标位置到位超时，已中止")

    def _execute(self, points: np.ndarray, durations: np.ndarray) -> None:
        bounds = np.r_[0, np.cumsum(durations)]
        started = previous = time.monotonic()
        while True:
            self._check()
            now = time.monotonic()
            if now - previous > 0.15:
                raise ValueError("控制循环超过 150 ms 未更新")
            elapsed = min(now - started, bounds[-1])
            index = min(int(np.searchsorted(bounds, elapsed, side="right") - 1), len(durations) - 1)
            alpha = (elapsed - bounds[index]) / durations[index]
            target = points[index] + alpha * (points[index + 1] - points[index])
            self._commands("send", dict(zip(self.arms, target.tolist(), strict=True)))
            self._set_command(target)
            previous = now
            if elapsed >= bounds[-1]:
                break
            self.job.cancel.wait(max(0, 1 / 30 - (time.monotonic() - now)))
        with self.lock:
            self.commanded = False

    def _run(self, kind: str, settings: dict, robot: str) -> None:
        attempted, predicting, recovery = False, False, False
        clamped: list[str] = []
        phase, message = "done", "执行完成，机械臂保持当前位置"
        try:
            self.job.wait_for_browser()
            self._check()
            self.job.progress("正在核对机械臂状态与限位…")
            attempted = True  # prepare can partially enable before raising.
            prepared = self._commands("prepare", dict.fromkeys(self.arms, kind == "home"))
            limits = np.asarray([value["limits"] for value in prepared.values()])
            if kind == "home":
                target = np.asarray(home_target(robot))
                if np.any(target < limits[..., 0]) or np.any(target > limits[..., 1]):
                    raise ValueError("初始位置超出控制器共同限位")
                # Same fixed NERO collection pose / Piper zero joint pose. Only followers move.
                for side in self.arms:
                    self.job.progress(f"{side} 从臂以 10% 速度回到初始位置…")
                    worker = self.manager.workers[self.arms[side]]
                    future = worker.command("home", target.tolist(), self._check)
                    while not future.done():
                        self._check()
                        self.job.cancel.wait(0.02)
                    future.result()
                    targets = np.asarray([v["point"] for v in self._commands("read").values()])
                    targets[list(self.arms).index(side), :-1] = target
                    self._wait_target(targets, 30)
                message = "从臂已回到初始位置，保持使能；主臂未动作"
            else:
                predicting = True
                self.policy.call({"op": "reset"}, self._check_policy)
                predicting = False
                if settings.get("record"):
                    self.job.progress("正在建立本次 LeRobot 数据集…")
                    self.recorder = self.recorder_factory(
                        settings, self.contract, self._capture, self._record_command
                    )
                    self.record_path = str(self.recorder.path)
                while True:
                    self._check()
                    self.job.progress("正在读取观测并运行 π0.5 预测…")
                    observation = self._capture()
                    predicting = True
                    result = self.policy.call(
                        {
                            "op": "predict",
                            "state": observation["state"],
                            "images": observation["images"],
                            "task": settings["task"],
                        },
                        self._check_policy,
                    )
                    predicting = False
                    self._check()
                    self.inference_ms = result["inference_ms"]
                    current = np.asarray([v["point"] for v in self._commands("read").values()])
                    if time.monotonic() - observation["captured"] > 10:
                        raise ValueError("预测耗时过长，观测已过期")
                    actions = self.contract.actions(result["actions"])
                    points, durations, warnings = plan(actions, self.contract, current, limits, self._check)
                    if warnings:
                        clamped.extend(warnings)
                    if time.monotonic() - observation["captured"] > 10:
                        raise ValueError("规划耗时过长，观测已过期")
                    if self.recorder:
                        self.recorder.chunk(actions, self.inference_ms)
                    self.job.progress(f"正在低速执行第 {self.chunks + 1} 段 · {len(actions)} 个动作")
                    self._execute(points, durations)
                    self.chunks += 1
                    if kind == "single" and time.monotonic() - self.started >= settings.get("duration", 60.0):
                        message = f"单集测试完成，共执行 {self.chunks} 段，机械臂保持当前位置"
                        break
        except JogStopped as exc:
            phase, message = "stopped", str(exc)
        except Exception as exc:
            phase, message = "error", str(exc)
        finally:
            # Invalidate every queued control RPC before adding stop commands.
            # A late owner-thread wakeup must never replay a timed-out action.
            self.job.cancel.set()
            if attempted:
                self.job.progress("正在确认机械臂停稳…")
                try:
                    self._commands("hold", stopping=True)
                except Exception as exc:
                    recovery = True
                    phase, message = "error", f"{message}；停止未确认：{exc}；请现场检查后重启平台"
            if predicting:
                self.policy.close()
                self.policy = None
                self.model_state = "empty"
                message += "；已终止未完成预测，下次执行需重新加载模型"
            with self.lock:
                self.commanded = False
            if self.recorder:
                self.job.progress("机械臂已停止，正在保存本次视频与数据…")
                try:
                    self.recorder.finish(phase, message)
                except Exception as exc:
                    phase, message = "error", f"{message}；保存失败：{exc}（原始数据保留）"
            if clamped:
                message += f"；已将 {len(clamped)} 处模型越界目标钳制到限位"
            with self.lock:
                self.elapsed = round(time.monotonic() - self.started, 1)
                self.message = message
                self.job.finish(phase, message, recovery)
                self.revision = uuid4().hex

    def close(self) -> None:
        self.closed = True
        self.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(8)
        if self.policy:
            self.policy.close()
            self.policy = None
