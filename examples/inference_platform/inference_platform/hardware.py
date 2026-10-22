"""Arm feedback, explicit identification jog and collection camera adapters.

Driver imports happen only after an explicit device operation. Importing the web
application neither imports model runtimes nor opens hardware.
"""

from __future__ import annotations

import importlib
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .discovery import camera_attachment, can_inventory, rgb_inventory
from .jog import ArmJog


def camera_module(name: str) -> Any:
    collection = Path(__file__).resolve().parents[2] / "nero_to_nero"
    if str(collection) not in sys.path:
        sys.path.insert(0, str(collection))
    return importlib.import_module(f"nero_platform.{name}")


def discover() -> tuple[dict, dict]:
    devices = {**can_inventory(), **rgb_inventory()}
    try:
        depth, detection = camera_module("depth_discovery").detect()
        serials = {info["serial"] for info in depth.values()}
        devices = {
            key: info for key, info in devices.items()
            if info["kind"] == "arm" or (
                info.get("serial") not in serials and not any(serial and serial in key for serial in serials)
            )
        }
        for identity, info in depth.items():
            entry = {"id": identity, "kind": "camera", **info}
            entry["attachment"] = camera_attachment(entry)
            devices[identity] = entry
    except Exception as exc:
        detection = {"rgbd": {"label": "深度相机", "state": "error", "message": str(exc)}}
    return devices, detection


class FreshFeedback:
    """Require advancing device timestamps, using host time only for expiry."""

    def __init__(self) -> None:
        self.seen: dict[str, tuple[float, float, bool]] = {}

    def read(self, name: str, sample: Any, now: float) -> Any:
        stamp = float(getattr(sample, "timestamp", 0))
        if not math.isfinite(stamp) or stamp <= 0 or sample.msg is None:
            raise ValueError(f"{name} 尚无有效反馈")
        previous = self.seen.get(name)
        if previous is None or stamp < previous[0]:
            self.seen[name] = (stamp, now, False)
            raise ValueError(f"{name} 等待连续更新")
        if stamp > previous[0]:
            self.seen[name] = (stamp, now, True)
        current = self.seen[name]
        if not current[2] or now - current[1] >= 0.5:
            raise ValueError(f"{name} 超过 0.5 秒未更新")
        return sample.msg


class ArmReader:
    def __init__(self, info: dict, robot: str) -> None:
        sdk = importlib.import_module("pyAgxArm")
        self.robot = robot
        self.info = info
        self.motion = None
        self.count = {"nero": 7, "piper": 6}[robot]
        self.fresh = FreshFeedback()
        config = sdk.create_agx_arm_config(
            robot=robot,
            firmeware_version=sdk.NeroFW.V121 if robot == "nero" else sdk.PiperFW.DEFAULT,
            channel=info["id"],
        )
        self.limits = [config.get("joint_limits", {}).get(f"joint{i}") for i in range(1, self.count + 1)]
        self.arm = sdk.AgxArmFactory.create_arm(config)
        self.gripper = None

    def connect(self) -> None:
        self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.arm.connect()

    def sample(self) -> tuple[dict, bytes | None]:
        now = time.monotonic()
        result: dict = {"ready": False, "joint_degrees": None, "gripper_m": None, "source": None}
        errors = []
        for name, method in (("普通关节", self.arm.get_joint_angles),
                             ("示教关节", self.arm.get_leader_joint_angles)):
            try:
                values = [float(value) for value in self.fresh.read(name, method(), now)]
                if len(values) != self.count or not all(math.isfinite(value) for value in values):
                    raise ValueError("关节维数或数值无效")
                if result["joint_degrees"] is None:
                    result.update(joint_radians=values, joint_degrees=[round(math.degrees(value), 1) for value in values], source=name)
            except (ValueError, TypeError, AttributeError) as exc:
                if name == "普通关节":
                    errors.append(str(exc))
        try:
            grip = self.fresh.read("夹爪", self.gripper.get_gripper_status(), now)
            value = float(grip.value)
            if not math.isfinite(value) or not -0.002 <= value <= 0.102:
                raise ValueError("夹爪开度无效")
            result["gripper_m"] = value
            if getattr(getattr(grip, "foc_status", None), "driver_error_status", False):
                errors.append("夹爪驱动报告错误")
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
        result["ready"] = result["source"] == "普通关节" and not errors
        result["message"] = "普通关节与夹爪反馈正常" if result["ready"] else "；".join(errors)
        return result, None

    def close(self) -> None:
        self.arm.disconnect()

    def control(self, operation: str, value: Any, check: Any) -> Any:
        from .motion import ArmMotion
        if operation == "disable":
            self.motion = None
            deadline = time.monotonic() + 3
            while not self.arm.disable():
                if time.monotonic() >= deadline:
                    raise ValueError("失能超时，未确认关节已失能")
                time.sleep(0.02)
            if self.gripper is not None:
                self.gripper.disable_gripper()
            return {"disabled": True}
        if operation == "prepare":
            self.motion = ArmMotion(self, home=bool(value))
            return self.motion.prepare(check)
        if self.motion is None:
            if operation == "hold":
                return None
            raise ValueError("机械臂未准备")
        if operation == "read":
            return self.motion.read()
        if operation == "send":
            return self.motion.send(value, check)
        if operation == "home":
            return self.motion.home(value, check)
        if operation == "hold":
            try:
                return self.motion.hold()
            finally:
                self.motion = None
        raise ValueError("未知机械臂控制操作")

    def measure_limits(self, stop: threading.Event, progress: Any) -> dict:
        jl = camera_module("joint_limits")
        deadline = time.monotonic() + 3
        while not self.arm.enable():
            if time.monotonic() >= deadline:
                raise ValueError("使能超时，未开始测量")
            time.sleep(0.02)
        if self.gripper is not None:
            try:
                self.gripper.disable_gripper()
            except Exception:
                pass
        time.sleep(0.3)
        self.arm.set_leader_mode()
        time.sleep(0.3)
        setattr(self.arm, "_channel", self.info["id"])
        try:
            return jl.measure(self.arm, stop, progress)
        finally:
            try:
                time.sleep(0.1)
                self.arm.set_follower_mode()
                time.sleep(0.3)
            except Exception:
                pass

    def jog(self, check: Any, progress: Any) -> tuple[str, str, bool]:
        if any(limit is None for limit in self.limits):
            return "error", "SDK 未提供完整关节限位，未执行微动", False
        return ArmJog(
            self.arm, self.robot, self.limits, FreshFeedback(), label=self.info.get("label")
        ).run(check, progress)


class CameraReader:
    def __init__(self, info: dict, robot: str) -> None:
        self.info = info
        self.backend = camera_module("camera_io").backend(info)
        self.preview = camera_module("camera_preview")
        self.focus_setting = None
        if info["backend"] == "opencv":
            runtime_module = camera_module("runtime")
            settings = runtime_module.Runtime()
            camera_module("config").load_saved(
                settings, legacy_path=runtime_module.EXAMPLE_DIR / "collect_config.json",
            )
            if settings.state["config_error"]:
                raise ValueError(settings.state["config_error"])
            self.focus_setting = settings.state["camera_focus_settings"].get(info["id"])
            self.camera = camera_module("cameras").create_preview_camera(info["path"])
        else:
            self.camera = self.preview.FACTORIES[info["backend"]](info)
        self.last_stamp: float | None = None
        self.jpeg: bytes | None = None
        self.rgb = None
        self.captured = 0.0
        self.frames = 0
        self.started = time.monotonic()
        self.fps = 0.0

    def connect(self) -> None:
        self.backend.connect(self.camera, self.info, self.focus_setting)

    def sample(self) -> tuple[dict, bytes | None]:
        images, timing = self.backend.snapshot(self.camera)
        frame = images["rgb"]
        stamp = float(timing["capture_timestamp"])
        age = float(timing["age_ms"])
        if not math.isfinite(stamp) or not math.isfinite(age) or not 0 <= age < 500:
            raise ValueError("相机画面已过期")
        if frame.ndim != 3 or frame.shape[2] != 3 or str(frame.dtype) != "uint8":
            raise ValueError("相机未返回 RGB uint8 图像")
        if stamp != self.last_stamp:
            self.jpeg = self.preview.encode_jpeg(frame)
            self.last_stamp = stamp
            self.frames += 1
        elapsed = time.monotonic() - self.started
        if elapsed >= 1:
            self.fps, self.frames, self.started = round(self.frames / elapsed, 1), 0, time.monotonic()
        self.rgb = frame.copy()
        self.captured = time.monotonic() - age / 1000
        metadata = getattr(self.camera, "metadata", {})
        details = [metadata[key]["message"] for key in ("white_balance", "focus")
                   if metadata.get(key, {}).get("state") in ("verified", "warning")]
        return {
            "ready": True, "message": " · ".join(["画面正常", *details]), "age_ms": round(age),
            "width": frame.shape[1], "height": frame.shape[0], "preview_fps": self.fps,
        }, self.jpeg

    def close(self) -> None:
        self.camera.disconnect()


def make_reader(info: dict, robot: str) -> ArmReader | CameraReader:
    return (ArmReader if info["kind"] == "arm" else CameraReader)(info, robot)
