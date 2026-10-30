"""Direct right-follower hand guiding, separate from limit measurement and recording."""

import math
import time
from typing import Any

from . import devices, hardware_status, joint_limits
from .runtime import Runtime


def _wait_for_drag_feedback(arm: Any) -> None:
    reader = joint_limits.AngleReader(arm)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        sample = reader.read(time.monotonic())
        if sample is not None and sample[1] == "拖动反馈":
            return
        time.sleep(0.02)
    raise TimeoutError("未收到完整的新示教关节反馈")


def start(runtime: Runtime, ch: str) -> None:
    """Release gripper control and request zero-force drag. Never send joint targets."""
    with runtime.hand_guiding_lock:
        with runtime.lifecycle_lock:
            if ch != "can1" or runtime.channels.get(ch, {}).get("label") != "从右":
                raise ValueError("直接拖动目前仅支持从右 can1")
            if runtime.operation_busy() or runtime.state["connected"] or runtime.state["measuring"]:
                raise RuntimeError("先停止遥操作、录制或其他设备操作")
            if runtime.state["recovery_required"]:
                raise RuntimeError(runtime.state["recovery_required"])
            info = runtime.channels[ch]
            if not info.get("detected") or info.get("arm") is None:
                raise RuntimeError("先连接并读取从右 can1")
            runtime.state.update(
                hand_guiding_channel=ch,
                hand_guiding_phase="starting",
                hand_guiding_message="正在准备从右直接拖动…",
            )
            runtime.device_busy.add(ch)
        changed = False
        try:
            hardware_status.require_ready_for_motion(runtime, ch)
            gripper = info.get("gripper")
            if gripper is not None:
                packet = gripper.get_gripper_status()
                width = float(packet.msg.value)
                if not math.isfinite(width):
                    raise RuntimeError("夹爪开度反馈无效")
                runtime.state["hand_guiding_gripper_m"] = width
                # Normal gripper position control can conflict with teaching mode.
                changed = True
                devices._disable_and_confirm(gripper.disable_gripper, lambda: [gripper.get_gripper_status()])
            changed = True
            devices.set_control_role(runtime, ch, "leader")
            _wait_for_drag_feedback(info["arm"])
            with runtime.lifecycle_lock:
                runtime.state.update(
                    hand_guiding_phase="active",
                    hand_guiding_message="可直接扶右从臂摆姿态；不要捏夹爪。停稳后点击结束拖动。",
                )
                info["status"] = "直接拖动示教中（不测限位、不录制）"
        except Exception as exc:
            try:
                if changed:
                    devices.set_control_role(runtime, ch, "follower")
                    hardware_status.require_ready_for_motion(runtime, ch)
            except Exception as recovery:
                runtime.state.update(
                    hand_guiding_phase="error",
                    hand_guiding_message=f"拖动启动失败：{exc}；恢复失败：{recovery}，请重试结束拖动",
                )
                raise RuntimeError(runtime.state["hand_guiding_message"]) from exc
            with runtime.lifecycle_lock:
                runtime.state.update(
                    hand_guiding_channel=None,
                    hand_guiding_phase="idle",
                    hand_guiding_message=f"拖动未启动，已恢复受控模式：{exc}",
                )
                runtime.device_busy.discard(ch)
            raise


def stop(runtime: Runtime, ch: str) -> None:
    """Restore follower mode and holding torque; leave gripper released, without homing."""
    with runtime.hand_guiding_lock:
        with runtime.lifecycle_lock:
            active = runtime.state.get("hand_guiding_channel")
            if active is None:
                return
            if active != ch:
                raise ValueError("停止通道与正在拖动的通道不一致")
            runtime.state.update(
                hand_guiding_phase="stopping", hand_guiding_message="正在结束拖动并恢复位置保持…"
            )
        try:
            devices.set_control_role(runtime, ch, "follower")
            hardware_status.require_ready_for_motion(runtime, ch)
        except Exception as exc:
            runtime.state.update(
                hand_guiding_phase="error", hand_guiding_message=f"结束拖动未确认：{exc}，请重试"
            )
            raise
        with runtime.lifecycle_lock:
            runtime.state.update(
                hand_guiding_channel=None,
                hand_guiding_phase="idle",
                hand_guiding_message="已结束拖动，关节保持使能；夹爪保持释放。可读取标定点。",
            )
            runtime.device_busy.discard(ch)
            runtime.channels[ch]["status"] = "已结束直接拖动，恢复从臂受控模式"
