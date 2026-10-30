"""采集反馈校验；不假定 SDK 与主机时钟同源。"""

import math
from collections.abc import Callable
from typing import Any

FEEDBACK_TIMEOUT_SECONDS = 0.5


class FeedbackError(RuntimeError):
    """反馈不可用时停止发送新的跟随指令，不用零值代替。"""


class CommandError(RuntimeError):
    """一侧指令失败即停止整集；不把部分下发的帧作为完整数据，也不自动重发。"""


def native_action(readings: dict[str, list[float]]) -> list[float]:
    """VLA capture's absolute seven-joint target and 0–90 mm gripper target."""
    return readings["主臂关节"][:] + [min(max(readings["主臂夹爪"][0], 0.0), 0.09)]


def send_js_action(follower: dict[str, Any], action: list[float], force: float) -> None:
    """Shared VLA/RL manual follow command: no additional trajectory filtering."""
    follower["arm"].move_js(action[:7])
    follower["gripper"].move_gripper_m(value=action[7], force=force)


class RobotFeedback:
    def __init__(self, leader: dict, follower: dict):
        if leader["gripper"] is None or follower["gripper"] is None:
            raise FeedbackError("每对主从臂的 8 维采集均需要夹爪反馈，请检查夹爪连接")
        self.sources: dict[str, tuple[Callable, int]] = {
            "主臂关节": (leader["arm"].get_leader_joint_angles, 7),
            "主臂夹爪": (leader["gripper"].get_gripper_ctrl_states, 1),
            "从臂关节": (follower["arm"].get_joint_angles, 7),
            "从臂夹爪": (follower["gripper"].get_gripper_status, 1),
        }
        self.seen: dict[str, tuple[float, float, bool]] = {}

    @property
    def ready(self) -> bool:
        return len(self.seen) == len(self.sources) and all(v[2] for v in self.seen.values())

    def read(self, now: float) -> dict[str, list[float]]:
        result = {}
        for name, (read, size) in self.sources.items():
            try:
                sample = read()
                stamp = float(sample.timestamp)
                values = list(sample.msg) if size == 7 else [sample.msg.value]
                values = [float(v) for v in values]
                if len(values) != size or not all(math.isfinite(v) for v in values):
                    raise ValueError("数值缺失、维数错误或含 NaN/Inf")
                if not math.isfinite(stamp) or stamp <= 0:
                    raise ValueError("时间戳无效")
                if name == "从臂夹爪":
                    # 夹爪驱动保护态（过流/过载）会静默失能并锁死，运动指令无效；
                    # 一旦检测到立即停止跟随，避免继续空转顶到极限。
                    foc = getattr(sample.msg, "foc_status", None)
                    if foc is not None and getattr(foc, "driver_error_status", False):
                        raise FeedbackError(
                            "从臂夹爪驱动错误保护（过流/过载），已停止跟随；请断电重启夹爪后重新连接"
                        )
            except FeedbackError:
                raise
            except Exception as exc:
                raise FeedbackError(f"{name}反馈无效：{exc}") from exc
            old = self.seen.get(name)
            if old is None:
                self.seen[name] = (stamp, now, False)
            elif stamp < old[0]:
                raise FeedbackError(f"{name}时间戳倒退，请检查设备重启或通信状态")
            elif stamp > old[0]:
                self.seen[name] = (stamp, now, True)
            elif now - old[1] >= FEEDBACK_TIMEOUT_SECONDS:
                raise FeedbackError(f"{name}反馈超过 {FEEDBACK_TIMEOUT_SECONDS:.1f} 秒未更新")
            result[name] = values
        return result


def camera_sample(camera: Any) -> tuple[Any, dict[str, float]]:
    """图像和采集时间戳来自同一次加锁快照，帧龄由相机实现计算。"""
    frame, timestamp, age_ms = camera.read_latest_with_timestamp()
    if not math.isfinite(timestamp) or not math.isfinite(age_ms) or age_ms < 0:
        raise ValueError("相机帧时间戳无效")
    return frame, {"capture_timestamp": timestamp, "age_ms": round(age_ms, 3)}
