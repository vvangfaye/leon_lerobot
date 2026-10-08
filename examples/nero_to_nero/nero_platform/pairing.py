"""单对/左右双对的固定顺序与通道解析，不执行任何硬件操作。"""

from dataclasses import asdict, dataclass
from typing import Any

from .runtime import JOINT_KEYS, Runtime


@dataclass(frozen=True)
class Pair:
    name: str
    leader_label: str
    follower_label: str
    leader: str
    follower: str

    @property
    def display(self) -> str:
        return f"{self.leader_label}→{self.follower_label}"


def labels(state: dict[str, Any]) -> list[tuple[str, str, str]]:
    if state["pair_mode"] == "dual":
        return [("left", "主左", "从左"), ("right", "主右", "从右")]
    return [("single", state["leader_label"], state["follower_label"])]


def joint_keys(mode: str) -> list[str]:
    if mode == "single":
        return JOINT_KEYS[:]
    if mode == "dual":
        return [f"{side}.{key}" for side in ("left", "right") for key in JOINT_KEYS]
    raise ValueError("采集模式必须为 single 或 dual")


def resolve(runtime: Runtime, *, require_detected: bool = True) -> list[Pair]:
    pairs = []
    used = set()
    for name, leader, follower in labels(runtime.state):
        channels = []
        for label in (leader, follower):
            matches = [ch for ch, info in runtime.channels.items() if info["label"] == label]
            if len(matches) != 1 or matches[0] in used:
                raise ValueError(f"{label} 通道映射不唯一，请为每条臂分配独立 CAN 通道")
            ch = matches[0]
            info = runtime.channels[ch]
            if info["arm"] is None or (require_detected and not info["detected"]):
                raise ValueError(f"{label} ({ch}) 尚未连接，请先连接对应通道")
            channels.append(ch)
            used.add(ch)
        pairs.append(Pair(name, leader, follower, *channels))
    return pairs


def targets(pairs: list[Pair]) -> dict[str, str]:
    return {
        ch: label
        for pair in pairs
        for ch, label in ((pair.leader, pair.leader_label), (pair.follower, pair.follower_label))
    }


def describe(pairs: list[Pair]) -> list[dict[str, str]]:
    return [asdict(pair) for pair in pairs]
