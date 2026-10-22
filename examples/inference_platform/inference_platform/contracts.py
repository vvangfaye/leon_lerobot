"""Explicit π0.5 observation/action contracts; no dimension-only inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMERAS = {
    "head": "head",
    "base_0_rgb": "head",
    "left": "left",
    "left_wrist_0_rgb": "left",
    "right": "right",
    "right_wrist_0_rgb": "right",
}
EEF_NAMES = ["eef_x", "eef_y", "eef_z", *[f"rot6d_col{c}_{a}" for c in (0, 1) for a in "xyz"], "gripper"]


class Contract:
    def __init__(self, path: str, config: dict) -> None:
        self.path = Path(path).expanduser().resolve()
        self.config = {key: config[key] for key in ("robot", "arms", "space")}
        if config.get("model", "pi05") != "pi05":
            raise ValueError("本阶段只支持 π0.5；OpenWAM 暂不接入")
        self.raw = json.loads((self.path / "config.json").read_text())
        if self.raw.get("type") != "pi05":
            raise ValueError("请选择 π0.5 checkpoint")
        self.count = {"nero": 7, "piper": 6}[config["robot"]]
        self.sides = ["left", "right"] if config["arms"] == "dual" else [""]
        base = (
            [*[f"joint{i}.pos" for i in range(1, self.count + 1)], "gripper.pos"]
            if config["space"] == "joint"
            else EEF_NAMES
        )
        self.names = self.raw.get("action_feature_names")
        expected = [f"{side}.{name}" if side else name for side in self.sides for name in base]
        if self.names != expected:
            # Explicitly sided single-arm checkpoints retain their side contract.
            matches = [
                side for side in ("left", "right") if self.names == [f"{side}.{name}" for name in base]
            ]
            if config["arms"] != "single" or len(matches) != 1:
                raise ValueError("动作名称/顺序与所选机械臂、单/双臂、关节/EEF 不一致；不按维数猜测")
            self.sides = matches
        self.dim = len(self.names)
        if self.raw.get("input_features", {}).get("observation.state", {}).get("shape") != [
            self.dim
        ] or self.raw.get("output_features", {}).get("action", {}).get("shape") != [self.dim]:
            raise ValueError("状态、动作维数与动作名称不一致")
        if self.raw.get("use_peft") or self.raw.get("rtc_config"):
            raise ValueError("首版需要完整 π0.5 权重，暂不支持 PEFT / RTC checkpoint")
        self.cameras = {}
        for key, feature in self.raw["input_features"].items():
            if key == "observation.state":
                continue
            if not key.startswith("observation.images.") or feature.get("type") != "VISUAL":
                raise ValueError(f"暂不支持输入 {key}")
            role = CAMERAS.get(key.removeprefix("observation.images."))
            if role is None or role in self.cameras.values():
                raise ValueError(f"无法唯一映射相机 {key}；支持 head/left/right 和 base/wrist_0_rgb")
            if len(feature["shape"]) != 3 or feature["shape"][0] != 3:
                raise ValueError("仅支持三通道 RGB 相机输入")
            self.cameras[key] = role
        if not self.cameras:
            raise ValueError("checkpoint 缺少相机输入")
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (self.path / name).is_file():
                raise ValueError(f"checkpoint 缺少 {name}")
        self.steps = int(self.raw.get("n_action_steps", 50))
        if not 1 <= self.steps <= int(self.raw.get("chunk_size", 50)) <= 100:
            raise ValueError("动作段长度必须为 1–100")

    def bind(self, status: dict) -> tuple[dict, dict]:
        if not status["confirmed"] or status["config"] != self.config:
            raise ValueError("请先连接设备，按当前机械臂/动作空间确认用途")
        followers = {
            d["role"].removesuffix("_follower"): d["id"]
            for d in status["devices"]
            if d["kind"] == "arm" and d["role"].endswith("_follower")
        }
        if len(followers) != len(self.sides):
            raise ValueError("从臂数量与 checkpoint 不一致")
        sides = list(followers) if self.sides == [""] else self.sides
        if set(sides) != set(followers):
            raise ValueError("checkpoint 左/右臂与所分配从臂不一致")
        cameras = {d["role"]: d["id"] for d in status["devices"] if d["kind"] == "camera" and d["role"]}
        if not set(self.cameras.values()) <= cameras.keys():
            raise ValueError(
                "缺少模型需要的相机：" + ", ".join(sorted(set(self.cameras.values()) - cameras.keys()))
            )
        return {side: followers[side] for side in sides}, {
            key: cameras[role] for key, role in self.cameras.items()
        }

    def state(self, samples: dict, arms: dict) -> np.ndarray:
        result = []
        for side in arms:
            sample = samples[arms[side]]
            q, g = sample["joint_radians"], sample["gripper_m"]
            if len(q) != self.count or not np.isfinite([*q, g]).all() or not -0.002 <= g <= 0.102:
                raise ValueError("机械臂原始反馈无效")
            if self.config["space"] == "joint":
                result.extend([*q, float(np.clip(g, 0, 0.1))])
            else:
                from .kinematics import eef

                result.extend(eef(self.config["robot"], q, g))
        return np.asarray(result, dtype=np.float32)

    def actions(self, actions: np.ndarray) -> np.ndarray:
        result = np.asarray(actions, dtype=float)
        if (
            result.ndim != 2
            or result.shape[1] != self.dim
            or not 1 <= len(result) <= self.steps
            or not np.isfinite(result).all()
        ):
            raise ValueError("模型返回的动作形状或数值无效")
        return result
