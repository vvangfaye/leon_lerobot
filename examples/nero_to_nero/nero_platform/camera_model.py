"""相机角色与数据流命名；不依赖相机 SDK 或运行状态。"""

ROLES = ("left", "right", "head")
ROLE_LABELS = {"left": "左", "right": "右", "head": "头部"}


def stream_names(role: str, kind: str) -> tuple[str, ...]:
    if role not in ROLES or kind not in ("rgb", "rgbd"):
        raise ValueError(f"不支持的相机角色或类型：{role} / {kind}")
    return (role, f"{role}_depth") if kind == "rgbd" else (role,)


def is_depth(name: str) -> bool:
    # depth 是旧数据集的单路深度字段，仅在回放与质检中兼容读取。
    return name == "depth" or name in {f"{role}_depth" for role in ROLES}


def stream_role(name: str) -> str:
    return name.removesuffix("_depth")


def stream_label(name: str) -> str:
    if name == "depth":
        return "深度（旧字段）"
    return ROLE_LABELS.get(stream_role(name), name) + ("深度" if is_depth(name) else " RGB")
