"""只读硬件状态；只有时间戳实际更新的反馈才用于当前状态显示。"""

import math
import time
from typing import Any

from .runtime import Runtime

STALE_SECONDS = 1.0
CONTROL_MODES = {
    0: "待机",
    1: "CAN 受控",
    2: "示教",
    3: "以太网控制",
    4: "Wi-Fi 控制",
    5: "遥控",
    6: "主从联动（0x06，角色未确认）",
    7: "离线轨迹",
    8: "TCP 控制",
}
ARM_STATES = {
    0: "正常",
    1: "急停",
    2: "无解",
    3: "奇异点",
    4: "目标超限",
    5: "关节通信故障",
    6: "抱闸未释放",
    7: "碰撞",
    8: "示教拖动超速",
    9: "关节状态异常",
    10: "其他故障",
    11: "示教记录",
    12: "示教执行",
    13: "示教暂停",
    14: "主控过温",
    15: "释放电阻过温",
}


def read_status(runtime: Runtime, ch: str) -> dict[str, Any]:
    info = runtime.channels[ch]
    arm = info["arm"]
    reads = {}
    errors = []
    now = time.monotonic()
    with runtime.lifecycle_lock:
        cache = runtime.feedback_seen.get(ch)
        if cache is None or cache["identity"] != id(arm):
            cache = {"identity": id(arm), "samples": {}}
            runtime.feedback_seen[ch] = cache

    def read(key: str, obj: Any, method: str, *args: Any) -> Any:
        try:
            value = getattr(obj, method)(*args) if obj is not None else None
            stamp = float(getattr(value, "timestamp", 0))
            valid = value is not None and value.msg is not None and math.isfinite(stamp) and stamp > 0
            fresh = False
            if valid:
                with runtime.lifecycle_lock:
                    old = cache["samples"].get(key)
                    if old is None or stamp < old[0]:
                        cache["samples"][key] = (stamp, now, False)
                    elif stamp > old[0]:
                        cache["samples"][key] = (stamp, now, True)
                    current = cache["samples"][key]
                    fresh = current[2] and now - current[1] <= STALE_SECONDS
            reads[key] = fresh
            return value.msg if fresh else None
        except Exception as exc:
            reads[key] = False
            if obj is not None and hasattr(obj, method):
                errors.append(f"{key}: {exc}")
            return None

    ordinary = read("关节", arm, "get_joint_angles")
    teaching = read("示教", arm, "get_leader_joint_angles")
    controller = read("控制器", arm, "get_arm_status")
    joints = []
    for joint in range(1, 8):
        driver = read(f"J{joint}", arm, "get_driver_states", joint)
        enabled = getattr(getattr(driver, "foc_status", None), "driver_enable_status", None)
        joints.append(enabled if isinstance(enabled, bool) else None)
    gripper = info["gripper"]
    grip_status = read("夹爪", gripper, "get_gripper_status")
    grip_enabled = getattr(getattr(grip_status, "foc_status", None), "driver_enable_status", None)
    if grip_enabled is not True and grip_enabled is not False:
        grip_enabled = None

    communication = (
        "disconnected"
        if arm is None
        else ("live" if any(v for k, v in reads.items() if k != "夹爪") else "unknown")
    )
    if all(v is True for v in joints):
        enabled_state, enabled_label = "enabled", "全部使能（7/7）"
    elif all(v is False for v in joints):
        enabled_state, enabled_label = "disabled", "全部失能（7/7）"
    elif any(v is None for v in joints):
        enabled_state, enabled_label = "unknown", "使能状态未完整确认"
    else:
        enabled_state, enabled_label = "partial", "部分使能"
    ctrl = getattr(controller, "ctrl_mode", None)
    requested_role = info.get("requested_role")
    mode_label = CONTROL_MODES.get(ctrl, "未知")
    if ctrl == 6 and requested_role in ("leader", "follower"):
        role_label = "主臂示教" if requested_role == "leader" else "从臂受控"
        mode_label = f"主从联动（0x06，已请求{role_label}）"
    if ctrl is None and teaching is not None:
        mode_label = "收到示教广播（控制状态未知）"
    code = getattr(controller, "arm_status", None)
    error_code = getattr(controller, "err_code", None)
    health = (
        "unknown"
        if code is None or error_code is None
        else ("normal" if code == 0 and error_code == 0 else "fault")
    )
    health_label = (
        "未知（控制状态无完整新反馈）" if health == "unknown" else ARM_STATES.get(code, f"状态码 {code}")
    )
    # Nero 在所有关节失能时也可上报 0x06。保留控制器状态，不据此断言机械抱死，
    # 也不把这个组合当作必须先清除的故障，否则会挡住正常的显式使能流程。
    # 这不是运动就绪：使能后仍必须等到 arm_status=0、err_code=0 才能运动。
    if code == 6 and error_code == 0 and enabled_state == "disabled":
        health = "not_ready"
        health_label = "关节未使能／控制器未就绪（状态码 0x06）"
    if error_code:
        health_label += f"，故障码 {error_code}"
    # 0x06 是主从联动模式，不等同于主臂。优先按明确请求选择位置来源；
    # 从臂始终使用实际关节反馈，不能用主臂目标广播替代。
    use_teaching = requested_role == "leader" or (
        requested_role != "follower"
        and (ctrl == 2 or (ctrl in (None, 6) and ordinary is None and teaching is not None))
    )
    positions = teaching if use_teaching else ordinary
    angles = None
    if (
        isinstance(positions, (list, tuple))
        and len(positions) == 7
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in positions)
    ):
        angles = [round(math.degrees(v), 1) for v in positions]
    if arm is None:
        guidance = "先连接并读取状态；连接不会使能、失能或复位。"
    elif code == 1:
        guidance = "设备报告急停：先停止采集并托稳机械臂，按设备说明排查和解除急停；本平台不自动复位。"
    elif health == "fault":
        guidance = "检测到异常，暂停运动；可读取状态或在托稳后原地失能。"
    elif health == "not_ready":
        guidance = (
            "SDK 将 0x06 称为“抱闸未释放”，不能单凭此项判断机械臂锁死。"
            "连接不会使能；运动操作会先使能并重新确认设备正常。"
        )
    elif communication != "live" or enabled_state == "unknown" or health == "unknown":
        guidance = "反馈未完整确认；重新检查状态。运动前仍须确认使能与设备正常。"
    elif enabled_state == "enabled":
        guidance = "电机保持力矩；退出脚本不会自动失能。"
    elif enabled_state == "partial":
        guidance = "各关节状态不一致；完成使能确认前不执行运动。"
    else:
        guidance = "关节已失能；开始采集时会重新准备设备。"
    return {
        "communication": {
            "state": communication,
            "label": {"live": "反馈正在更新", "unknown": "等待新反馈／反馈已过期", "disconnected": "未连接"}[
                communication
            ],
        },
        "mode": {"code": ctrl, "label": mode_label, "requested_role": requested_role},
        "enabled": {"state": enabled_state, "label": enabled_label, "joints": joints},
        "health": {"state": health, "label": health_label, "code": code, "error_code": error_code},
        "gripper": {"present": gripper is not None, "enabled": grip_enabled},
        "angles": angles,
        "angles_source": ("teaching_broadcast" if use_teaching else "joint_feedback")
        if angles is not None
        else None,
        "guidance": guidance,
        "errors": errors,
    }


def inspect_status(runtime: Runtime, ch: str) -> dict[str, Any]:
    read_status(runtime, ch)
    time.sleep(0.05)
    return read_status(runtime, ch)


def require_ready_for_motion(runtime: Runtime, ch: str) -> None:
    # 给切换模式后的低频反馈留出时间；只等反馈，不重复使能或复位。
    deadline = time.monotonic() + STALE_SECONDS
    while True:
        status = read_status(runtime, ch)
        mode = status["mode"]
        controlled = (mode["code"] == 1 and mode["requested_role"] != "leader") or (
            mode["code"] == 6 and mode["requested_role"] == "follower"
        )
        if (
            status["health"]["state"] == "normal"
            and status["enabled"]["state"] == "enabled"
            and status["angles"] is not None
            and status["angles_source"] == "joint_feedback"
            and controlled
        ):
            return
        health = status["health"]
        # 驱动使能反馈与控制器状态可能先后到达。0x06 可短暂等待消退，
        # 期间不发送运动；其他故障或到期仍未正常立即拒绝。
        waiting_for_ready = health["code"] == 6 and health["error_code"] == 0
        if (health["state"] == "fault" and not waiting_for_ready) or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    details = []
    if not controlled:
        details.append("未确认可接收运动指令的控制模式和角色")
    if status["angles_source"] != "joint_feedback":
        details.append("实际关节位置无有效新反馈")
    detail = "；" + "；".join(details) if details else ""
    raise RuntimeError(
        f"{ch} 暂不执行运动：{status['mode']['label']}；{status['health']['label']}；{status['enabled']['label']}{detail}。请先检查硬件状态"
    )
