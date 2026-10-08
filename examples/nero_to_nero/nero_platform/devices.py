"""从已验证 GUI 提取的设备流程，保留调用顺序和模式切换延时。"""

import math
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
from typing import Any

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from . import hardware_status, pairing
from .runtime import Runtime, available_channels, default_channel_state

HOME_POS = [0.0, 1.5708, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_TOLERANCE_RAD = 0.03
HOME_TIMEOUT_SECONDS = 12.0
ENABLE_TIMEOUT_SECONDS = 3.0
DISABLE_TIMEOUT_SECONDS = 3.0
DISABLE_FINAL_CHECK_SECONDS = 1.0
ALIGN_TOLERANCE_RAD = 0.1
ALIGN_STILLNESS_RAD = 0.02
ALIGN_SAMPLES = 3
ALIGN_INTERVAL_SECONDS = 0.1
ALIGN_MOVE_INTERVAL_SECONDS = 0.02
ALIGN_MIN_MOVE_SECONDS = 3.0
ALIGN_MAX_SPEED_RAD_S = 0.25
ALIGN_FEEDBACK_TIMEOUT_SECONDS = 0.5
ALIGN_SETTLE_TIMEOUT_SECONDS = 5.0
ALIGN_REACHED_RAD = 0.03


class AlignmentError(RuntimeError):
    """开录前关节对齐、静止或反馈检查未通过，不允许发送跟随指令。"""


def require_joint_alignment(target: list[float], actual: list[float], source: str = "主从关节") -> None:
    if len(target) != 7 or len(actual) != 7 or not all(math.isfinite(v) for v in target + actual):
        raise AlignmentError("关节反馈无效，无法确认主从对齐，请检查设备后重试")
    differences = [abs(a - b) for a, b in zip(target, actual, strict=True)]
    joint = max(range(7), key=differences.__getitem__)
    if differences[joint] > ALIGN_TOLERANCE_RAD:
        raise AlignmentError(
            f"{source}未对齐：J{joint + 1} 相差 {math.degrees(differences[joint]):.1f}°"
            f"（上限 {math.degrees(ALIGN_TOLERANCE_RAD):.1f}°）。请保持主臂静止，重新执行从臂对齐后录制"
        )


def _disable_feedback_snapshot(feedback: Any) -> tuple[float, bool | None]:
    """SDK 返回可变缓存；只保存本次读到的时间戳和使能位。"""
    stamp = getattr(feedback, "timestamp", 0)
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp <= 0:
        return 0.0, None
    enabled = getattr(
        getattr(getattr(feedback, "msg", None), "foc_status", None), "driver_enable_status", None
    )
    return float(stamp), enabled if isinstance(enabled, bool) else None


def _disable_and_confirm(
    send: Callable[[], Any] | None,
    read: Callable[[], list[Any]],
    *,
    timeout: float = DISABLE_TIMEOUT_SECONDS,
) -> None:
    """确认本次操作之后的新反馈；send=None 时只读取，不发送设备命令。"""
    before = [_disable_feedback_snapshot(feedback)[0] for feedback in read()]
    if not before:
        raise RuntimeError("没有可确认的反馈项")
    deadline = time.monotonic() + timeout
    if send is not None:
        send()
    next_send = time.monotonic() + 0.05
    while True:
        # 先收取上次命令的反馈，再决定是否重发；截止时也要检查最后一次等待中的反馈。
        snapshots = [_disable_feedback_snapshot(feedback) for feedback in read()]
        problems = []
        if len(snapshots) != len(before):
            problems.append("反馈项数量不完整")
        else:
            for i, ((stamp, enabled), baseline) in enumerate(zip(snapshots, before, strict=True)):
                reason = ""
                if stamp == 0:
                    reason = "未收到有效反馈"
                elif stamp <= baseline:
                    reason = "反馈时间戳未更新，当前使能状态未知"
                elif enabled is True:
                    reason = "新反馈仍显示使能"
                elif enabled is not False:
                    reason = "新反馈中的使能标志无效"
                if reason:
                    problems.append((f"J{i + 1}: " if len(before) == 7 else "") + reason)
        if not problems:
            return
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise TimeoutError("失能确认超时：" + "；".join(problems))
        if send is not None and now >= next_send:
            send()
            next_send = time.monotonic() + 0.05
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.05, remaining))


def disable_channel(runtime: Runtime, ch: str, *, confirm_only: bool = False) -> None:
    """确认关节和夹爪失能；confirm_only 只复核反馈，不发送命令。"""
    errors: dict[str, str] = {}
    result = {"joints": "unknown", "gripper": "not_present", "message": "正在确认失能"}
    runtime.channels[ch]["disable_result"] = result
    with runtime.ch_locks[ch]:
        arm = runtime.channels[ch]["arm"]
        gripper = runtime.channels[ch]["gripper"]
        # Nero 失能整臂后可能不再执行控制指令。必须趁整臂仍可受控时，
        # 先下发并确认夹爪失能，再失能关节。即使夹爪失败也继续处理关节。
        if gripper is not None:
            result["gripper"] = "unknown"
            try:
                _disable_and_confirm(
                    None if confirm_only else gripper.disable_gripper, lambda: [gripper.get_gripper_status()]
                )
                result["gripper"] = "disabled"
            except Exception as exc:
                errors["夹爪"] = str(exc)
        try:
            if arm is None:
                raise RuntimeError("机械臂未连接")
            _disable_and_confirm(
                None if confirm_only else lambda: arm.disable(timeout=0.0),
                lambda: [arm.get_driver_states(i) for i in range(1, 8)],
            )
            result["joints"] = "disabled"
        except Exception as exc:
            errors["关节"] = str(exc)
        if gripper is not None and not confirm_only:
            # 等整臂失能命令发完后再读一轮新反馈：自动接纳迟到的失能回报，
            # 也能发现夹爪被重新使能。此阶段绝不为了重试而重新使能整臂。
            try:
                _disable_and_confirm(
                    None,
                    lambda: [gripper.get_gripper_status()],
                    timeout=DISABLE_FINAL_CHECK_SECONDS,
                )
                result["gripper"] = "disabled"
                errors.pop("夹爪", None)
            except Exception as exc:
                result["gripper"] = "unknown"
                errors["夹爪"] = str(exc)
        if errors:
            result["message"] = "；".join(f"{part}: {message}" for part, message in errors.items())
            runtime.channels[ch]["status"] = "失能未确认：" + result["message"]
            raise RuntimeError(result["message"])
        runtime.channels[ch]["status"] = (
            "关节及夹爪已失能" if gripper is not None else "关节已失能（未接入夹爪）"
        )
        if confirm_only:
            runtime.channels[ch]["status"] = "新反馈已确认：" + runtime.channels[ch]["status"]
        result["message"] = runtime.channels[ch]["status"]


def _joint_positions(arm: Any, leader: bool = False) -> list[float] | None:
    feedback = arm.get_leader_joint_angles() if leader else arm.get_joint_angles()
    if feedback is None or feedback.msg is None:
        return None
    values = list(feedback.msg)
    return values if len(values) == 7 and all(math.isfinite(value) for value in values) else None


def set_control_role(runtime: Runtime, ch: str, role: str) -> None:
    """记录明确发出的角色请求；控制模式 0x06 本身不能区分主臂与从臂。"""
    if role not in ("leader", "follower"):
        raise ValueError("未知机械臂角色")
    info = runtime.channels[ch]
    info.update(requested_role=None, feedback_mode=None)
    if role == "leader":
        info["arm"].set_leader_mode()
    else:
        info["arm"].set_follower_mode()
    info.update(requested_role=role, feedback_mode=role)
    # 角色切换前的缓存不能证明切换后的运动条件；重新观察时间戳实际推进。
    with runtime.lifecycle_lock:
        runtime.feedback_seen.pop(ch, None)


def _exit_teaching_before_enable(runtime: Runtime, ch: str) -> None:
    """恢复上次会话遗留的示教模式，使普通控制状态和驱动使能反馈可用。"""
    info = runtime.channels[ch]
    status = hardware_status.inspect_status(runtime, ch)
    if status["health"]["state"] == "fault":
        raise RuntimeError(f"{ch} 设备异常：{status['health']['label']}；未执行使能或运动")
    needs_exit = (
        info.get("feedback_mode") == "leader"
        or info.get("requested_role") == "leader"
        or status["mode"]["code"] == 2
        or (status["mode"]["code"] == 6 and info.get("requested_role") != "follower")
    )
    if not needs_exit:
        return
    info["status"] = "正在退出残留示教模式，准备使能…"
    set_control_role(runtime, ch, "follower")
    time.sleep(0.2)


def _enable_failure_details(arm: Any) -> str:
    """仅用于诊断最近反馈，不把缺失反馈当作已失能，也不自动清除故障。"""
    details = []
    try:
        feedback = arm.get_arm_status()
        if feedback is None or feedback.msg is None:
            details.append("未收到机械臂控制状态反馈")
        else:
            status = feedback.msg
            details.append(
                f"控制模式={status.ctrl_mode}，机械臂状态={status.arm_status}，故障码={status.err_code}"
            )
    except Exception as exc:
        details.append(f"控制状态读取失败: {exc}")
    missing, disabled = [], []
    try:
        for joint in range(1, 8):
            feedback = arm.get_driver_states(joint)
            if feedback is None or feedback.msg is None:
                missing.append(f"J{joint}")
            elif not feedback.msg.foc_status.driver_enable_status:
                disabled.append(f"J{joint}")
        if missing:
            details.append("驱动反馈缺失=" + ",".join(missing))
        if disabled:
            details.append("未使能关节=" + ",".join(disabled))
    except Exception as exc:
        details.append(f"驱动状态读取失败: {exc}")
    return "最近反馈：" + "；".join(details)


def start_return_home(runtime: Runtime, expected_targets: dict | None) -> tuple[bool, str]:
    """将确认过的全部配对依次回位；禁止与采集和设备配置重叠。"""
    with runtime.lifecycle_lock:
        if runtime.state["recovery_required"]:
            return False, runtime.state["recovery_required"]
        if runtime.state["recording"]:
            return False, "请先结束本集并等待保存完成"
        if runtime.operation_busy(include_recording=False):
            return False, "机械臂正在执行操作，请等待完成"
        if runtime.state["disable_error"] or runtime.state["close_error"]:
            return False, "请先处理结束结果中的未完成项"
        try:
            targets = pairing.targets(pairing.resolve(runtime))
        except ValueError as exc:
            return False, str(exc)
        if expected_targets != targets:
            return False, "主从臂选择已变化，请重新确认全部操作对象"
        runtime.state.update(homing=True, home_error=None, home_status="正在检查主从臂回位条件…")
        runtime.state["disable_status"] = ""
        runtime.home_thread = threading.Thread(
            target=_return_home_worker, args=(runtime, targets), name="nero-return-home", daemon=False
        )
        try:
            runtime.home_thread.start()
        except Exception as exc:
            runtime.state.update(homing=False, home_error=str(exc), home_status=f"回位启动失败: {exc}")
            return False, runtime.state["home_status"]
        return True, "已开始返回默认位置"


def _return_one_arm_home(runtime: Runtime, ch: str) -> None:
    arm = runtime.channels[ch]["arm"]
    _exit_teaching_before_enable(runtime, ch)
    deadline = time.monotonic() + ENABLE_TIMEOUT_SECONDS
    runtime.channels[ch]["status"] = "正在使能并回位…"
    while not arm.enable():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"使能超时，未发送回位指令；{_enable_failure_details(arm)}")
        time.sleep(0.02)
    set_control_role(runtime, ch, "follower")
    time.sleep(0.2)
    hardware_status.require_ready_for_motion(runtime, ch)
    arm.move_j(HOME_POS[:])
    deadline = time.monotonic() + HOME_TIMEOUT_SECONDS
    settled = 0
    while time.monotonic() < deadline:
        positions = _joint_positions(arm)
        status = arm.get_arm_status()
        idle = status is not None and status.msg is not None and status.msg.motion_status == 0
        at_home = positions is not None and all(
            abs(actual - target) <= HOME_TOLERANCE_RAD
            for actual, target in zip(positions, HOME_POS, strict=True)
        )
        settled = settled + 1 if idle and at_home else 0
        if settled >= 3:
            runtime.channels[ch]["status"] = "已回默认位置（保持使能）"
            return
        time.sleep(0.1)
    raise TimeoutError("等待默认位置超时，请检查机械臂状态后重试回位")


def _return_home_worker(runtime: Runtime, targets: dict[str, str]) -> None:
    leaders = [ch for ch, label in targets.items() if label.startswith("主")]
    followers = [ch for ch, label in targets.items() if label.startswith("从")]
    ch = leaders[0]
    try:
        with ExitStack() as stack:
            for channel in sorted(targets):
                stack.enter_context(runtime.ch_locks[channel])
            for ch in targets:
                if (
                    _joint_positions(
                        runtime.channels[ch]["arm"],
                        leader=runtime.channels[ch].get("feedback_mode") == "leader",
                    )
                    is None
                ):
                    raise RuntimeError("读不到有效关节位置，未发送回位指令")
            runtime.state["teleop_needs_restore"] = True
            for ch in leaders:
                set_control_role(runtime, ch, "follower")
                runtime.channels[ch]["status"] = "已退出示教，等待回位"
                time.sleep(0.2)
            for ch in followers + leaders:
                runtime.state["home_status"] = f"{targets[ch]} ({ch}) 正在返回默认位置…"
                _return_one_arm_home(runtime, ch)
        runtime.state["home_error"] = None
        runtime.state["home_status"] = "主臂和从臂均已回默认位置，保持使能；再次录制时恢复示教模式"
    except Exception as exc:
        runtime.state["home_error"] = f"{targets[ch]} ({ch}): {exc}"
        runtime.state["home_status"] = f"回位失败: {runtime.state['home_error']}"
        runtime.channels[ch]["status"] = runtime.state["home_status"]
    finally:
        runtime.state["homing"] = False


def _leader_in_teaching_mode(runtime: Runtime, ch: str | None = None) -> bool:
    ch = ch or channel_by_label(runtime, runtime.state["leader_label"])
    mode = runtime.channels[ch].get("feedback_mode") if ch is not None else None
    if mode is not None:
        return mode == "leader"
    return runtime.state["connected"] and not runtime.state["teleop_needs_restore"]


def _alignment_snapshot(
    runtime: Runtime,
    leader_mode: bool | None = None,
    require_aligned: bool = True,
    pair: pairing.Pair | None = None,
) -> dict[str, tuple[list[float], float]]:
    # Nero 进入示教模式后，普通关节反馈可能停止推送并保留旧缓存。
    # 此时使用与录制动作相同的示教广播，不能再要求普通反馈刷新或与它比较。
    pair = pair or pairing.resolve(runtime)[0]
    if leader_mode is None:
        leader_mode = _leader_in_teaching_mode(runtime, pair.leader)
    snapshot = {}
    for ch, name in (
        (pair.leader, "主臂"),
        (pair.follower, "从臂"),
    ):
        info = runtime.channels[ch]
        if info["arm"] is None or not info["detected"]:
            raise AlignmentError(f"{name}尚未连接，请先连接对应通道")
        teaching = name == "主臂" and leader_mode
        source = "主臂示教" if teaching else name
        arm = info["arm"]
        feedback = arm.get_leader_joint_angles() if teaching else arm.get_joint_angles()
        if feedback is None or feedback.msg is None:
            raise AlignmentError(f"读不到{source}的有效关节位置，无法确认主从对齐")
        values = list(feedback.msg)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise AlignmentError(f"{source}关节反馈无效，无法确认主从对齐")
        snapshot[source] = (values, float(getattr(feedback, "timestamp", 0)))
    leader_source = "主臂示教" if leader_mode else "主臂"
    if require_aligned:
        require_joint_alignment(snapshot[leader_source][0], snapshot["从臂"][0], f"{leader_source}与从臂")
    return snapshot


def check_alignment(runtime: Runtime, require_aligned: bool = True) -> tuple[bool, str]:
    """开录准备前验证反馈；自动对齐流程允许两臂初始位置不同。"""
    try:
        for pair in pairing.resolve(runtime):
            try:
                _alignment_snapshot(runtime, require_aligned=require_aligned, pair=pair)
            except Exception as exc:
                raise AlignmentError(f"{pair.display}：{exc}") from exc
        return True, ""
    except Exception as exc:
        return False, f"主从对齐检查失败: {exc}"


def confirm_stable_alignment(
    runtime: Runtime, require_aligned: bool = True, pair: pairing.Pair | None = None
) -> dict[str, tuple[list[float], float]] | None:
    """准备完成后用主臂示教广播与从臂反馈复核三次，确认对齐且静止。"""
    first = previous = None
    for sample in range(ALIGN_SAMPLES):
        if runtime.state["stop_requested"]:
            return
        current = _alignment_snapshot(runtime, leader_mode=True, require_aligned=require_aligned, pair=pair)
        for source, (positions, timestamp) in current.items():
            if (
                not math.isfinite(timestamp)
                or timestamp <= 0
                or (previous and timestamp <= previous[source][1])
            ):
                raise AlignmentError(f"{source}关节反馈未更新，请检查设备后重试；尚未录制")
            if first and any(
                abs(a - b) > ALIGN_STILLNESS_RAD for a, b in zip(positions, first[source][0], strict=True)
            ):
                raise AlignmentError(f"{source}仍在移动，请保持主臂静止、等待从臂停稳后重试；尚未录制")
        if first is None:
            first = current
        previous = current
        if sample < ALIGN_SAMPLES - 1:
            time.sleep(ALIGN_INTERVAL_SECONDS)
    return previous


def align_follower_before_recording(runtime: Runtime) -> None:
    """逐对对齐，全部完成后再开始同一集；过程中不更改配置选择。"""
    pairs = pairing.resolve(runtime)
    for pair in pairs:
        if runtime.state["stop_requested"]:
            return
        try:
            _align_one_follower(runtime, pair)
        except Exception as exc:
            raise AlignmentError(f"{pair.display}：{exc}") from exc
    if len(pairs) > 1:
        for pair in pairs:
            confirm_stable_alignment(runtime, pair=pair)


def _align_one_follower(runtime: Runtime, pair: pairing.Pair) -> None:
    """固定主臂目标，以限速平滑插值移动从臂；整个过程不写入数据。"""
    runtime.state["record_message"] = "正在读取主臂当前位置，请保持主臂静止…"
    initial = confirm_stable_alignment(runtime, require_aligned=False, pair=pair)
    if initial is None or runtime.state["stop_requested"]:
        return
    follower_ch = pair.follower
    hardware_status.require_ready_for_motion(runtime, follower_ch)
    if runtime.state["stop_requested"]:
        return
    target, _ = initial["主臂示教"]
    origin, _ = initial["从臂"]
    distance = max(abs(a - b) for a, b in zip(target, origin, strict=True))
    if distance <= ALIGN_REACHED_RAD:
        return
    follower = runtime.channels[follower_ch]["arm"]
    # smoothstep 最大导数为 1.5；偏差大时延长时间，限制每关节指令速度。
    duration = max(ALIGN_MIN_MOVE_SECONDS, 1.5 * distance / ALIGN_MAX_SPEED_RAD_S)
    steps = math.ceil(duration / ALIGN_MOVE_INTERVAL_SECONDS)
    latest = initial
    stamps = {source: (stamp, time.monotonic()) for source, (_, stamp) in initial.items()}
    moved = False
    completed = False

    def observe() -> dict[str, tuple[list[float], float]]:
        current = _alignment_snapshot(runtime, leader_mode=True, require_aligned=False, pair=pair)
        now = time.monotonic()
        for source, (_, stamp) in current.items():
            old_stamp, updated = stamps[source]
            if not math.isfinite(stamp) or stamp <= 0 or stamp < old_stamp:
                raise AlignmentError(f"{source}关节反馈无效，已中止对齐；尚未录制")
            if stamp > old_stamp:
                stamps[source] = (stamp, now)
            elif now - updated >= ALIGN_FEEDBACK_TIMEOUT_SECONDS:
                raise AlignmentError(f"{source}关节反馈未更新，已中止对齐；尚未录制")
        if max(abs(a - b) for a, b in zip(current["主臂示教"][0], target, strict=True)) > ALIGN_STILLNESS_RAD:
            raise AlignmentError("对齐期间主臂位置发生变化，请保持主臂静止后重试；尚未录制")
        return current

    try:
        runtime.state.update(record_phase="aligning", record_message="从臂正在平缓对齐主臂，请保持主臂静止…")
        for step in range(1, steps + 1):
            if runtime.state["stop_requested"]:
                return
            latest = observe()
            alpha = step / steps
            blend = alpha * alpha * (3.0 - 2.0 * alpha)
            command = [a + blend * (b - a) for a, b in zip(origin, target, strict=True)]
            if max(abs(a - b) for a, b in zip(command, latest["从臂"][0], strict=True)) > ALIGN_TOLERANCE_RAD:
                raise AlignmentError("从臂未能跟上对齐动作，已中止；请检查机械臂后重试，尚未录制")
            moved = True
            follower.move_js(command)
            runtime.state["record_message"] = (
                f"{pair.display} 从臂正在平缓对齐：{step * 100 // steps}%（尚未录制）"
            )
            time.sleep(ALIGN_MOVE_INTERVAL_SECONDS)
        runtime.state["record_message"] = "正在确认从臂到位并停稳（尚未录制）…"
        deadline = time.monotonic() + ALIGN_SETTLE_TIMEOUT_SECONDS
        settled = 0
        previous = None
        while time.monotonic() < deadline:
            if runtime.state["stop_requested"]:
                return
            latest = observe()
            actual = latest["从臂"][0]
            reached = max(abs(a - b) for a, b in zip(actual, target, strict=True)) <= ALIGN_REACHED_RAD
            stable = previous is not None and all(
                latest[source][1] > previous[source][1]
                and max(abs(a - b) for a, b in zip(values, previous[source][0], strict=True))
                <= ALIGN_STILLNESS_RAD
                for source, (values, _) in latest.items()
            )
            settled = settled + 1 if reached and stable else 0
            if settled >= ALIGN_SAMPLES:
                completed = True
                return
            previous = latest
            time.sleep(ALIGN_INTERVAL_SECONDS)
        raise AlignmentError("等待从臂对齐到位超时，尚未录制；请检查机械臂后重试")
    finally:
        if moved and not completed:
            # 中断渐进指令后保持当前反馈位置，避免继续追赶最后一个目标。
            try:
                feedback = follower.get_joint_angles()
                values = list(feedback.msg) if feedback is not None and feedback.msg is not None else []
                stamp = float(getattr(feedback, "timestamp", 0))
                fresh = math.isfinite(stamp) and (
                    stamp > stamps["从臂"][0]
                    or (
                        stamp == stamps["从臂"][0]
                        and time.monotonic() - stamps["从臂"][1] < ALIGN_FEEDBACK_TIMEOUT_SECONDS
                    )
                )
                if not fresh or len(values) != 7 or not all(math.isfinite(v) for v in values):
                    raise AlignmentError("无法读取从臂位置以保持")
                follower.move_js(values)
            except Exception as exc:
                message = f"{follower_ch} 对齐中止时无法保持从臂，已尝试阻尼急停"
                try:
                    follower.electronic_emergency_stop()
                except Exception as stop_error:
                    message += f"，急停指令发送失败：{stop_error}"
                message += (
                    "。运动操作已锁定；请先结束采集，托稳机械臂和夹持物，读取设备状态。"
                    "若需失能请单独确认；按设备说明排查并解除急停或故障后，重启平台重新检查状态。"
                    "平台不会自动复位或再次使能，急停是否生效以实际反馈为准。"
                )
                runtime.state["recovery_required"] = message
                raise AlignmentError(message) from exc


def hold_planned_followers(
    runtime: Runtime, attempted: dict[str, tuple[float, float, bool] | None], now: float
) -> str | None:
    """J 跟随退出时替换尚未完成的目标；尝试过下发的每条从臂都单独收尾。"""
    errors = []
    for ch, seen in attempted.items():
        arm = runtime.channels[ch]["arm"]
        try:
            feedback = arm.get_joint_angles()
            positions = list(feedback.msg)
            stamp = float(feedback.timestamp)
            fresh = (
                seen is not None
                and math.isfinite(stamp)
                and stamp > 0
                and (
                    stamp > seen[0]
                    or (stamp == seen[0] and 0 <= now - seen[1] < ALIGN_FEEDBACK_TIMEOUT_SECONDS)
                )
            )
            if not fresh or len(positions) != 7 or not all(math.isfinite(v) for v in positions):
                raise ValueError("从臂关节反馈缺失、过期或无效，不能用旧目标保持")
            arm.move_j(positions)
        except Exception as exc:
            message = f"{ch} 结束底层平滑跟随时无法保持当前反馈位置：{exc}；已尝试阻尼急停"
            try:
                arm.electronic_emergency_stop()
            except Exception as stop_error:
                message += f"，急停指令发送失败：{stop_error}"
            errors.append(message)
    if errors:
        message = "；".join(errors) + (
            "。运动操作已锁定；请先结束采集并检查设备状态，按设备说明解除急停或故障。"
            "处理后重启平台重新检查状态。平台不自动复位或失能；急停是否生效以实际反馈为准。"
        )
        with runtime.lifecycle_lock:
            runtime.state.update(recovery_required=message, follow_stop_message=message)
        return message
    if attempted:
        runtime.state["follow_stop_message"] = "已请求从臂保持当前反馈位置，停止追赶上一目标；保持使能"
    return None


def _detect_feedback_mode(arm: Any, timeout: float = 1.0) -> str | None:
    """只读两次更新的完整角度，识别重启后仍保留的示教/普通反馈。"""
    stamps = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for mode, reader in (("leader", arm.get_leader_joint_angles), ("follower", arm.get_joint_angles)):
            feedback = reader()
            if feedback is None or feedback.msg is None:
                continue
            values = list(feedback.msg)
            stamp = float(getattr(feedback, "timestamp", 0))
            if (
                len(values) != 7
                or not all(math.isfinite(v) for v in values)
                or not math.isfinite(stamp)
                or stamp <= 0
            ):
                continue
            if mode in stamps and stamp > stamps[mode]:
                return mode
            stamps.setdefault(mode, stamp)
        time.sleep(0.05)
    return None


def redetect_channels(runtime: Runtime) -> list[str]:
    """重新扫描 CAN 接口并重建通道表；已连接的臂先断开（释放 CAN 句柄）。"""
    for ch in list(runtime.channels):
        arm = runtime.channels[ch].get("arm")
        if arm is not None:
            with suppress(Exception):
                arm.disconnect()
    found = available_channels()
    runtime.channels = {ch: default_channel_state() for ch in found}
    runtime.ch_locks = {ch: threading.Lock() for ch in found}
    return found


def connect_channel(runtime: Runtime, ch: str) -> None:
    """建立通信不改变力矩或运动模式；固件查询失败时以新反馈确认连接。"""
    info = runtime.channels[ch]
    info.update(status="正在连接并读取反馈…", detected=False, feedback_mode=None, requested_role=None)
    arm = None
    try:
        previous = info["arm"]
        if previous is not None:
            previous.disconnect()
        info.update(arm=None, gripper=None)
        arm = AgxArmFactory.create_arm(
            create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121, channel=ch)
        )
        try:
            info["gripper"] = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception:
            info["gripper"] = None
        arm.connect()
        info["arm"] = arm
        firmware_error = "固件查询未响应"
        try:
            fw = arm.get_firmware(timeout=1.0)
        except Exception as exc:
            fw = None
            firmware_error = f"固件查询失败: {exc}"
        mode = _detect_feedback_mode(arm)
        info["feedback_mode"] = mode
        feedback_text = "主臂示教反馈正常" if mode == "leader" else "普通关节反馈正常"
        if fw and fw.get("software_version"):
            info["status"] = f"已连接，固件 {fw['software_version']}" + (
                f"；{feedback_text}" if mode else "；等待关节反馈"
            )
            info["detected"] = True
        elif mode is not None:
            info.update(status=f"已连接，{feedback_text}（{firmware_error}）", detected=True)
        else:
            info["status"] = f"通信未确认：{firmware_error}，也未收到更新的关节反馈，请检查供电和 CAN 状态"
    except Exception as e:
        if arm is not None:
            with suppress(Exception):
                arm.disconnect()
            info.update(arm=None, gripper=None)
        info.update(status=f"连接失败 {e}", detected=False, feedback_mode=None)


def wait_motion_done(arm: Any, timeout: float = 12.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = arm.get_arm_status()
        if st is not None and st.msg.motion_status == 0:
            return True
        time.sleep(0.1)
    return False


def micro_move(runtime: Runtime, ch: str) -> None:
    arm = runtime.channels[ch]["arm"]
    if arm is None:
        runtime.channels[ch]["status"] = "未连接，先点连接"
        return
    with runtime.ch_locks[ch]:
        try:
            _exit_teaching_before_enable(runtime, ch)
            runtime.channels[ch]["status"] = "正在使能并微动…"
            deadline = time.monotonic() + ENABLE_TIMEOUT_SECONDS
            while not arm.enable():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"使能超时，未执行微动；{_enable_failure_details(arm)}")
                time.sleep(0.01)
            hardware_status.require_ready_for_motion(runtime, ch)
            ja = arm.get_joint_angles()
            if ja is None or ja.msg is None:
                runtime.channels[ch]["status"] = "读不到关节角"
                return
            angles = list(ja.msg)
            moved = angles[:]
            moved[6] += 0.0873
            runtime.channels[ch]["status"] = "微动中(末端关节转动)..."
            arm.move_j(moved)
            time.sleep(1.6)
            arm.move_j(angles)
            time.sleep(1.6)
            runtime.channels[ch]["status"] = "微动完成(已回原位，保持使能)"
        except Exception as e:
            runtime.channels[ch]["status"] = f"微动失败 {e}"


def home_and_disable(runtime: Runtime, ch: str) -> None:
    arm = runtime.channels[ch]["arm"]
    if arm is None:
        runtime.channels[ch]["status"] = "未连接"
        return
    try:
        with runtime.ch_locks[ch]:
            runtime.channels[ch]["status"] = "正在回位，到位后失能…"
            _return_one_arm_home(runtime, ch)
        # disable_channel 自己持有通道锁；整个接口操作仍由 device_busy 保护。
        disable_channel(runtime, ch)
    except Exception as e:
        runtime.channels[ch]["status"] = f"回位后失能未完成：{e}"


def arm_count(runtime: Runtime) -> int:
    """通过固件响应或更新的关节反馈确认通信的臂数量。"""
    return sum(1 for ch in runtime.channels if runtime.channels[ch]["detected"])


def channel_by_label(runtime: Runtime, label: str) -> str | None:
    for ch in runtime.channels:
        if runtime.channels[ch]["label"] == label:
            return ch
    return None


def transition_to_collection(runtime: Runtime) -> list[pairing.Pair]:
    pairs = pairing.resolve(runtime)
    # 在第一条臂使能前确认所有配对存在，避免缺少右侧时只启动左侧。
    for pair in pairs:
        _transition_one_pair(runtime, pair)
    return pairs


def _set_gripper_teaching(gripper: Any) -> bool:
    """设置示教参数；SDK 未确认时以新的参数读回为准，返回是否用了读回兜底。"""
    expected = {"teaching_range_per": 100, "max_range_config": 0.07, "teaching_friction": 8}
    if gripper.set_gripper_teaching_pendant_param(**expected) is True:
        return False
    # SDK 要求设置 ACK + 参数一致；没有 ACK 时甚至不会发出参数查询。
    # 主动查询：第一次取走并清空可能残留的参数缓存，第二次等待新的反馈。
    read_params = getattr(gripper, "get_gripper_teaching_pendant_param", None)
    if not callable(read_params):
        raise RuntimeError("夹爪示教参数未确认：SDK 不支持参数读回，未进入主臂示教模式")
    previous = read_params(timeout=0.0, min_interval=0.0)
    baseline = getattr(previous, "timestamp", 0)
    if type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 0:
        baseline = 0
    current = read_params(timeout=1.0, min_interval=0.0)
    stamp = getattr(current, "timestamp", 0)
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= baseline:
        raise RuntimeError("夹爪示教参数未确认：未收到新的有效参数反馈，未进入主臂示教模式")
    message = getattr(current, "msg", None)
    actual = {key: getattr(message, key, None) for key in expected}
    if any(
        type(actual[key]) not in (int, float)
        or not math.isfinite(actual[key])
        or not math.isclose(actual[key], value, rel_tol=0.0, abs_tol=1e-6)
        for key, value in expected.items()
    ):
        raise RuntimeError(
            "夹爪示教参数未确认：读回值不匹配"
            f"（比例={actual['teaching_range_per']}%，开口={actual['max_range_config']} m，"
            f"摩擦={actual['teaching_friction']}；期望 100%、0.07 m、8），未进入主臂示教模式"
        )
    return True


def _prepare_leader_gripper(runtime: Runtime, ch: str) -> None:
    """仅释放主臂夹爪的残留控制，再确认示教参数；不失能关节。"""
    info = runtime.channels[ch]
    gripper = info["gripper"]
    if gripper is None:
        return
    try:
        # 必须在整臂使能、尚未进入 leader 模式时操作，普通夹爪反馈才可用于确认。
        # Ctrl+C/回位保持上电，不能假定重新打开程序就清除了夹爪的位置保持状态。
        info["status"] = "正在释放主臂夹爪残留控制，关节保持使能…"
        _disable_and_confirm(gripper.disable_gripper, lambda: [gripper.get_gripper_status()])
        time.sleep(0.2)
        info["status"] = "正在设置并确认主臂夹爪示教参数…"
        if _set_gripper_teaching(gripper):
            print(f"[采集] {info['label']} ({ch}) 夹爪：SDK 未确认，已用新参数反馈核对 100% / 70 mm / 摩擦 8")
    except Exception as exc:
        info["status"] = f"{info['label']} ({ch}) 夹爪准备失败：{exc}；关节保持使能，本次采集未启动"
        raise RuntimeError(info["status"]) from exc


def _transition_one_pair(runtime: Runtime, pair: pairing.Pair) -> None:
    """切到采集模式；不失能关节，仅在主臂进入示教前释放夹爪残留控制。"""
    leader_ch, follower_ch = pair.leader, pair.follower
    leader_arm = runtime.channels[leader_ch]["arm"]
    follower_arm = runtime.channels[follower_ch]["arm"]
    if leader_arm is None or follower_arm is None:
        raise ValueError("主臂/从臂未连接，先在左边点连接")
    _exit_teaching_before_enable(runtime, leader_ch)
    runtime.channels[leader_ch]["status"] = "正在使能主臂…"
    deadline = time.monotonic() + ENABLE_TIMEOUT_SECONDS
    while not leader_arm.enable():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"主臂使能超时；{_enable_failure_details(leader_arm)}")
        time.sleep(0.01)
    hardware_status.require_ready_for_motion(runtime, leader_ch)
    time.sleep(0.3)
    _prepare_leader_gripper(runtime, leader_ch)
    time.sleep(0.2)
    set_control_role(runtime, leader_ch, "leader")
    runtime.channels[leader_ch]["status"] = "主臂已使能（示教模式）"
    time.sleep(0.2)
    _exit_teaching_before_enable(runtime, follower_ch)
    runtime.channels[follower_ch]["status"] = "正在使能从臂…"
    deadline = time.monotonic() + ENABLE_TIMEOUT_SECONDS
    while not follower_arm.enable():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"从臂使能超时；{_enable_failure_details(follower_arm)}")
        time.sleep(0.01)
    time.sleep(0.3)
    set_control_role(runtime, follower_ch, "follower")
    runtime.channels[follower_ch]["status"] = "已请求从臂受控模式，正在确认反馈…"
    time.sleep(0.2)
    hardware_status.require_ready_for_motion(runtime, follower_ch)
    runtime.channels[follower_ch]["status"] = "从臂已使能（跟随模式）"
    print(f"[采集] {pair.display}（{leader_ch} → {follower_ch}）已切到采集模式")
