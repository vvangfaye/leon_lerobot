"""Original VLA manual follow with asynchronous, absolute-action RL recording."""

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from examples.nero_to_nero.nero_platform import devices, pairing, sampling

from .active_master import ActiveMasterSync
from .config import JOINT_LIMITS, TaskConfig
from .environment import NeroEnv
from .hardware import NeroHardware, Snapshot
from .learning import Trainer, load_episodes
from .master_input import MasterInput, WorkbenchGuard
from .native_recording import CameraCache, NativeRecorder, NativeTick
from .native_takeover import NativeTakeover, engage_takeover
from .web_control import WebControls


def policy_start_rejection(snap: Snapshot, env: NeroEnv) -> str | None:
    """Explain the autonomous start gate in the follower's base coordinates."""
    xyz = snap.pose[:3, 3]
    if not np.isfinite(xyz).all() or not np.isfinite(snap.joints).all():
        return "当前姿态反馈包含无效数值，未启动策略。"
    reasons = []
    spatial_axes = zip("XYZ", xyz, env.lo, env.hi, strict=True) if env.cfg.policy_workspace_enabled else ()
    for axis, value, lo, hi in spatial_axes:
        if value < lo:
            reasons.append(
                f"{axis}={value * 100:.1f} cm，低于下界 {lo * 100:.1f} cm，需增大至少 {(lo - value) * 100:.1f} cm"
            )
        elif value > hi:
            reasons.append(
                f"{axis}={value * 100:.1f} cm，高于上界 {hi * 100:.1f} cm，需减小至少 {(value - hi) * 100:.1f} cm"
            )
    for index, (value, (lo, hi)) in enumerate(zip(snap.joints, JOINT_LIMITS, strict=True), 1):
        if not lo <= value <= hi:
            reasons.append(
                f"关节 {index}={np.rad2deg(value):.1f}°，允许 {np.rad2deg(lo):.1f}°～{np.rad2deg(hi):.1f}°"
            )
    if reasons:
        return (
            "未启动策略（右从臂底座坐标）："
            + "；".join(reasons)
            + "。请用主臂调整后重新点击策略执行；人工跟随仍可使用。"
        )
    return None


def prepare_current_pose_probe(backend: NeroHardware, env: NeroEnv, cameras: CameraCache) -> Snapshot:
    """Prepare only the follower at its measured pose; never align to the master."""
    snap = backend.read(images=False)
    rejection = policy_start_rejection(snap, env)
    if rejection:
        raise ValueError(rejection)
    if not cameras.ready(env.cfg.cameras):
        raise ValueError("相机尚未就绪，未启动测试。")
    if not backend.prepared:
        backend.prepare()
    held = backend.hold()
    rejection = policy_start_rejection(held, env)
    if rejection:
        raise ValueError(rejection)
    return held


def policy_observation(
    env: NeroEnv, snap: Snapshot, capture: tuple[float, dict, dict]
) -> dict[str, np.ndarray]:
    """Use exactly the timestamped images that are submitted to the recorder."""
    timestamp, frames, stamps = capture
    if set(frames) != set(env.cfg.cameras) or set(stamps) != set(frames):
        raise ValueError("策略相机不完整，未发送动作。")
    if any(not -0.001 <= timestamp - stamp <= 0.5 for stamp in stamps.values()):
        raise ValueError("策略相机画面已过期，未发送动作。")
    return env.observation(Snapshot(snap.joints, snap.width, snap.pose, frames))


def send_native(
    backend: NeroHardware, readings: dict, takeover: NativeTakeover | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Use exactly the sender used by VLA recording, without an RL motion filter."""
    requested = sampling.native_action(readings) if takeover is None else takeover.action(readings)
    sampling.send_js_action(backend.runtime.channels["can1"], requested, backend.cfg.gripper_force_n)
    executed = backend.native_executed_target(requested)
    backend.last_servo_target = executed[:7].copy()
    backend.last_servo_width = float(executed[7])
    return executed, np.asarray(requested, dtype=float)


def engage_native(
    backend: NeroHardware, master: MasterInput, keys: Any
) -> tuple[sampling.RobotFeedback, np.ndarray] | None:
    """Reuse VLA startup alignment before sending the first absolute master goal."""
    runtime = backend.runtime
    runtime.channels["can1"].update(label="从右", detected=True)
    runtime.channels["can3"] = master.backend.runtime.channels["can3"]
    runtime.channels["can3"].update(label="主右", detected=True)
    runtime.state["stop_requested"] = False
    cancel = getattr(keys, "cancel", keys.abort)
    cancel.clear()
    watch_done = threading.Event()

    def watch_cancel() -> None:
        while not watch_done.wait(0.01):
            if cancel.is_set() or keys.abort.is_set():
                runtime.state["stop_requested"] = True

    watcher = threading.Thread(target=watch_cancel, name="nero-align-cancel", daemon=True)
    watcher.start()
    try:
        pair = pairing.Pair("right", "主右", "从右", "can3", "can1")
        if not backend.prepared or not master.prepared:
            master.changed_mode = True
            devices._transition_one_pair(runtime, pair)
            backend.prepared = master.prepared = True
            backend.last_servo_target = backend.last_servo_width = None
        held = backend.hold()
        if runtime.state["stop_requested"]:
            return None
        aligned_target = devices._align_one_follower(runtime, pair)
        if runtime.state["stop_requested"]:
            return None
        feedback = sampling.RobotFeedback(runtime.channels["can3"], runtime.channels["can1"])
        deadline = time.monotonic() + 1
        while not feedback.ready:
            readings = feedback.read(time.monotonic())
            if time.monotonic() >= deadline or cancel.is_set() or keys.abort.is_set():
                raise TimeoutError("Native master/follower feedback did not become ready")
            if not feedback.ready:
                time.sleep(0.01)
        devices.require_joint_alignment(readings["主臂关节"], readings["从臂关节"], "原 VLA 首帧主从对齐")
        previous = np.r_[held.joints if aligned_target is None else aligned_target, held.width]
        return feedback, previous
    finally:
        watch_done.set()
        watcher.join(timeout=0.2)


def run_native(
    args: Any,
    cfg: TaskConfig,
    *,
    keyboard_factory: Callable,
    episode_writer: Callable,
) -> None:
    cfg.validate(live=not args.mock)
    if args.human_input != "master":
        raise ValueError("Native VLA follow requires --human-input master")
    backend, master = NeroHardware(cfg), MasterInput(cfg)
    master_sync = ActiveMasterSync(master, cfg) if getattr(args, "active_master_sync", False) else None
    guard = WorkbenchGuard(args.workbench_url)
    trainer = Trainer(cfg, mock=args.mock) if args.mode != "record" else None
    if args.mode == "eval" and not args.resume:
        raise ValueError("Evaluation requires --resume")
    if trainer:
        for root in sorted({root.resolve() for root in [args.output / "episodes", *args.data]}):
            if root.exists():
                print(f"Loaded {load_episodes(root, trainer.replay, cfg, mock=args.mock)} transitions")
        if args.resume:
            trainer.load(args.resume)
        elif (args.output / "checkpoint.pt").exists():
            raise ValueError("Specify --resume or a new output directory")
    cameras = recorder = None
    active = following = False
    mode, episode, started = "human", 0, 0.0
    policy_started = 0.0
    policy_burst = getattr(args, "policy_burst_seconds", None)
    metrics_path = args.output / "metrics.jsonl"
    history = (
        [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if metrics_path.exists()
        else []
    )
    saved_episodes = len(history)
    successful_episodes = sum(bool(row["success"]) for row in history)
    previous_command: np.ndarray | None = None
    takeover: NativeTakeover | None = None
    robot_feedback: dict = {}
    policy_details: dict = {}
    feedback = None
    control_times: deque[float] = deque(maxlen=200)
    env = NeroEnv(cfg, backend, mock=args.mock)
    controls = (
        WebControls(args.web_host, args.web_port, args.output / "control_url.txt", args.web_public_host)
        if args.web
        else keyboard_factory()
    )
    with controls as keys:
        backend.abort_requested = master.abort_requested = keys.abort.is_set
        if master_sync:
            master_sync.cancel_requested = lambda: keys.abort.is_set() or keys.cancel.is_set()
        message = (
            "主动主臂同步已开启。先用相对接管准备任务并开始本集，再暂停并同步主臂，随后执行策略。"
            if master_sync
            else "点击主臂跟随；先按原 VLA 流程对齐，请保持主臂不动。"
        )

        def publish(phase: str | None = None) -> None:
            if hasattr(keys, "publish"):
                hz = (
                    (len(control_times) - 1) / (control_times[-1] - control_times[0])
                    if len(control_times) > 1
                    else 0
                )
                keys.publish(
                    phase=phase
                    or (
                        "master_syncing"
                        if master_sync and master_sync.aligning
                        else mode
                        if active and following
                        else "paused"
                        if active
                        else "waiting"
                    ),
                    message=message,
                    task=cfg.task,
                    episode=saved_episodes + 1,
                    saved_episodes=saved_episodes,
                    successful_episodes=successful_episodes,
                    failed_episodes=saved_episodes - successful_episodes,
                    mode=args.mode,
                    steps=len(recorder.rows) if recorder and active else 0,
                    training_updates=trainer.updates if trainer else 0,
                    control_hz_target=cfg.hz,
                    control_hz_measured=round(hz, 1) if following else 0,
                    action_mode=cfg.action_mode,
                    human_follow="vla_native",
                    manual_control="relative_takeover" if takeover is not None else "absolute_vla",
                    policy_burst_seconds=policy_burst,
                    policy_workspace_enabled=cfg.policy_workspace_enabled,
                    checkpoint_name=args.resume.name if args.resume else None,
                    robot_feedback=robot_feedback,
                    policy_details=policy_details,
                    active_master_sync=master_sync is not None,
                    master_sync=master_sync.diagnostics if master_sync else {},
                    frames=backend.last_frames,
                )

        def remember_feedback(snap: Snapshot) -> None:
            robot_feedback.update(
                joints_deg=np.rad2deg(snap.joints).tolist(),
                tcp_m=snap.pose[:3, 3].tolist(),
                gripper_mm=snap.width * 1000,
                host_time=time.perf_counter(),
            )

        def tick(
            joints: np.ndarray,
            width: float,
            command: np.ndarray,
            requested: np.ndarray,
            source: str,
            info: dict | None = None,
            capture: tuple | None = None,
        ) -> NativeTick:
            if capture is None:
                frames, stamps = cameras.latest
                timestamp = time.perf_counter()
            else:
                timestamp, frames, stamps = capture
            return NativeTick(
                timestamp,
                joints.copy(),
                width,
                previous_command.copy(),
                command.copy(),
                requested.copy(),
                source,
                frames,
                stamps,
                info or {},
            )

        def hold_boundary() -> NativeTick | None:
            nonlocal following, previous_command
            following = False
            control_times.clear()
            try:
                held = backend.hold() if backend.prepared else None
            finally:
                if master_sync:
                    master_sync.stop()
            if held is None:
                return None
            remember_feedback(held)
            boundary = None
            if previous_command is not None:
                boundary = tick(held.joints, held.width, previous_command, previous_command, "boundary")
            previous_command = np.r_[held.joints, backend.last_servo_width]
            return boundary

        def finish(outcome: str | None, *, truncated: bool, train: bool) -> None:
            nonlocal active, episode, message, saved_episodes, successful_episodes
            boundary = hold_boundary()
            rows = recorder.finish(boundary, outcome, truncated=truncated)
            active = False
            if rows:
                message = (
                    "正在保存测试结果；从臂保持当前位置。"
                    if args.mode == "eval"
                    else "正在保存示范；从臂保持当前位置。"
                    if args.mode == "record"
                    else "正在保存示范并更新模型；从臂保持当前位置。"
                )
                publish("updating")
                path = episode_writer(args, cfg, rows)
                saved_episodes += 1
                successful_episodes += int(rows[-1]["reward"] == 1)
                message = f"已保存 {len(rows)} 条实际指令记录。"
                print(f"Saved {path}; success={rows[-1]['reward'] == 1}", flush=True)
                if recorder.error:
                    message += f"记录曾中断，本集按截断保存：{recorder.error}。"
                if trainer and args.mode == "train" and train:
                    for row in rows:
                        trainer.replay.add(row)
                    publish("updating")
                    trainer.update(cfg.updates_per_episode)
                    trainer.save(args.output / "checkpoint.pt")
            else:
                message = "本集没有可保存的有效记录。"
            episode += 1
            publish()

        try:
            publish("starting")
            guard.check()
            backend.open()
            master.open()
            initial = backend.read(images=False)
            remember_feedback(initial)
            initial_report = {
                "read_only": True,
                "joints_rad": initial.joints.tolist(),
                "gripper_m": initial.width,
                "tcp_m": initial.pose[:3, 3].tolist(),
                "manual_workspace_gate": False,
                "status": backend.status_api.inspect_status(backend.runtime, cfg.channel),
            }
            (args.output / "initial_feedback.json").write_text(
                json.dumps(initial_report, ensure_ascii=False, indent=2) + "\n"
            )
            if master_sync:
                try:
                    master_initial = master_sync._current_master()
                    master_report = {
                        "read_only": True,
                        "joints_deg": np.rad2deg(master_initial.joints).tolist(),
                        "gripper_mm": master_initial.width * 1000,
                        "pair_gap_deg": float(
                            np.rad2deg(np.max(np.abs(master_initial.joints - initial.joints)))
                        ),
                    }
                    master_sync.diagnostics.update(phase="not_prepared", **master_report)
                except Exception as exc:
                    master_report = {"read_only": True, "error": str(exc)}
                (args.output / "master_initial_feedback.json").write_text(
                    json.dumps(master_report, ensure_ascii=False, indent=2) + "\n"
                )
            cameras = CameraCache(backend)
            cameras.start()
            recorder = NativeRecorder(cfg, backend, mock=args.mock)
            last_publish = last_guard = 0.0
            next_control = time.monotonic()
            publish()
            # The web console remains available after a collection target, so
            # saving the last episode cannot silently close the robot controls.
            # Terminal batch runs retain their requested episode limit.
            while args.web or episode < args.episodes:
                now = time.monotonic()
                if now - last_guard >= 0.5:
                    guard.check()
                    last_guard = now
                events = keys.poll()
                if "q" in events:
                    print("Operator requested session close (q); saving any unfinished episode.", flush=True)
                    if active:
                        finish(None, truncated=True, train=False)
                    return
                if "p" in events:
                    boundary = hold_boundary()
                    if active:
                        recorder.segment(boundary)
                    message = "跟随已暂停；点击相对接管可从当前位置接管，无需手动对齐。"
                    events = [key for key in events if key not in ("i", "h", "n", "a", "b", "s", "k")]
                if "k" in events and master_sync:
                    boundary = hold_boundary()
                    if active:
                        recorder.segment(boundary)
                    keys.cancel.clear()
                    try:
                        master_sync.prepare_hold(backend.read(images=False))
                        message = "主臂已使能并原位保持，尚未开始同步。确认能稳定保持后，松开主臂再点同步。"
                    except Exception as exc:
                        hold_boundary()
                        message = f"主臂原位保持未完成：{exc}"
                    publish()
                    events = [key for key in events if key not in ("i", "h", "n", "a", "b", "s")]
                if "s" in events and master_sync:
                    boundary = hold_boundary()
                    if active:
                        recorder.segment(boundary)
                    keys.cancel.clear()
                    try:
                        master_sync.begin(backend.read(images=False), align=True)
                        message = "正在慢速同步主臂；从臂保持不动，请勿扶住主臂。暂停或接管可中断。"
                    except Exception as exc:
                        hold_boundary()
                        message = f"主臂同步未开始：{exc}"
                    publish()
                    events = [key for key in events if key not in ("i", "h", "n", "a", "b")]
                if "b" in events:
                    if args.mode != "eval" or active or following or policy_burst is None:
                        message = "仅可在独立测试空闲时，从当前位置开始限时策略测试。"
                    else:
                        try:
                            held = prepare_current_pose_probe(backend, env, cameras)
                            if master_sync:
                                keys.cancel.clear()
                                master_sync.begin(held, align=False)
                            remember_feedback(held)
                            previous_command = np.r_[held.joints, backend.last_servo_width]
                            env.command_joints = held.joints.copy()
                            env.command_width = float(previous_command[7])
                            env.finished, env.steps = False, 0
                            recorder.start()
                            active, started = True, time.monotonic()
                            following, mode = True, "policy"
                            policy_started = time.monotonic()
                            next_control = policy_started
                            control_times.clear()
                            message = f"从当前实测位置执行策略，{policy_burst:g} 秒后自动保持；未与主臂对齐。"
                        except Exception as exc:
                            hold_boundary()
                            message = f"未启动当前位置测试：{exc}"
                        publish()
                    events = [key for key in events if key not in ("i", "h", "n", "a")]
                begin = "n" in events and not active
                if "i" in events or "h" in events or begin:
                    if not following or mode != "human" or "h" in events:
                        boundary = hold_boundary()
                        if active:
                            recorder.segment(boundary)
                        use_takeover = (
                            "h" in events
                            or mode == "policy"
                            or master.prepared
                            or (master_sync is not None and master_sync.drive_ready)
                        )
                        message = (
                            "保持从臂当前位置，正在记录接管基准；请短暂保持主臂不动。"
                            if use_takeover
                            else "按原 VLA 流程对齐中，请保持主臂不动；暂停可取消。"
                        )
                        publish("aligning")
                        try:
                            if master_sync:
                                master_sync.yield_to_human()
                            if use_takeover:
                                prepared = engage_takeover(backend, master, keys)
                            else:
                                native = engage_native(backend, master, keys)
                                prepared = None if native is None else (*native, None)
                        except Exception as exc:
                            hold_boundary()
                            message = f"未开始跟随：{exc}"
                            prepared = None
                        if prepared is not None:
                            feedback, previous_command, takeover = prepared
                            following, mode = True, "human"
                            control_times.clear()
                            next_control = time.monotonic()
                            message = (
                                "相对接管已就绪，从臂保持接管位置；现在可移动主臂纠正。"
                                if takeover is not None
                                else "原 VLA 主臂跟随中；未录制，点击开始本集采集示范。"
                            )
                            if takeover is not None and not active:
                                message += "当前未录制，点击开始本集可记录。"
                        else:
                            begin = False
                            if getattr(keys, "cancel", keys.abort).is_set():
                                message = "人工控制准备已取消；可重新点击主臂跟随。"
                    if begin and following:
                        if not cameras.ready(cfg.cameras):
                            message = "相机尚未就绪；原 VLA 跟随可继续，图像就绪后再开始本集。"
                        else:
                            recorder.start()
                            active, started = True, time.monotonic()
                            message = "原 VLA 跟随示范录制中；完成放入、松开和撤回后确认成功。"
                if "a" in events:
                    if not active or trainer is None:
                        message = "先开始一集，再选择策略执行。"
                    elif args.mode == "train" and not any(row["reward"] > 0 for row in trainer.replay.demos):
                        message = "请先录制至少一条成功示范。"
                    else:
                        snap = backend.read()
                        rejection = policy_start_rejection(snap, env)
                        if rejection is not None:
                            message = rejection
                            print(message, flush=True)
                        else:
                            recorder.segment(hold_boundary())
                            if master_sync:
                                keys.cancel.clear()
                                try:
                                    master_sync.begin(backend.read(images=False), align=False)
                                except Exception as exc:
                                    hold_boundary()
                                    message = f"策略未开始：{exc}"
                                    publish()
                                    continue
                            env.command_joints = previous_command[:7].copy()
                            env.command_width = float(previous_command[7])
                            env.finished, env.steps = False, 0
                            following, mode = True, "policy"
                            policy_started = time.monotonic()
                            message = "策略执行中，使用自主动作约束；主臂跟随按钮可人工接管。"
                            if policy_burst is not None:
                                message += f"本次自主试动 {policy_burst:g} 秒后自动保持位置。"
                if (
                    following
                    and mode == "policy"
                    and policy_burst is not None
                    and time.monotonic() - policy_started >= policy_burst
                ):
                    boundary = hold_boundary()
                    if active:
                        recorder.segment(boundary)
                    message = (
                        f"本次 {policy_burst:g} 秒自主试动已结束，已保持当前位置；可保存结果或人工接管。"
                    )
                    publish()
                outcome = "success" if "1" in events else "failure" if "0" in events else None
                if active and (outcome is not None or time.monotonic() - started >= cfg.episode_seconds):
                    finish(outcome, truncated=outcome is None, train=True)
                    continue
                if active and recorder.error:
                    message = f"本集记录已中断，人工跟随仍可使用；结束时保存有效部分：{recorder.error}"
                if now - last_publish >= 0.2:
                    publish()
                    last_publish = now
                if not following:
                    if master_sync and master_sync.aligning:
                        try:
                            if master_sync.update(backend.read(images=False)):
                                master_sync.stop()
                                message = "主臂已同步，保持当前位置；可开始本集后执行策略，或点击相对接管。"
                                publish()
                        except Exception as exc:
                            hold_boundary()
                            message = f"主臂同步已暂停：{exc}"
                            publish()
                        time.sleep(1 / cfg.hz)
                        continue
                    time.sleep(0.01)
                    continue
                now = time.monotonic()
                if now < next_control:
                    time.sleep(min(next_control - now, 0.002))
                    continue
                # Skip missed slots; never burst old master commands to catch up.
                next_control = max(next_control + 1 / cfg.hz, now + 1 / cfg.hz)
                failure_boundary = None
                try:
                    if mode == "human":
                        readings = feedback.read(now)
                        q, width = np.asarray(readings["从臂关节"]), float(readings["从臂夹爪"][0])
                        frames, stamps = cameras.latest
                        capture = time.perf_counter(), frames, stamps
                        if active:
                            failure_boundary = tick(
                                q, width, previous_command, previous_command, "boundary", capture=capture
                            )
                        command, requested = send_native(backend, readings, takeover)
                        source = "right_master_relative" if takeover is not None else "right_master_vla"
                        current = tick(q, width, command, requested, source, capture=capture)
                    else:
                        before = backend.read(images=False)
                        if master_sync:
                            master_sync.update(before)
                        remember_feedback(before)
                        frames, stamps = cameras.latest
                        capture = time.perf_counter(), frames, stamps
                        if active:
                            failure_boundary = tick(
                                before.joints,
                                before.width,
                                previous_command,
                                previous_command,
                                "boundary",
                                capture=capture,
                            )
                        observation = policy_observation(env, before, capture)
                        requested_action = trainer.act(observation, deterministic=args.mode == "eval")
                        _, _, _, _, info = env.step(requested_action, autonomous=True, cycle_started=now)
                        policy_details.update(
                            tracking_error_deg=info["tracking_error_deg"],
                            workspace_limited=info["workspace_limited"],
                            tracking_limited=info["tracking_limited"],
                            camera_age_ms={
                                name: (capture[0] - stamp) * 1000 for name, stamp in stamps.items()
                            },
                            safety_rejection=info["safety_rejection"],
                        )
                        q, width = cfg.decode_joint_target(info["executed_action"])
                        command = np.r_[q, width]
                        rq, rw = cfg.decode_joint_target(requested_action)
                        current = tick(
                            before.joints, before.width, command, np.r_[rq, rw], "policy", info, capture
                        )
                    if active:
                        recorder.submit(current)
                    previous_command = command.copy()
                    control_times.append(time.monotonic())
                    if mode == "policy" and info["safety_rejection"]:
                        boundary = hold_boundary()
                        if active:
                            recorder.segment(boundary)
                        message = f"策略动作暂停：{info['safety_rejection']}。可点击主臂跟随接管。"
                except Exception as exc:
                    following = False
                    message = f"设备反馈或指令错误，跟随已停止：{exc}"
                    print(message, flush=True)
                    try:
                        boundary = hold_boundary()
                    except Exception:
                        backend.stop()
                        boundary = None
                        message += "；反馈无法用于保持，已请求控制器阻尼停止。"
                    if active:
                        # A partially failed command is excluded. The snapshot
                        # before its send can still complete the prior transition.
                        recorder.segment(failure_boundary)
                    publish()
        finally:
            try:
                if keys.abort.is_set():
                    backend.stop()
                if active and recorder:
                    finish(None, truncated=True, train=False)
                elif backend.prepared:
                    hold_boundary()
            finally:
                try:
                    try:
                        if master_sync:
                            master_sync.stop()
                    finally:
                        try:
                            if recorder:
                                recorder.close()
                        finally:
                            if cameras:
                                cameras.close()
                finally:
                    try:
                        master.close()
                    finally:
                        backend.close()
