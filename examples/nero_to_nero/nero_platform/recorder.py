"""单对或左右双对 Nero 的统一采样节拍与 LeRobot 写入。"""

import json
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from . import (
    camera_focus,
    camera_io,
    cameras,
    dataset_edit,
    dataset_transaction,
    depth_images,
    devices,
    pairing,
    resolution,
    sampling,
    timing,
)
from .camera_model import is_depth, stream_names
from .runtime import Runtime, target_control_hz


def dataset_path(runtime: Runtime) -> Path:
    parts = str(runtime.state["repo_id"]).split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("数据集名需要为 owner/name，例如 nero/platform_step1")
    return runtime.dataset_root.joinpath(*parts)


def disk_episode_count(runtime: Runtime) -> int | None:
    """未连接时直接读磁盘 info.json 的集数；数据集不存在或无效时返回 None。

    用于「删除上一条」按钮：刚进平台尚未开始采集时，内存里的 episode_count
    仍是 0，不能反映磁盘上已有多少集。
    """
    try:
        info_path = dataset_path(runtime) / "meta/info.json"
        if not info_path.is_file():
            return None
        info = json.loads(info_path.read_text(encoding="utf-8"))
        total = info.get("total_episodes")
        if type(total) is not int or total < 0:
            return None
        return total
    except (OSError, ValueError):
        return None


def dataset_edit_pending(runtime: Runtime) -> bool:
    try:
        return dataset_transaction.pending(dataset_path(runtime))
    except ValueError:
        return False


def make_features(
    sources: dict,
    pair_mode: str = "single",
    camera_resolution: dict | None = None,
) -> dict:
    """sources: {role: {id, type(rgb/rgbd), path/serial}}；每个角色一路 RGB，RGBD 另写 {role}_depth，与同角色 RGB 对齐。"""
    sizes = camera_resolution or resolution.defaults()
    action_ft = dict.fromkeys(pairing.joint_keys(pair_mode), float)
    obs_ft = dict.fromkeys(pairing.joint_keys(pair_mode), float)
    for role, info in sources.items():
        camera_io.backend(info)  # 不支持的驱动在接触设备前拒绝。
        w, h = resolution.output_size(sizes[role])
        for name in stream_names(role, info["type"]):
            obs_ft[name] = (h, w, 1 if is_depth(name) else 3)
    return combine_feature_dicts(
        hw_to_dataset_features(action_ft, prefix="action", use_video=True),
        hw_to_dataset_features(obs_ft, prefix="observation", use_video=True),
    )


def validate_resume(root: Path, fps: int, features: dict) -> None:
    """在切换硬件模式之前，拒绝向不兼容或不完整的本地目录续录。"""
    if not root.exists():
        return
    dataset_transaction.recover(root)
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"已有目录不是完整的 LeRobot 数据集，请换一个数据集名：{root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["fps"] != fps:
        raise ValueError(f"续录帧率不一致：已有 {info['fps']}，当前 {fps}")
    existing = info["features"]
    existing_cameras = {key for key in existing if key.startswith("observation.images.")}
    selected_cameras = {key for key in features if key.startswith("observation.images.")}
    if existing_cameras != selected_cameras:
        previous = ", ".join(sorted(key.rsplit(".", 1)[-1] for key in existing_cameras))
        selected = ", ".join(sorted(key.rsplit(".", 1)[-1] for key in selected_cameras))
        raise ValueError(
            f"数据集相机字段不一致：已有 {previous}，当前 {selected}。请使用新的数据集名，或恢复原相机选择"
        )
    for key, feature in features.items():
        previous = existing.get(key, {})
        for field in ("dtype", "shape", "names"):
            if json.dumps(previous.get(field)) != json.dumps(feature.get(field)):
                raise ValueError(f"续录字段不一致：{key}.{field}，请恢复原配置或换一个数据集名")
    for key in features:
        if not key.startswith("observation.images.") or not is_depth(key.rsplit(".", 1)[-1]):
            continue
        depth_info = existing[key].get("info", {})
        if (
            depth_info.get("video.depth_min") != 0.0
            or depth_info.get("video.depth_max") != depth_images.DEPTH_MAX_METERS
        ):
            raise ValueError("已有深度编码范围与当前 0～10 米不一致，请用新数据集；避免混淆无效深度 0")


def do_connect(runtime: Runtime) -> tuple[bool, str]:
    if runtime.state["config_error"] or runtime.state["recovery_required"]:
        return False, runtime.state["config_error"] or runtime.state["recovery_required"]
    if runtime.state["close_error"]:
        return False, "数据收尾尚未完成，请查看结束结果并处理数据项"
    if runtime.operation_busy(include_recording=False):
        return False, "机械臂正在执行操作，请等待完成"
    if runtime.state["disable_error"]:
        return False, "上次结束采集并失能未完成，请先重试失能"
    if runtime.state["writer_error"]:
        return False, "本集写入失败，已暂停续录；请保留错误信息，检查数据后重新启动"
    if runtime.state["connected"]:
        return True, "已连接"
    transition_started = False
    preview_closed = False
    try:
        detected = cameras.detect_all_cameras()
        selected = cameras.recording_sources(runtime, detected)
        sizes = runtime.state["camera_resolution"]
        resolution.validate(sizes)
        has_depth = any(info.get("type") == "rgbd" for info in selected.values())
        features = make_features(
            selected,
            runtime.state["pair_mode"],
            camera_resolution=sizes,
        )
        root = dataset_path(runtime)
        validate_resume(root, runtime.state["fps"], features)
        pairs = pairing.resolve(runtime)
        # 同一 UVC 设备不能同时被预览和录制占用；失败时恢复预览。
        cameras.close_preview_cams(runtime)
        preview_closed = True
        runtime.state["camera_sources"] = selected
        for role, info in selected.items():
            adapter = camera_io.backend(info)
            cam = adapter.create(info, sizes[role], runtime.state["fps"])
            runtime.collection_cams[role] = cam  # connect 中途失败也必须释放。
            adapter.connect(cam, info, runtime.state["camera_focus_settings"].get(info["id"]))
            focus = getattr(cam, "metadata", {}).get("focus")
            if focus:
                runtime.camera_focus[info["id"]] = focus
                if focus["state"] == "warning":
                    print(f"[相机对焦] {role}：{focus['message']}")
        # 所有相机的模式与分辨率验证完成后，才切换机械臂模式。
        transition_started = True
        runtime.state["disable_status"] = ""
        runtime.state["end_status"] = ""
        runtime.state["close_result"] = None
        devices.transition_to_collection(runtime)
        runtime.state["camera_names"] = list(selected)
        runtime.state["active_cameras"] = {
            name: info.get("path", info.get("serial", info["id"]))
            for role, info in selected.items()
            for name in stream_names(role, info["type"])
        }
        for role, info in selected.items():
            print(f"[采集相机] {role}: {info}")
        try:
            runtime.dataset = LeRobotDataset.create(
                repo_id=runtime.state["repo_id"],
                fps=runtime.state["fps"],
                features=features,
                root=root,
                robot_type="nero_dual_follower" if runtime.state["pair_mode"] == "dual" else "nero_follower",
                use_videos=True,
                image_writer_threads=4,
                concatenate_videos=False,
                depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=depth_images.DEPTH_MAX_METERS)
                if has_depth
                else None,
            )
        except FileExistsError:
            runtime.dataset = LeRobotDataset.resume(
                repo_id=runtime.state["repo_id"],
                root=root,
                image_writer_threads=4,
                concatenate_videos=False,
                depth_encoder=DepthEncoderConfig(depth_min=0.0, depth_max=depth_images.DEPTH_MAX_METERS)
                if has_depth
                else None,
            )
        runtime.state["episode_count"] = runtime.dataset.meta.total_episodes
        runtime.state["connected"] = True
        runtime.state["teleop_needs_restore"] = False
        runtime.state["last_error"] = None
        return True, "已就绪：" + "；".join(
            f"{pair.display}（{pair.leader} → {pair.follower}）" for pair in pairs
        )
    except Exception as exc:
        traceback.print_exc()
        runtime.state["last_error"] = f"启动失败: {exc}"
        _close_collection_cams(runtime)
        if transition_started:
            _leave_leader_mode(runtime)
        if preview_closed:
            cameras.open_preview_cams(runtime)
        return False, runtime.state["last_error"]


def _close_collection_cams(runtime: Runtime) -> None:
    runtime.record_preview = {}
    for cam in runtime.collection_cams.values():
        try:
            cam.disconnect()
        except Exception as exc:
            print(f"采集相机关闭失败: {exc}")
    runtime.collection_cams.clear()
    runtime.state["active_cameras"] = {}
    runtime.state["camera_sources"] = {}


def _leave_leader_mode(runtime: Runtime) -> None:
    for _, leader_label, _ in pairing.labels(runtime.state):
        leader_ch = devices.channel_by_label(runtime, leader_label)
        if leader_ch is not None and runtime.channels[leader_ch]["arm"] is not None:
            try:
                devices.set_control_role(runtime, leader_ch, "follower")
                time.sleep(0.3)
            except Exception as exc:
                print(f"{leader_label} 退出主臂模式失败: {exc}")


def do_disconnect(runtime: Runtime, reopen_preview: bool = True) -> None:
    """等待本集保存后收尾，保持原版退出主臂模式但不失能的行为。"""
    if runtime.state["homing"]:
        raise RuntimeError("正在返回默认位置，请等待回位结束再断开")
    request_stop(runtime)
    runtime.state["stop_requested"] = True
    if runtime.record_thread is not None:
        runtime.record_thread.join()
        runtime.record_thread = None
    if runtime.dataset is not None:
        runtime.dataset.finalize()
        runtime.dataset = None
    _close_collection_cams(runtime)
    if runtime.state["connected"]:
        _leave_leader_mode(runtime)
    runtime.state["connected"] = False
    runtime.state["recording"] = False
    if reopen_preview:
        cameras.open_preview_cams(runtime)


def start_teleop_test(runtime: Runtime) -> tuple[bool, str]:
    """启动只遥操作、不录制数据的训练模式；不需要相机。"""
    with runtime.lifecycle_lock:
        if runtime.state["shutting_down"]:
            return False, "程序正在退出，请等待收尾完成"
        if runtime.state["config_error"] or runtime.state["recovery_required"]:
            return False, runtime.state["config_error"] or runtime.state["recovery_required"]
        if runtime.operation_busy():
            return False, "机械臂正在执行操作，请等待完成"
        if runtime.state["connected"]:
            return False, "已开启采集会话，请先退出采集"
        if runtime.state["teleop_test_active"]:
            return False, "遥操作测试已在进行"
        try:
            pairing.resolve(runtime)
        except ValueError as exc:
            return False, str(exc)
        runtime.state.update(
            teleop_test_active=True,
            teleop_test_message="正在使能主从臂并切换跟随模式…",
            teleop_test_stop=False,
        )
        try:
            runtime.teleop_test_thread = threading.Thread(
                target=_teleop_test_loop, args=(runtime,), name="nero-teleop-test", daemon=False
            )
            runtime.teleop_test_thread.start()
        except Exception as exc:
            runtime.teleop_test_thread = None
            runtime.state.update(teleop_test_active=False, teleop_test_message="")
            return False, f"遥操作测试线程启动失败: {exc}"
        return True, "已启动遥操作测试"


def stop_teleop_test(runtime: Runtime) -> tuple[bool, str]:
    """请求停止遥操作测试；等待跟随循环退出并恢复主臂角色。"""
    with runtime.lifecycle_lock:
        if not runtime.state["teleop_test_active"]:
            return True, "当前没有进行中的遥操作测试"
        runtime.state["teleop_test_stop"] = True
        # 对齐/跟随循环内部用 stop_requested 感知取消，同步置位以覆盖对齐阶段。
        runtime.state["stop_requested"] = True
        runtime.state["teleop_test_message"] = "正在停止遥操作测试…"
    if runtime.teleop_test_thread is not None:
        runtime.teleop_test_thread.join()
    with runtime.lifecycle_lock:
        runtime.state["stop_requested"] = False
    return True, "遥操作测试已停止"


def _teleop_test_loop(runtime: Runtime) -> None:
    """只跟随、不采样相机、不写数据集；使能主从臂后循环下发从臂指令。"""
    try:
        pairs = pairing.resolve(runtime)
        motion_mode = runtime.state["follow_motion_mode"]
        if motion_mode not in ("js", "j"):
            raise ValueError("跟随方式配置无效，尚未下发指令")
        control_hz = target_control_hz(runtime.state)
        # 分阶段更新 teleop_test_message，供前端语音按阶段播报。
        with runtime.lifecycle_lock:
            runtime.state["teleop_test_message"] = "使能主从臂并切换示教/跟随模式…"
        devices.transition_to_collection(runtime)
        with runtime.lifecycle_lock:
            runtime.state["teleop_test_message"] = "夹爪示教参数已设置（行程 100mm），正在对齐从臂…"
        devices.align_follower_before_recording(runtime)
        if runtime.state["teleop_test_stop"]:
            return
        feedbacks = [
            sampling.RobotFeedback(runtime.channels[pair.leader], runtime.channels[pair.follower])
            for pair in pairs
        ]

        def read_all() -> list[dict]:
            return [feedback.read(time.monotonic()) for feedback in feedbacks]

        read_all()
        while not all(feedback.ready for feedback in feedbacks):
            if runtime.state["teleop_test_stop"]:
                return
            time.sleep(0.01)
            read_all()
        control = timing.ControlTiming(control_hz, time.monotonic())
        first_command = True
        attempted_followers: dict[str, tuple[float, float, bool] | None] = {}
        with runtime.lifecycle_lock:
            runtime.state["teleop_test_message"] = "遥操作测试进行中：拖动主臂，从臂跟随（不录制数据）"
        while not runtime.state["teleop_test_stop"]:
            started = time.monotonic()
            if started + 1e-9 < control.deadline:
                time.sleep(min(control.deadline - started, 0.05))
                continue
            control_slot = control.begin(started)
            try:
                readings = read_all()
                actions = []
                for pair, values in zip(pairs, readings, strict=True):
                    angles = values["主臂关节"]
                    if first_command:
                        devices.require_joint_alignment(
                            angles, values["从臂关节"], f"{pair.display} 第一帧主臂指令与从臂"
                        )
                    actions.append(angles[:] + [min(max(values["主臂夹爪"][0], 0.0), 0.09)])
                first_command = False
                for pair, action, feedback in zip(pairs, actions, feedbacks, strict=True):
                    follower = runtime.channels[pair.follower]
                    if motion_mode == "j":
                        attempted_followers[pair.follower] = feedback.seen.get("从臂关节")
                        follower["arm"].move_j(action[:7])
                    else:
                        follower["arm"].move_js(action[:7])
                    follower["gripper"].move_gripper_m(
                        value=action[7], force=runtime.state["teleop_gripper_force"]
                    )
                control.finish(control_slot, started, time.monotonic(), control.frames, None)
            except (sampling.FeedbackError, sampling.CommandError) as exc:
                with runtime.lifecycle_lock:
                    runtime.state["teleop_test_message"] = f"遥操作测试中止：{exc}"
                break
            except Exception as exc:
                traceback.print_exc()
                with runtime.lifecycle_lock:
                    runtime.state["teleop_test_message"] = f"遥操作测试出错：{exc}"
                raise
        if motion_mode == "j" and attempted_followers:
            devices.hold_planned_followers(runtime, attempted_followers, time.monotonic())
    except Exception as exc:
        traceback.print_exc()
        with runtime.lifecycle_lock:
            runtime.state["teleop_test_message"] = f"遥操作测试失败: {exc}"
    finally:
        _leave_leader_mode(runtime)
        with runtime.lifecycle_lock:
            runtime.state["teleop_test_active"] = False
            runtime.state["teleop_test_stop"] = False
            if not runtime.state["teleop_test_message"].startswith("遥操作测试失败"):
                runtime.state["teleop_test_message"] = "遥操作测试已结束"


def start_recording(runtime: Runtime) -> tuple[bool, str]:
    with runtime.lifecycle_lock:
        if runtime.state["config_error"] or runtime.state["recovery_required"]:
            return False, runtime.state["config_error"] or runtime.state["recovery_required"]
        if runtime.state["close_error"]:
            return False, "数据收尾尚未完成，请查看结束结果并处理数据项"
        if runtime.state["shutting_down"]:
            return False, "程序正在退出，请等待保存收尾完成"
        if runtime.operation_busy(include_recording=False):
            return False, "机械臂正在执行操作，请等待完成"
        if runtime.state["disable_error"]:
            return False, "上次结束采集并失能未完成，请先重试失能"
        if runtime.state["home_error"]:
            return False, "回位未成功，请检查机械臂并重新完成回位后再录制"
        if runtime.state["writer_error"]:
            return False, "本集写入失败，已暂停续录；请结束采集并保留错误信息，检查数据后重新启动"
        if runtime.state["recording"] or (runtime.record_thread and runtime.record_thread.is_alive()):
            return False, "正在准备、录制或保存，请等待当前操作完成"
        aligned, message = devices.check_alignment(runtime, require_aligned=False)
        if not aligned:
            runtime.state.update(record_phase="idle", record_message="等待有效关节反馈", last_error=message)
            return False, message
        runtime.record_preview = {}
        runtime.state.update(
            stop_requested=False,
            cancel_requested=False,
            last_error=None,
            recording=True,
            record_phase="preparing",
            record_message="正在准备机械臂、相机和数据集…",
            record_started_at=None,
            record_elapsed=0.0,
            record_frames=0,
            record_frame_errors=0,
            record_frame_error=None,
            record_timing=None,
            control_timing=None,
            timing_report_error=None,
        )
        try:
            runtime.record_thread = threading.Thread(
                target=_prepare_and_record, args=(runtime,), name="nero-recording", daemon=False
            )
            runtime.record_thread.start()
        except Exception as exc:
            runtime.record_thread = None
            message = f"录制线程启动失败: {exc}"
            runtime.state.update(
                recording=False, record_phase="error", record_message=message, last_error=message
            )
            return False, message
        return True, "已开始准备录制"


def _prepare_and_record(runtime: Runtime) -> None:
    """启动阶段也由同一个工作线程持有，接口立即返回，允许取消和查询状态。"""
    try:
        if not runtime.state["stop_requested"]:
            if not runtime.state["connected"]:
                ok, message = do_connect(runtime)
                if not ok:
                    raise RuntimeError(message)
            elif runtime.state["teleop_needs_restore"]:
                try:
                    devices.transition_to_collection(runtime)
                except Exception:
                    _leave_leader_mode(runtime)
                    raise
                runtime.state["teleop_needs_restore"] = False
        with runtime.lifecycle_lock:
            if not runtime.state["stop_requested"]:
                runtime.state["record_message"] = "正在准备从臂对齐，请保持主臂静止…"
        devices.align_follower_before_recording(runtime)
        with runtime.lifecycle_lock:
            if runtime.state["stop_requested"]:
                runtime.state.update(record_phase="idle", record_message="准备已取消，未录制新的一集")
                return
            runtime.state.update(home_status="", disable_status="")
        record_worker(runtime, runtime.state.get("camera_names", []))
    except Exception as exc:
        if not isinstance(exc, devices.AlignmentError):
            traceback.print_exc()
        message = f"准备录制失败: {exc}"
        with runtime.lifecycle_lock:
            runtime.state.update(record_phase="error", record_message=message, last_error=message)
    finally:
        with runtime.lifecycle_lock:
            runtime.state["recording"] = False


def request_stop(runtime: Runtime) -> tuple[bool, str]:
    with runtime.lifecycle_lock:
        phase = runtime.state["record_phase"]
        if phase == "cancelling":
            return True, "正在取消本集，等待清理完成；本集不会保存"
        if phase == "saving":
            return True, "本集正在保存，请等待完成"
        if not runtime.state["recording"]:
            return True, "当前没有正在录制的一集"
        runtime.state["stop_requested"] = True
        if phase in ("preparing", "aligning"):
            message = (
                "已请求取消对齐，正在停止从臂…"
                if phase == "aligning"
                else "已请求取消准备，等待当前设备操作结束…"
            )
        else:
            message = "正在结束本集，随后保存…"
            runtime.state["record_phase"] = "stopping"
        runtime.state["record_message"] = message
        return True, message


def request_cancel(runtime: Runtime) -> tuple[bool, str]:
    """录制进行中丢弃当前这一集，不保存；区别于 request_stop（结束并保存）。"""
    with runtime.lifecycle_lock:
        phase = runtime.state["record_phase"]
        if phase == "cancelling" and runtime.state["cancel_requested"]:
            return True, "正在取消本集，请等待清理完成"
        if not runtime.state["recording"] or phase != "recording" or runtime.state["stop_requested"]:
            return False, "只有正在录制的一集可以取消；已结束或开始保存的集不能取消"
        runtime.state["cancel_requested"] = True
        runtime.state["stop_requested"] = True
        runtime.state["record_phase"] = "cancelling"
        runtime.state["record_message"] = "正在取消本集并丢弃已录帧…"
        return True, "已请求中途取消，本集将不保存"


def recording_status(runtime: Runtime) -> dict:
    """一次快照供页面使用；保存时冻结录制时长，不把编码时间算入本集。"""
    with runtime.lifecycle_lock:
        state = runtime.state
        elapsed = state["record_elapsed"]
        if state["record_started_at"] is not None and state["record_phase"] in (
            "recording",
            "stopping",
            "cancelling",
        ):
            elapsed = max(0.0, time.monotonic() - state["record_started_at"])
        return {
            "busy": state["recording"],
            "episode_count": state["episode_count"],
            "phase": state["record_phase"],
            "stop_requested": state["stop_requested"],
            "cancel_requested": state["cancel_requested"],
            "message": state["record_message"],
            "frames": state["record_frames"],
            "elapsed_seconds": round(elapsed, 1),
            "limit_seconds": state["episode_sec"],
            "follow_motion_mode": state["follow_motion_mode"],
            "control_frequency": state["control_frequency"],
            "follow_stop_message": state["follow_stop_message"],
            "frame_errors": state["record_frame_errors"],
            "frame_error": state["record_frame_error"],
            "last_episode": state["last_episode"],
            "writer_error": state["writer_error"],
            "timing": state["record_timing"],
            "control_hz": target_control_hz(state),
            "control_timing": state["control_timing"],
            "timing_report_error": state["timing_report_error"],
        }


def end_collection_and_disable(
    runtime: Runtime,
    expected_targets: dict | None,
    retry_stage: str | None = None,
    channel: str | None = None,
    operation: str = "combined",
) -> tuple[bool, str]:
    """分别执行数据收尾或失能；combined 仅兼容旧接口，重试只处理选定步骤。"""
    if operation not in ("combined", "end", "disable"):
        return False, "未知结束操作"
    data_only = operation == "end" or retry_stage == "data"
    with runtime.lifecycle_lock:
        if runtime.operation_busy(include_recording=False):
            return False, "机械臂正在执行操作，请等待完成"
        if runtime.state["record_phase"] in ("preparing", "aligning") and runtime.state["recording"]:
            return False, "正在准备录制，请先取消准备并等待完成，再结束采集"
        if operation == "disable" and (
            runtime.state["connected"]
            or runtime.state["recording"]
            or (runtime.record_thread and runtime.record_thread.is_alive())
        ):
            return False, "请先结束采集并等待数据收尾，再单独确认机械臂失能"
        try:
            targets = pairing.targets(pairing.resolve(runtime, require_detected=False))
        except ValueError as exc:
            return False, str(exc)
        if expected_targets != targets:
            return False, "主从臂选择已变化，请刷新页面并重新确认操作目标"
        if retry_stage is not None:
            previous = runtime.state["close_result"]
            if (
                runtime.state["connected"]
                or runtime.state["recording"]
                or not previous
                or previous["targets"] != targets
            ):
                return False, "没有可重试的结束操作，请刷新状态"
            if retry_stage == "data":
                if previous["data"]["state"] != "error":
                    return False, "数据没有可重试的收尾步骤"
            elif retry_stage in ("arm", "status"):
                if channel not in targets or previous["arms"][channel]["state"] != "error":
                    return False, "该机械臂没有待重试的失能步骤"
            else:
                return False, "未知重试步骤"
            result = previous
        elif operation == "disable":
            if runtime.state["disable_error"]:
                return False, "请在结束结果中仅重试未完成的失能项"
            previous = runtime.state["close_result"]
            if runtime.state["close_error"] and (not previous or previous["targets"] != targets):
                return False, "数据收尾目标不一致，请刷新页面并检查结束结果"
            result = (
                previous
                if previous and previous["targets"] == targets
                else {
                    "targets": targets.copy(),
                    "data": {"state": "done", "message": "未开启采集会话，无需数据收尾"},
                    "preview": {"state": "done", "message": "相机预览保持当前状态"},
                    "warnings": [],
                }
            )
            result["arms"] = {
                ch: {"label": label, "state": "pending", "message": "等待失能确认"}
                for ch, label in targets.items()
            }
            runtime.state["close_result"] = result
        else:
            if runtime.state["disable_error"] or runtime.state["close_error"]:
                return False, "请在结束结果中仅重试未完成项"
            result = {
                "targets": targets.copy(),
                "data": {"state": "pending", "message": "等待保存和收尾"},
                "arms": {
                    ch: {
                        "label": label,
                        "state": "not_requested" if data_only else "pending",
                        "message": "未执行失能；当前电机状态请看左侧" if data_only else "尚未确认失能",
                    }
                    for ch, label in targets.items()
                },
                "preview": {"state": "pending", "message": "尚未恢复"},
                "warnings": [],
            }
            runtime.state["close_result"] = result
        if retry_stage is None:
            result["operation"] = operation
        runtime.state.update(disabling=not data_only, ending=data_only)
        if data_only:
            runtime.state["end_status"] = "正在等待本集保存并完成数据收尾；不会失能机械臂…"
        else:
            runtime.state["disable_status"] = "正在处理，请继续托稳机械臂，等待逐项确认…"
        runtime.state["stop_requested"] = True

    try:
        if retry_stage is None and runtime.record_thread is not None:
            runtime.record_thread.join()
            runtime.record_thread = None
        if retry_stage == "data" or (retry_stage is None and operation != "disable"):
            try:
                had_dataset = runtime.dataset is not None
                if had_dataset:
                    runtime.dataset.finalize()
                    runtime.dataset = None
                if runtime.state["writer_error"]:
                    result["data"] = {
                        "state": "needs_review",
                        "message": f"数据写入曾失败，需要检查文件：{runtime.state['writer_error']}",
                    }
                else:
                    result["data"] = {
                        "state": "done",
                        "message": "数据已保存并完成收尾" if had_dataset else "没有待收尾的采集数据",
                    }
            except Exception as exc:
                result["data"] = {"state": "error", "message": f"数据收尾失败，保留句柄可重试：{exc}"}
        if retry_stage is None and operation != "disable":
            _close_collection_cams(runtime)
            if operation == "end":
                _leave_leader_mode(runtime)
        arm_targets = (
            list(targets)
            if retry_stage is None and not data_only
            else ([channel] if retry_stage in ("arm", "status") else [])
        )
        for ch in arm_targets:
            confirm_only = retry_stage == "status"
            action_text = "只读取新反馈，重新确认失能" if confirm_only else "失能并读取确认反馈"
            result["arms"][ch].update(state="pending", message="正在" + action_text)
            if not confirm_only and targets[ch].startswith("主"):
                try:
                    devices.set_control_role(runtime, ch, "follower")
                    time.sleep(0.3)
                except Exception as exc:
                    result["warnings"].append(f"{ch} 退出示教失败：{exc}；失能结果以关节反馈为准")
            runtime.state["disable_status"] = f"{targets[ch]} ({ch})：正在{action_text}…"
            try:
                devices.disable_channel(runtime, ch, confirm_only=confirm_only)
                result["arms"][ch].update(
                    state="done", message=runtime.channels[ch]["disable_result"]["message"]
                )
            except Exception as exc:
                result["arms"][ch].update(state="error", message=str(exc))
            result["arms"][ch]["details"] = runtime.channels[ch].get("disable_result")
        runtime.state.update(recording=False, home_status="")
        if retry_stage is None and operation != "disable":
            try:
                cameras.open_preview_cams(runtime)
                result["preview"] = {"state": "done", "message": "相机预览已恢复"}
            except Exception as exc:
                result["preview"] = {"state": "warning", "message": f"预览恢复失败，不影响失能结果：{exc}"}
        runtime.state["connected"] = False
    except Exception as exc:
        result["warnings"].append(f"结束操作被中断：{exc}")
        for entry in [result["data"], *result["arms"].values()]:
            if entry["state"] == "pending":
                entry.update(state="error", message=f"该步骤未完成：{exc}")
        if runtime.record_thread is None or not runtime.record_thread.is_alive():
            runtime.state.update(connected=False, recording=False)
    finally:
        with runtime.lifecycle_lock:
            arm_errors = [
                f"{ch} ({entry['label']})：{entry['message']}"
                for ch, entry in result["arms"].items()
                if entry["state"] not in ("done", "not_requested")
            ]
            data_error = result["data"]["message"] if result["data"]["state"] != "done" else None
            completed = (
                "采集已结束，数据收尾完成；未执行机械臂失能，请查看设备使能状态"
                if any(entry["state"] == "not_requested" for entry in result["arms"].values())
                else "采集已结束，数据收尾完成，所选主从臂及已接入夹爪已失能"
            )
            message = "；".join(arm_errors + ([data_error] if data_error else [])) or completed
            runtime.state.update(
                disabling=False,
                ending=False,
                disable_error="；".join(arm_errors) if arm_errors else None,
                close_error=data_error,
            )
            runtime.state["end_status" if data_only else "disable_status"] = message
            if operation == "disable" and runtime.state["end_status"]:
                runtime.state["end_status"] = data_error or "采集已结束，数据收尾完成；机械臂失能结果见下方"
            runtime.state["last_error"] = message if arm_errors or data_error else None
    if operation == "disable" and not arm_errors:
        message = "所选主从臂及已接入夹爪已确认失能" + (f"；数据仍需处理：{data_error}" if data_error else "")
        runtime.state["disable_status"] = message
        return True, message
    return not arm_errors and data_error is None, message


def record_worker(runtime: Runtime, camera_names: list[str]) -> None:
    """录制与保存分阶段显示，只有 save_episode 成功才增加集数。"""
    with runtime.lifecycle_lock:
        runtime.state.update(
            record_phase="recording",
            record_message="正在录制",
            recording=True,
            record_frames=0,
            record_elapsed=0.0,
            record_frame_errors=0,
            record_frame_error=None,
            record_timing=None,
            control_timing=None,
            timing_report_error=None,
            follow_stop_message="",
        )
    try:
        _record_episode(runtime, camera_names)
    except Exception as exc:
        traceback.print_exc()
        message = f"本集未确认保存成功: {exc}"
        with runtime.lifecycle_lock:
            if runtime.state["record_frames"] > 0 and not runtime.state["writer_error"]:
                runtime.state["writer_error"] = f"本集异常中止，已有帧未确认保存: {exc}"
            if runtime.state["record_started_at"] is not None and runtime.state["record_phase"] != "saving":
                runtime.state["record_elapsed"] = max(
                    0.0, time.monotonic() - runtime.state["record_started_at"]
                )
            runtime.state.update(record_phase="error", record_message=message, last_error=message)
    finally:
        with runtime.lifecycle_lock:
            runtime.state.update(recording=False, record_started_at=None, cancel_requested=False)


def _inspect_camera_focus(runtime: Runtime, camera_names: list[str], stage: str) -> dict:
    """仅在集边界读回，状态轮询使用缓存；RGBD 由适配器声明不适用。"""
    results = {}
    for role in camera_names:
        info = runtime.state["camera_sources"].get(role)
        if not info:
            continue
        opened = getattr(runtime.collection_cams[role], "metadata", {}).get("focus") or {}
        result = camera_io.backend(info).inspect_focus(info, stage, opened.get("setting"))
        if result is not None:
            results[role] = result
            runtime.camera_focus[info["id"]] = result
    return results


def _record_episode(runtime: Runtime, camera_names: list[str]) -> None:
    pairs = pairing.resolve(runtime)
    motion_mode = runtime.state["follow_motion_mode"]
    if motion_mode not in ("js", "j"):
        raise ValueError("跟随方式配置无效，尚未下发指令")
    frequency_setting = runtime.state["control_frequency"]
    control_hz = target_control_hz(runtime.state)
    follows_capture = frequency_setting == "capture"
    print(f"[采集] 记录线程启动: {[pair.display for pair in pairs]} 相机={camera_names}")
    focus_checks = {
        role: {
            "device_id": runtime.state["camera_sources"][role]["id"],
            "opened": getattr(runtime.collection_cams[role], "metadata", {}).get("focus"),
            "before": result,
        }
        for role, result in _inspect_camera_focus(runtime, camera_names, "episode_start").items()
    }
    feedbacks = []
    for pair in pairs:
        try:
            feedbacks.append(
                sampling.RobotFeedback(runtime.channels[pair.leader], runtime.channels[pair.follower])
            )
        except sampling.FeedbackError as exc:
            raise sampling.FeedbackError(f"{pair.display}：{exc}") from exc

    def read_all() -> tuple[list[dict], dict]:
        values, traces = [], {}
        for pair, feedback in zip(pairs, feedbacks, strict=True):
            begin = time.monotonic()
            try:
                values.append(feedback.read(begin))
            except sampling.FeedbackError as exc:
                raise sampling.FeedbackError(f"{pair.display}：{exc}") from exc
            traces[pair.name] = {
                "read_started_at": begin,
                "read_completed_at": time.monotonic(),
                "feedback_timestamps": {key: seen[0] for key, seen in feedback.seen.items()},
            }
        return values, traces

    # 先观察每路时间戳至少更新一次，避免将上次会话遗留缓存作为第一条运动指令。
    read_all()
    while not all(feedback.ready for feedback in feedbacks):
        if runtime.state["stop_requested"]:
            runtime.state.update(record_phase="idle", record_message="准备已取消，未新增集")
            return
        time.sleep(0.01)
        read_all()
    t0 = time.monotonic()
    runtime.state["record_started_at"] = t0
    cadence = timing.EpisodeTiming(runtime.state["fps"], t0)
    control = timing.ControlTiming(control_hz, t0)
    end_at = t0 + runtime.state["episode_sec"]
    first_command = True
    aborted = None
    attempted_followers: dict[str, tuple[float, float, bool] | None] = {}

    def command_cycle() -> tuple[list[list[float]], list[float], dict]:
        nonlocal first_command
        # 先完整读取、校验全部配对，再发送任一侧指令，避免读坏右侧后仍控制左侧。
        readings, robot_samples = read_all()
        actions, observations = [], []
        for pair, values in zip(pairs, readings, strict=True):
            angles, fangles = values["主臂关节"], values["从臂关节"]
            if first_command:
                devices.require_joint_alignment(angles, fangles, f"{pair.display} 第一帧主臂指令与从臂")
            robot_samples[pair.name]["leader_joint_positions"] = angles[:]
            actions.append(angles[:] + [min(max(values["主臂夹爪"][0], 0.0), 0.09)])
            observations.extend(fangles + values["从臂夹爪"])
        first_command = False
        for pair, action, feedback in zip(pairs, actions, feedbacks, strict=True):
            robot_samples[pair.name]["command_started_at"] = time.monotonic()
            try:
                follower = runtime.channels[pair.follower]
                if motion_mode == "j":
                    # 即使 SDK 在发送中途异常，也可能已有部分指令到达，必须尝试收尾。
                    attempted_followers[pair.follower] = feedback.seen.get("从臂关节")
                    follower["arm"].move_j(action[:7])
                else:
                    follower["arm"].move_js(action[:7])
                follower["gripper"].move_gripper_m(value=action[7], force=runtime.state["gripper_force"])
            except Exception as exc:
                raise sampling.CommandError(
                    f"{pair.display} 跟随指令下发失败：{exc}；本帧不写入，不重试指令"
                ) from exc
            robot_samples[pair.name]["command_completed_at"] = time.monotonic()
            robot_samples[pair.name]["commanded_action"] = action[:]
        return actions, observations, robot_samples

    actions, observations, robot_samples = [], [], {}
    latest_control_index = None
    latest_control_started = t0
    try:
        while not runtime.state["stop_requested"]:
            started = time.monotonic()
            if started + 1e-9 >= end_at:
                cadence.finish_duration(runtime.state["episode_sec"])
                control.finish_duration(runtime.state["episode_sec"])
                break
            deadline = cadence.deadline if follows_capture else min(cadence.deadline, control.deadline)
            if started + 1e-9 < deadline:
                time.sleep(min(deadline - started, end_at - started, 0.05))
                continue
            # 控制和采样各有绝对时刻；同一线程串行调度，避免并发写同一 CAN 设备。
            # 相机/写入超时可能延误控制；只执行最新时刻，记录错过次数，绝不突发补发。
            if follows_capture or started + 1e-9 >= control.deadline:
                control_slot = control.begin(started)
                control_error = None
                control_index = None
                trace = {}
                try:
                    actions, observations, trace = command_cycle()
                    control_index = control.frames
                    latest_control_index = control_index
                    latest_control_started = started
                except (sampling.FeedbackError, sampling.CommandError) as exc:
                    aborted = control_error = str(exc)
                    with runtime.lifecycle_lock:
                        runtime.state["record_frame_errors"] += 1
                        runtime.state["record_frame_error"] = aborted
                    break
                except Exception as exc:
                    control_error = str(exc)
                    raise
                finally:
                    completed = time.monotonic()
                    for entry in trace.values():
                        for key in list(entry):
                            if key.endswith("_at"):
                                entry[key.removesuffix("_at") + "_seconds"] = entry.pop(key) - t0
                    control.finish(
                        control_slot, started, completed, control_index, control_error, robots=trace
                    )
                    robot_samples = trace
                    with runtime.lifecycle_lock:
                        runtime.state["control_timing"] = control.control_summary(completed - t0)
            started = time.monotonic()
            if runtime.state["stop_requested"] or started + 1e-9 >= end_at:
                continue
            if started + 1e-9 < cadence.deadline:
                continue
            slot = cadence.begin(started)
            frame_index = None
            sample_error = None
            camera_samples = {}
            try:
                frame = {
                    "action": np.array([value for action in actions for value in action], dtype=np.float32),
                    "observation.state": np.array(observations, dtype=np.float32),
                    "task": runtime.state["task"],
                }
                preview = {}
                for role in camera_names:
                    cam = runtime.collection_cams[role]
                    info = runtime.state["camera_sources"].get(role, {"type": "rgb"})
                    images, details = camera_io.sample(role, info, cam)
                    for name, image in images.items():
                        frame[f"observation.images.{name}"] = image
                        preview[name] = (cam, image, started)
                    camera_samples.update(details)
            except Exception as e:
                sample_error = str(e)
                print(f"帧出错: {e}")
                with runtime.lifecycle_lock:
                    runtime.state["record_frame_errors"] += 1
                    runtime.state["record_frame_error"] = str(e)
            else:
                try:
                    runtime.dataset.add_frame(frame)
                except Exception as exc:
                    sample_error = str(exc)
                    runtime.state["writer_error"] = f"本集写入失败: {exc}"
                    runtime.state["record_elapsed"] = time.monotonic() - t0
                    raise
                runtime.record_preview = preview
                with runtime.lifecycle_lock:
                    frame_index = runtime.state["record_frames"]
                    runtime.state["record_frames"] += 1
            finally:
                completed = time.monotonic()
                cadence.finish(
                    slot, started, completed, frame_index, sample_error, camera_samples, robot_samples
                )
                cadence.samples[-1].update(
                    control_index=latest_control_index,
                    control_age_ms=round(max(0.0, started - latest_control_started) * 1000, 3),
                )
                with runtime.lifecycle_lock:
                    runtime.state["record_timing"] = cadence.summary(completed - t0)
    finally:
        with runtime.lifecycle_lock:
            # 结束跟随后请求保持的耗时不属于采样时长。
            runtime.state["record_elapsed"] = max(0.0, time.monotonic() - t0)
        if motion_mode == "j" and attempted_followers:
            # 正常到时、手动结束和异常退出都在视频保存之前替换未完成目标。
            stop_error = devices.hold_planned_followers(runtime, attempted_followers, time.monotonic())
            if stop_error:
                aborted = f"{aborted}；{stop_error}" if aborted else stop_error
        # 运动收尾已完成；读回不计入采样时长，也绝不修改录制中的对焦/曝光参数。
        for role, result in _inspect_camera_focus(runtime, camera_names, "episode_end").items():
            focus_checks[role]["after"] = result
    with runtime.lifecycle_lock:
        runtime.state["record_timing"] = cadence.summary(runtime.state["record_elapsed"])
        runtime.state["record_timing"]["warnings"].extend(camera_focus.audit_warnings(focus_checks))
        runtime.state["control_timing"] = control.control_summary(runtime.state["record_elapsed"])
    has_frames = runtime.dataset.has_pending_frames()
    with runtime.lifecycle_lock:
        # 取消决策与封闭保存入口必须是同一个临界区。
        cancelled = runtime.state["cancel_requested"]
        if not cancelled:
            runtime.state.update(
                record_phase="saving" if has_frames else "idle",
                record_message="正在保存本集数据和视频，请稍候…"
                if has_frames
                else "本集尚无有效帧，未新增集",
            )
    if cancelled:
        runtime.dataset.clear_episode_buffer()
        with runtime.lifecycle_lock:
            runtime.state.update(
                record_phase="cancelled",
                record_message="本集已取消，已录帧已丢弃，未保存；机械臂保持使能"
                + (f"；机械臂状态需处理：{aborted}" if aborted else ""),
                record_frames=0,
                record_elapsed=0.0,
                cancel_requested=False,
                record_timing=None,
                control_timing=None,
                last_error=aborted,
            )
        runtime.record_preview = {}
        return
    if not has_frames:
        if aborted:
            raise sampling.FeedbackError(aborted)
        if runtime.state["stop_requested"]:
            return
        raise RuntimeError("本集没有有效帧，未保存；请查看采集终端的错误")
    episode_index = runtime.state["episode_count"]
    try:
        runtime.dataset.save_episode()
    except Exception as exc:
        runtime.state["writer_error"] = f"本集保存失败: {exc}"
        raise
    saved_report = None
    # LeRobot 数据保存已成功；报告写入失败单独提示，不能把已保存的一集当成失败重写。
    root = getattr(runtime.dataset, "root", None)
    if root is not None:
        try:
            saved_report = str(
                timing.save_report(
                    Path(root),
                    episode_index,
                    {
                        "schema_version": 3,
                        "repo_id": runtime.state["repo_id"],
                        "episode_index": episode_index,
                        "measurement": "host_monotonic_cycle_start",
                        "sensor_sync_verified": False,
                        "pair_mode": runtime.state["pair_mode"],
                        "pairs": pairing.describe(pairs),
                        "joint_names": pairing.joint_keys(runtime.state["pair_mode"]),
                        "termination_error": aborted,
                        "camera_devices": {
                            role: {
                                **runtime.state["camera_sources"].get(role, {}),
                                **getattr(cam, "metadata", {}),
                            }
                            for role, cam in runtime.collection_cams.items()
                        },
                        "camera_resolution": runtime.state["camera_resolution"],
                        "camera_focus": {
                            "measurement": "v4l2_control_readback",
                            "continuous_monitoring": False,
                            "note": "仅检查打开时和本集首尾，不证明集内全程不变，也不代表光学清晰度已验证。",
                            "cameras": focus_checks,
                        },
                        "action_source": "commanded_joint_targets_and_leader_gripper",
                        "follow_motion_mode": motion_mode,
                        "control_frequency": frequency_setting,
                        "follow_stop_message": runtime.state["follow_stop_message"],
                        "control": {
                            "scheduler": "capture_cadence"
                            if follows_capture
                            else "independent_deadlines_serial_io",
                            "action_sampling": "latest_completed_control_cycle",
                            "summary": runtime.state["control_timing"],
                            "samples": control.samples,
                        },
                        "summary": runtime.state["record_timing"],
                        "samples": cadence.samples,
                    },
                )
            )
        except Exception as exc:
            runtime.state["timing_report_error"] = f"本集数据已保存，但节拍报告保存失败：{exc}"
    with runtime.lifecycle_lock:
        runtime.state["episode_count"] += 1
        runtime.state["last_episode"] = {
            "number": runtime.state["episode_count"],
            "frames": runtime.state["record_frames"],
            "elapsed_seconds": round(runtime.state["record_elapsed"], 1),
            "frame_errors": runtime.state["record_frame_errors"],
            "repo_id": runtime.state["repo_id"],
            "cameras": list(runtime.state["active_cameras"]) or camera_names[:],
            "pair_mode": runtime.state["pair_mode"],
            "follow_motion_mode": motion_mode,
            "control_frequency": frequency_setting,
            "timing": runtime.state["record_timing"],
            "control_timing": runtime.state["control_timing"],
            "timing_report": saved_report,
            "timing_report_error": runtime.state["timing_report_error"],
            "needs_review": bool(
                aborted
                or runtime.state["record_frame_errors"]
                or runtime.state["record_timing"]["warnings"]
                or runtime.state["control_timing"]["warnings"]
                or runtime.state["timing_report_error"]
            ),
            "termination_error": aborted,
        }
        needs_review = runtime.state["last_episode"]["needs_review"]
        message = f"第 {runtime.state['episode_count']} 集已保存"
        if needs_review:
            message += "，需复核本集数据；查看下方详情，检查设备后可重新准备下一集。"
        if aborted:
            message = f"本集因反馈或指令异常结束，已保存 {runtime.state['record_frames']} 帧，需复核：{aborted}。已停止全部配对的跟随指令，未执行失能。"
            message += (
                "运动操作已锁定，请按恢复提示处理。"
                if runtime.state["recovery_required"]
                else "排查设备后，可重新准备下一集。"
            )
        runtime.state.update(
            record_phase="saved_review" if needs_review else "saved",
            record_message=message,
        )
    print(f"第 {runtime.state['episode_count']} 集已保存")


@contextmanager
def _dataset_edit_session(runtime: Runtime) -> Iterator[tuple[Path, str]]:
    """仅管理平台互斥，文件格式和恢复逻辑交给独立模块。"""
    with runtime.lifecycle_lock:
        if runtime.operation_busy():
            raise ValueError("平台正在执行操作，请等待完成")
        if runtime.state["connected"] or runtime.dataset is not None:
            raise ValueError("请先结束采集并完成数据收尾，再删除上一集")
        root, repo_id = dataset_path(runtime), runtime.state["repo_id"]
        runtime.state["deleting"] = True
    try:
        dataset_transaction.recover(root)
        yield root, repo_id
    finally:
        with runtime.lifecycle_lock:
            runtime.state["deleting"] = False


def preview_delete_last(runtime: Runtime) -> tuple[dict | None, str]:
    try:
        with _dataset_edit_session(runtime) as (root, repo_id):
            target = dataset_edit.plan_delete(root).confirmation()
            return {"repo_id": repo_id, **target}, "请确认删除目标"
    except Exception as exc:
        return None, str(exc)


def delete_last_episode(runtime: Runtime, target: dict | None = None) -> tuple[bool, str]:
    """必须携带预检所得目标；旧页面、重复请求、换数据集后都不能盲删末集。"""
    try:
        with _dataset_edit_session(runtime) as (root, repo_id):
            if not isinstance(target, dict) or target.get("repo_id") != repo_id:
                return False, "删除目标缺失或数据集已变化，请重新打开确认框"
            result = dataset_edit.delete_last(root, {k: v for k, v in target.items() if k != "repo_id"})
            with runtime.lifecycle_lock:
                runtime.state.update(
                    episode_count=result["episode_count"],
                    last_episode=None,
                    record_phase="idle",
                    record_frames=0,
                    record_elapsed=0.0,
                    record_timing=None,
                    control_timing=None,
                    record_message=f"已删除第 {target['episode_index'] + 1} 集（{result['frames']} 帧）",
                )
            return True, runtime.state["record_message"]
    except Exception as exc:
        return False, str(exc)
