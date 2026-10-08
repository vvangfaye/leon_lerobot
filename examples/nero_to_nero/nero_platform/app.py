"""Flask 界面与接口。create_app 只组装应用，不连接硬件。"""

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, request

from . import (
    camera_focus,
    camera_io,
    camera_preview,
    cameras,
    config,
    depth_images,
    devices,
    hardware_status,
    recorder,
    review,
)
from .camera_model import is_depth, stream_names, stream_role
from .runtime import ARM_LABELS, CONFIG_FIELDS, EXAMPLE_DIR, Runtime, down_can_interfaces


def create_app(runtime: Runtime) -> Flask:
    app = Flask(__name__)
    app.extensions["nero_runtime"] = runtime
    app.register_blueprint(review.create_blueprint(runtime))

    @app.before_request
    def reject_commands_during_shutdown() -> Any:
        if request.method == "POST" and runtime.state["shutting_down"]:
            return jsonify(ok=False, msg="程序正在退出，请等待保存收尾完成"), 409
        if request.method == "POST" and runtime.state["config_error"]:
            return jsonify(ok=False, msg=runtime.state["config_error"]), 409
        return None

    @app.get("/")
    def index() -> str:
        directory = Path(__file__).parent
        manifest = json.loads((directory / "static/voice/prompts.json").read_text(encoding="utf-8"))
        return (
            (directory / "index.html")
            .read_text(encoding="utf-8")
            .replace("/*VOICE_MANIFEST*/{}", json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/"))
        )

    def device_command(ch: str, command: Any) -> Any:
        if ch not in runtime.channels:
            return jsonify(ok=False, msg="未知 CAN 通道"), 404
        with runtime.lifecycle_lock:
            if runtime.operation_busy(channel=ch):
                return jsonify(ok=False, msg="机械臂正在执行操作，请等待完成"), 409
            if runtime.state["recovery_required"] and command is not devices.connect_channel:
                return jsonify(ok=False, msg=runtime.state["recovery_required"]), 409
            if runtime.state["disable_error"] or runtime.state["close_error"]:
                return jsonify(ok=False, msg="请先处理结束结果中未完成的数据或失能项"), 409
            if runtime.state["connected"]:
                return jsonify(ok=False, msg="请先结束采集，再进行设备配置"), 409
            runtime.device_busy.add(ch)
            runtime.state["disable_status"] = ""

            def run_device_command() -> None:
                try:
                    command(runtime, ch)
                finally:
                    with runtime.lifecycle_lock:
                        runtime.device_busy.discard(ch)

            try:
                threading.Thread(target=run_device_command, daemon=True).start()
            except Exception:
                runtime.device_busy.discard(ch)
                raise
        return jsonify(ok=True)

    @app.post("/api/can/redetect")
    def api_can_redetect() -> Any:
        with runtime.lifecycle_lock:
            if runtime.operation_busy():
                return jsonify(ok=False, msg="当前有操作进行中，请等待完成"), 409
            if runtime.state["recovery_required"]:
                return jsonify(ok=False, msg=runtime.state["recovery_required"]), 409
            if runtime.state["connected"]:
                return jsonify(ok=False, msg="请先结束采集，再重新检测 CAN"), 409
            channels = devices.redetect_channels(runtime)
        return jsonify(ok=True, channels=channels)

    @app.post("/api/arm/connect/<ch>")
    def api_arm_connect(ch: str) -> Any:
        return device_command(ch, devices.connect_channel)

    @app.post("/api/arm/move/<ch>")
    def api_arm_move(ch: str) -> Any:
        return device_command(ch, devices.micro_move)

    @app.post("/api/arm/disable/<ch>")
    def api_arm_disable(ch: str) -> Any:
        return device_command(ch, devices.home_and_disable)

    @app.post("/api/pair/home")
    def api_pair_home() -> Any:
        data = request.get_json(force=True)
        ok, msg = devices.start_return_home(runtime, data.get("targets"))
        return jsonify(ok=ok, msg=msg), 202 if ok else 409

    @app.get("/api/cameras")
    def api_cameras() -> Response:
        # 首次加载初始化一次；刷新页面读取缓存，检测按钮才重新枚举。
        with runtime.lifecycle_lock:
            if (
                not runtime.cameras_scanned
                and not runtime.state["connected"]
                and not runtime.operation_busy()
            ):
                cameras.scan_inventory(runtime)
        return jsonify(cameras.inventory_payload(runtime))

    @app.post("/api/cameras/detect/<kind>")
    def api_cameras_detect(kind: str) -> Any:
        if kind not in ("rgb", "depth"):
            return jsonify(ok=False, msg="检测类型必须为 rgb 或 depth"), 400
        with runtime.lifecycle_lock:
            if runtime.operation_busy():
                return jsonify(ok=False, msg="当前有操作进行中，请等待完成"), 409
            if runtime.state["connected"]:
                return jsonify(ok=False, msg="请先结束采集，再检测相机"), 409
            runtime.device_busy.add("camera_detection")
        try:
            cameras.close_preview_cams(runtime, kind=kind)
            cameras.scan_inventory(runtime, kind)
            if kind == "rgb":
                cameras.open_preview_cams(runtime)
            return jsonify(ok=True, **cameras.inventory_payload(runtime))
        except Exception as exc:
            return jsonify(ok=False, msg=f"相机检测失败：{exc}"), 503
        finally:
            with runtime.lifecycle_lock:
                runtime.device_busy.discard("camera_detection")

    @app.get("/api/camera/resolutions/<role>")
    def camera_resolutions(role: str) -> Any:
        if role not in ("left", "right", "head"):
            return jsonify(ok=False, msg="未知相机角色"), 404
        with runtime.lifecycle_lock:
            if runtime.state["connected"] or runtime.operation_busy():
                return jsonify(ok=False, msg="请先结束采集并等待设备空闲，再读取分辨率"), 409
            runtime.device_busy.add("camera_profiles")
        try:
            camera_id = request.args.get("camera_id", "")
            detected = cameras.detect_all_cameras()
            if camera_id not in detected:
                raise ValueError("请先在左侧为此角色分配已检测到的相机")
            info = detected[camera_id]
            if info["type"] == "rgbd":
                cameras.close_preview_cams(runtime, cam_id=camera_id)
            result = camera_io.backend(info).profiles(info)
            return jsonify(ok=True, **result)
        except Exception as exc:
            return jsonify(ok=False, msg=f"读取分辨率失败：{exc}"), 503
        finally:
            with runtime.lifecycle_lock:
                runtime.device_busy.discard("camera_profiles")

    @app.post("/api/camera/<cam_id>/preview/start")
    def start_camera_preview(cam_id: str) -> Any:
        restart = (request.get_json(silent=True) or {}).get("restart") is True
        with runtime.lifecycle_lock:
            info = runtime.camera_inventory.get(cam_id)
            if not info:
                return jsonify(ok=False, msg="未检测到此相机，请重新检测"), 404
            if info["type"] != "rgbd" or not camera_preview.supported(info):
                return jsonify(ok=False, msg="此设备目前仅支持检测，尚未接通 RGBD 预览"), 400
            if runtime.state["connected"]:
                return jsonify(ok=False, msg="会话已锁定，请在采集区查看画面"), 409
            state = runtime.preview_states.get(cam_id, {})
            if state.get("phase") == "opening" or (state.get("phase") == "ready" and not restart):
                return jsonify(ok=True, **state)
            if runtime.operation_busy():
                return jsonify(ok=False, msg="当前有操作进行中，请等待完成"), 409
            busy_key = "camera_preview:" + cam_id
            runtime.device_busy.add(busy_key)
            runtime.preview_states[cam_id] = {"phase": "opening", "message": "正在打开 RGB / 深度预览…"}
        try:
            camera_preview.open_depth(runtime, cam_id, info)
            return jsonify(ok=True, **runtime.preview_states[cam_id])
        except Exception as exc:
            message = f"RGBD 预览失败：{exc}"
            runtime.preview_states[cam_id] = {"phase": "error", "message": message}
            return jsonify(ok=False, msg=message), 503
        finally:
            with runtime.lifecycle_lock:
                runtime.device_busy.discard(busy_key)

    @app.post("/api/camera/<cam_id>/focus")
    def apply_camera_focus(cam_id: str) -> Any:
        setting = request.get_json(silent=True)
        try:
            camera_focus.validate_setting(setting)
        except ValueError as exc:
            return jsonify(ok=False, msg=str(exc)), 400
        with runtime.lifecycle_lock:
            if runtime.state["connected"] or runtime.operation_busy():
                return jsonify(ok=False, msg="请先结束采集并等待设备空闲，再调整对焦"), 409
            info = runtime.camera_inventory.get(cam_id)
            if not info or info.get("backend") != "opencv" or cam_id not in runtime.preview_cams:
                return jsonify(ok=False, msg="此相机未打开 RGB 预览，或未接通对焦控制"), 400
            key = "camera_focus:" + cam_id
            runtime.device_busy.add(key)
        try:
            # 复用已打开预览的设备控制接口，不另开视频流。
            result = camera_focus.configure_focus(info["path"], setting)
            runtime.camera_focus[cam_id] = result
            if result["state"] != "verified":
                return jsonify(ok=False, msg=result["message"], focus=result), 400
            with runtime.lifecycle_lock:
                settings = {**runtime.state["camera_focus_settings"], cam_id: dict(setting)}
                config.save_config(runtime, runtime.camera_inventory, {"camera_focus_settings": settings})
            return jsonify(ok=True, focus=result, setting=setting)
        except (OSError, ValueError) as exc:
            return jsonify(ok=False, msg=f"对焦设置未保存：{exc}，请重试"), 503
        finally:
            with runtime.lifecycle_lock:
                runtime.device_busy.discard(key)

    def jpeg_response(frame: Any) -> Response:
        # JPEG 编码只在浏览器请求线程执行，保持数据集图像为 RGB。
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            return Response("preview encoding failed", status=503, headers={"Cache-Control": "no-store"})
        return Response(jpeg.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/<cam_id>/frame")
    def camera_frame(cam_id: str) -> Response:
        cam = runtime.preview_cams.get(cam_id)
        if cam is None:
            return Response("camera not open", status=404, headers={"Cache-Control": "no-store"})
        try:
            return jpeg_response(cam.read_latest())
        except Exception:
            return Response("camera frame unavailable", status=503, headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/<cam_id>/rgbd/frame")
    def camera_rgbd_frame(cam_id: str) -> Response:
        cam = runtime.preview_cams.get(cam_id)
        if cam is None or runtime.preview_states.get(cam_id, {}).get("phase") != "ready":
            return Response("camera not ready", status=404, headers={"Cache-Control": "no-store"})
        try:
            return jpeg_response(camera_preview.paired_frame(cam))
        except Exception:
            return Response("RGBD frame unavailable", status=503, headers={"Cache-Control": "no-store"})

    @app.get("/api/record/camera/<name>/frame")
    def recording_camera_frame(name: str) -> Response:
        role = stream_role(name)
        info = runtime.state["camera_sources"].get(role, {"type": "rgb"})
        cam = runtime.collection_cams.get(role)
        if cam is not None and name not in stream_names(role, info["type"]):
            cam = None
        if cam is None or not runtime.state["connected"]:
            return Response("camera not open", status=404, headers={"Cache-Control": "no-store"})
        try:
            if runtime.state["record_phase"] in ("recording", "stopping"):
                # 录制时只展示已提交的采样帧，绝不额外读取或重新打开相机。
                cached = runtime.record_preview.get(name)
                if cached is None or cached[0] is not cam or time.monotonic() - cached[2] > 1.0:
                    raise RuntimeError("waiting for fresh recorded frame")
                frame = cached[1]
            else:
                frame = camera_io.backend(info).preview(cam, is_depth(name))
            return jpeg_response(depth_images.depth_preview(frame) if is_depth(name) else frame)
        except Exception:
            return Response("camera frame unavailable", status=503, headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    def api_config() -> Response:
        return jsonify(config.settings(runtime, runtime.camera_inventory))

    @app.get("/api/status")
    def api_status() -> Response:
        capture = recorder.recording_status(runtime)
        return jsonify(
            arms={
                ch: {
                    "status": runtime.channels[ch]["status"],
                    "label": runtime.channels[ch]["label"],
                    "detected": runtime.channels[ch]["detected"],
                    "hardware": hardware_status.read_status(runtime, ch),
                }
                for ch in runtime.channels
            },
            arm_count=devices.arm_count(runtime),
            can_down=down_can_interfaces(),
            connected=runtime.state["connected"],
            recording=capture["busy"],
            record=capture,
            episode_count=capture["episode_count"],
            leader_label=runtime.state["leader_label"],
            follower_label=runtime.state["follower_label"],
            pair_mode=runtime.state["pair_mode"],
            last_error=runtime.state["last_error"],
            config_error=runtime.state["config_error"],
            camera_config_notice=runtime.state["camera_config_notice"],
            recovery_required=runtime.state["recovery_required"],
            homing=runtime.state["homing"],
            home_status=runtime.state["home_status"],
            home_error=runtime.state["home_error"],
            disabling=runtime.state["disabling"],
            ending=runtime.state["ending"],
            end_status=runtime.state["end_status"],
            disable_status=runtime.state["disable_status"],
            disable_error=runtime.state["disable_error"],
            close_error=runtime.state["close_error"],
            close_result=runtime.state["close_result"],
            active_cameras=runtime.state["active_cameras"],
            camera_resolution=runtime.state["camera_resolution"],
            camera_devices={
                role: {**runtime.state["camera_sources"].get(role, {}), **getattr(cam, "metadata", {})}
                for role, cam in list(runtime.collection_cams.items())
            },
            device_busy=sorted(runtime.device_busy),
            camera_previews=dict(runtime.preview_states),
            camera_focus=dict(runtime.camera_focus),
            camera_focus_settings=dict(runtime.state["camera_focus_settings"]),
        )

    @app.get("/api/arm/status/<ch>")
    def api_arm_status(ch: str) -> Any:
        if ch not in runtime.channels:
            return jsonify(ok=False, msg="未知 CAN 通道"), 404
        return jsonify(ok=True, hardware=hardware_status.inspect_status(runtime, ch))

    @app.post("/api/label/<ch>")
    def api_label(ch: str) -> Any:
        if ch not in runtime.channels:
            return jsonify(ok=False), 404
        with runtime.lifecycle_lock:
            if runtime.operation_busy():
                return jsonify(ok=False, msg="机械臂正在执行操作，请稍后修改配置"), 409
            if runtime.state["disable_error"] or runtime.state["close_error"]:
                return jsonify(ok=False, msg="请先处理结束结果中的未完成项，再修改配置"), 409
            if runtime.state["connected"]:
                return jsonify(ok=False, msg="请先结束采集，再修改机械臂角色"), 409
            label = request.get_json(force=True).get("label")
            if label in ARM_LABELS:
                runtime.channels[ch]["label"] = label
        return jsonify(ok=True, label=runtime.channels[ch]["label"])

    @app.post("/api/save")
    def api_save() -> Any:
        data = request.get_json(force=True)
        with runtime.lifecycle_lock:
            try:
                config.validate_settings(runtime, data)
            except ValueError as exc:
                return jsonify(ok=False, msg=str(exc)), 400
            if runtime.operation_busy():
                return jsonify(ok=False, msg="机械臂正在执行操作，请稍后修改配置"), 409
            if runtime.state["disable_error"] or runtime.state["close_error"]:
                return jsonify(ok=False, msg="请先处理结束结果中的未完成项，再修改配置"), 409
            if runtime.state["connected"]:
                changed = any(
                    data.get(key, runtime.state[key]) != runtime.state[key] for key in CONFIG_FIELDS
                )
                changed |= data.get("camera_map", runtime.state["camera_map"]) != runtime.state["camera_map"]
                changed |= any(
                    label != runtime.channels[ch]["label"]
                    for ch, label in data.get("labels", {}).items()
                    if ch in runtime.channels
                )
                if changed:
                    return jsonify(ok=False, msg="请先结束采集，再修改配置"), 409
            try:
                cfg = config.save_config(runtime, runtime.camera_inventory, data=data)
            except (ValueError, OSError) as exc:
                return jsonify(ok=False, msg=str(exc)), 400
        return jsonify(ok=True, path=str(runtime.config_path), config=cfg)

    @app.post("/api/connect")
    def api_connect() -> Response:
        with runtime.lifecycle_lock:
            if runtime.operation_busy():
                return jsonify(ok=False, msg="机械臂正在执行操作，请等待完成")
            try:
                config.save_config(runtime, runtime.camera_inventory)
            except (ValueError, OSError) as exc:
                return jsonify(ok=False, msg=f"配置未保存：{exc}")
            ok, msg = recorder.do_connect(runtime)
        return jsonify(ok=ok, msg=msg)

    @app.post("/api/disconnect")
    def api_disconnect() -> Any:
        data = request.get_json(silent=True) or {}
        ok, msg = recorder.end_collection_and_disable(runtime, data.get("targets"))
        return jsonify(ok=ok, msg=msg), 200 if ok else (
            500 if runtime.state["disable_error"] or runtime.state["close_error"] else 409
        )

    @app.post("/api/collection/end")
    def api_collection_end() -> Any:
        data = request.get_json(silent=True) or {}
        ok, msg = recorder.end_collection_and_disable(runtime, data.get("targets"), operation="end")
        return jsonify(ok=ok, msg=msg), 200 if ok else 409

    @app.post("/api/pair/disable")
    def api_pair_disable() -> Any:
        data = request.get_json(silent=True) or {}
        ok, msg = recorder.end_collection_and_disable(runtime, data.get("targets"), operation="disable")
        return jsonify(ok=ok, msg=msg), 200 if ok else 409

    @app.post("/api/close/retry")
    def api_close_retry() -> Any:
        data = request.get_json(silent=True) or {}
        ok, msg = recorder.end_collection_and_disable(
            runtime, data.get("targets"), retry_stage=data.get("stage", ""), channel=data.get("channel")
        )
        return jsonify(ok=ok, msg=msg), 200 if ok else 409

    @app.post("/api/record/start")
    def api_record_start() -> Any:
        ok, msg = recorder.start_recording(runtime)
        return jsonify(ok=ok, msg=msg), 202 if ok else 409

    @app.post("/api/record/stop")
    def api_record_stop() -> Response:
        ok, msg = recorder.request_stop(runtime)
        return jsonify(ok=ok, msg=msg)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Nero 数据采集工作台")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "collect_config.platform.json")
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / ".cache/huggingface/lerobot")
    parser.add_argument("--check", action="store_true", help="仅检查导入、配置和网页接口，不连接硬件")
    args = parser.parse_args()
    runtime = Runtime(config_path=args.config.resolve(), dataset_root=args.dataset_root.resolve())
    config.load_saved(runtime, legacy_path=EXAMPLE_DIR / "collect_config.json")
    if runtime.state["config_error"]:
        print(runtime.state["config_error"])
        if args.check:
            parser.exit(2, "软件自检失败：请先修复配置文件。\n")
    app = create_app(runtime)
    if args.check:
        with app.test_client() as client:
            for route in (
                "/",
                "/api/config",
                "/api/status",
                "/review",
                "/static/review.js",
                "/api/review/datasets",
            ):
                response = client.get(route)
                if response.status_code != 200:
                    raise RuntimeError(f"接口自检失败: {route} ({response.status_code})")
        print("软件自检通过：模块导入、配置读取、采集/回放网页和只读接口；未连接机械臂或相机。")
        print(f"配置文件: {runtime.config_path}")
        print(f"数据目录: {recorder.dataset_path(runtime)}")
        return
    print("Nero 数据采集工作台（单对 / 左右双对，共同采样时间轴）")
    print(f"浏览器打开: http://127.0.0.1:{args.port}")
    print(f"配置文件: {runtime.config_path}")
    print(f"数据目录: {recorder.dataset_path(runtime)}")
    try:
        if not runtime.state["config_error"]:
            cameras.open_preview_cams(runtime)
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        print("等待录制线程保存并收尾...")
        with runtime.lifecycle_lock:
            runtime.state["shutting_down"] = True
        try:
            if runtime.home_thread is not None:
                runtime.home_thread.join()
            while runtime.state["disabling"] or runtime.state["ending"]:
                time.sleep(0.05)
            # 工作线程的状态更新需要 lifecycle_lock；不能持锁等待它退出。
            recorder.do_disconnect(runtime, reopen_preview=False)
        finally:
            cameras.close_preview_cams(runtime)
