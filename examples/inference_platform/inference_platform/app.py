"""Local inference platform: configuration, device connections and live preview."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request

from .checkpoint_picker import DirectoryPicker
from .contracts import checkpoint_action_names
from .devices import DeviceManager
from .runtime import InferenceRuntime

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs"


def read_metadata(path: Path) -> dict:
    """Read a small JSON/YAML document, never weights or pickle data."""
    if path.stat().st_size > 256 * 1024:
        raise ValueError("配置文件超过 256 KB")
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("配置文件必须是对象")
    return value


def action_space_from_names(names: list[str]) -> str | None:
    # Dimension alone cannot distinguish joint commands from end-effector poses.
    if not names or not all(isinstance(name, str) for name in names):
        return None
    if all(re.fullmatch(r"(?:(?:left|right)\.)?(?:joint\d+|gripper)\.pos", name) for name in names):
        return "joint" if any("joint" in name for name in names) else None
    if all(re.fullmatch(r"(?:(?:left|right)\.)?(?:eef_[xyz]|rot6d_col[01]_[xyz]|gripper)", name) for name in names):
        return "eef" if any("eef_x" in name for name in names) else None
    return None


def inspect_checkpoint(directory: Path) -> dict:
    """Describe explicit metadata; unknown robot contracts remain unknown."""
    directory = directory.expanduser().resolve()
    result = {
        "path": str(directory), "name": directory.name,
        "model": None, "space": None, "robot": None, "arms": None,
        "action_dim": None, "joint_count": None, "cameras": [],
        "weights_present": False, "error": None,
    }
    try:
        if (directory / "config.json").is_file():
            cfg = read_metadata(directory / "config.json")
            if cfg.get("type") not in ("pi05", "fastwam"):
                raise ValueError("当前支持 π0.5 与 FastWAM")
            result["model"] = cfg["type"]
            shape = cfg.get("output_features", {}).get("action", {}).get("shape", [])
            if len(shape) == 1 and type(shape[0]) is int and shape[0] > 0:
                result["action_dim"] = shape[0]
            names = checkpoint_action_names(directory, cfg)
            if isinstance(names, list) and len(names) == result["action_dim"]:
                result["space"] = action_space_from_names(names)
                if result["space"]:
                    roles = {name.split(".")[0] for name in names if name.startswith(("left.", "right."))}
                    result["arms"] = "dual" if roles == {"left", "right"} else "single" if len(roles) <= 1 else None
                if result["space"] == "joint":
                    indices = {int(re.search(r"joint(\d+)", name).group(1)) for name in names if "joint" in name}
                    if indices == set(range(1, len(indices) + 1)):
                        result["joint_count"] = len(indices)
            result["cameras"] = [key for key in cfg.get("input_features", {}) if key.startswith("observation.images.")]
            result["weights_present"] = (directory / "model.safetensors").is_file()
            if directory.name == "pretrained_model":
                # Include the run and checkpoint step, not five identical names.
                if directory.parent.parent.name == "checkpoints":
                    result["name"] = f"{directory.parent.parent.parent.name} / {directory.parent.name}"
                else:
                    result["name"] = f"{directory.parent.parent.name} / {directory.parent.name}"
        elif (directory / "config.yaml").is_file():
            cfg = read_metadata(directory / "config.yaml")
            loader = cfg.get("dataloader", {})
            if loader.get("type") != "nero_dual_eef" or loader.get("action_mode") != "eef20":
                raise ValueError("不是当前可识别的 OpenWAM Nero 部署配置")
            result.update(model="openwam", space="eef", robot="nero", arms="dual", action_dim=20)
            result["cameras"] = loader.get("camera_layout", [])
            result["weights_present"] = any(path.is_file() for path in directory.glob("checkpoint_step_*.safetensors"))
        else:
            raise ValueError("未找到 config.json 或 config.yaml")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        result["error"] = str(exc)
    return result


def checkpoint_candidates() -> list[Path]:
    root = OUTPUTS_DIR.resolve()
    candidates = set()
    # 不跟随 outputs 下指向其他位置的目录链接，也不扫描 Downloads。
    for directory, _, filenames in os.walk(root, followlinks=False):
        if not {"config.json", "config.yaml"}.intersection(filenames):
            continue
        path = Path(directory).resolve()
        if path.is_relative_to(root):
            candidates.add(path)
    return sorted(candidates, reverse=True)


def selected_checkpoint(path: Path) -> dict:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("所选文件夹不存在，请重新选择")
    if not any((directory / name).is_file() for name in ("config.json", "config.yaml")) \
            and (directory / "pretrained_model").is_dir():
        directory /= "pretrained_model"
    item = inspect_checkpoint(directory)
    if item["error"]:
        raise ValueError(f"所选文件夹不是可识别的 checkpoint：{item['error']}")
    return item


def create_platform_app(
    *, candidates: list[Path] | None = None, devices: DeviceManager | None = None,
    directory_picker: Callable[[Path], Path | None] | None = None,
    runtime: InferenceRuntime | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.update(TRUSTED_HOSTS=["127.0.0.1", "localhost"], MAX_CONTENT_LENGTH=64 * 1024)
    manager = devices if devices is not None else DeviceManager()
    app.extensions["devices"] = manager
    engine = runtime if runtime is not None else InferenceRuntime(manager)
    app.extensions["inference"] = engine
    picker = directory_picker if directory_picker is not None else DirectoryPicker()

    @app.before_request
    def local_operations() -> Response | tuple[Response, int] | None:
        if request.method == "POST" and (
            not request.is_json or request.headers.get("X-Platform-Request") != "1"
            or request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/")
        ):
            return jsonify(error="请从本机推理平台发起操作"), 403
        return None

    @app.after_request
    def headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index() -> str:
        return render_template("platform.html")

    @app.get("/api/platform/checkpoints")
    def checkpoints() -> Response:
        paths = checkpoint_candidates() if candidates is None else candidates
        unique = dict.fromkeys(path.expanduser().resolve() for path in paths)
        return jsonify(
            checkpoints=[inspect_checkpoint(path) for path in unique],
            directory=str(OUTPUTS_DIR), selection_version=2, stage=4,
        )

    @app.post("/api/platform/checkpoints/pick")
    def pick_checkpoint() -> Response | tuple[Response, int]:
        try:
            path = picker(OUTPUTS_DIR)
            if path is None:
                return jsonify(cancelled=True)
            return jsonify(cancelled=False, checkpoint=selected_checkpoint(path))
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 409

    @app.post("/api/platform/checkpoints/inspect")
    def inspect_selected_checkpoint() -> Response | tuple[Response, int]:
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"].strip():
            return jsonify(error="缺少已选择的 checkpoint 路径"), 400
        try:
            return jsonify(checkpoint=selected_checkpoint(Path(data["path"])))
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 409

    @app.get("/api/devices")
    def device_status() -> Response:
        return jsonify(manager.status())

    @app.post("/api/devices/<action>")
    def device_action(action: str) -> Response | tuple[Response, int]:
        if action not in ("configure", "scan", "open", "disconnect", "assign", "confirm", "jog",
                          "jog-stop", "jog-heartbeat", "disable", "measure", "measure-stop"):
            return jsonify(error="未知设备操作"), 404
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify(error="操作参数必须是对象"), 400
        try:
            revision, identity = data.get("revision"), data.get("id")
            if action not in ("scan", "jog-stop", "jog-heartbeat") and not isinstance(revision, str):
                raise ValueError("缺少设备列表版本，请刷新页面")
            if action in ("open", "assign", "jog", "disable", "measure", "measure-stop") and not isinstance(identity, str):
                raise ValueError("请选择设备")
            if action == "configure":
                manager.configure(data.get("config"), revision)
            elif action == "scan":
                manager.scan(data.get("config"))
            elif action == "open":
                if data.get("config") != manager.status()["config"]:
                    raise ValueError("配置已变化，请重新检测设备")
                manager.open(identity, revision)
            elif action == "disconnect":
                if identity is not None and not isinstance(identity, str):
                    raise ValueError("设备编号无效")
                manager.disconnect(revision, identity)
            elif action == "assign":
                if not isinstance(data.get("role"), str):
                    raise ValueError("请选择设备用途")
                manager.assign(identity, data["role"], revision)
            elif action == "confirm":
                manager.confirm(revision, data.get("acknowledged"))
            elif action == "jog":
                manager.jog(identity, revision, data.get("acknowledged"))
            elif action == "jog-stop":
                manager.stop_jog()
            elif action == "jog-heartbeat":
                if not isinstance(data.get("job"), str):
                    raise ValueError("缺少微动任务编号")
                manager.heartbeat_jog(data["job"])
                return jsonify(ok=True)
            elif action == "disable":
                manager.disable_arm(identity, revision, data.get("acknowledged"))
            elif action == "measure":
                manager.measure(identity, revision)
            elif action == "measure-stop":
                manager.stop_measure(identity, revision)
            return jsonify(manager.status())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.get("/api/inference")
    def inference_status() -> Response:
        return jsonify(engine.status())

    @app.post("/api/inference/<action>")
    def inference_action(action: str) -> Response | tuple[Response, int]:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify(error="操作参数必须是对象"), 400
        try:
            if action == "stop":
                engine.stop()
            elif action == "heartbeat":
                engine.heartbeat(str(data.get("job", "")))
                return jsonify(ok=True)
            elif action == "load":
                engine.load(data.get("settings", {}), data.get("revision"))
            elif action == "unload":
                engine.unload(data.get("revision"))
            elif action in ("single", "continuous", "home"):
                engine.start(action, data.get("settings", {}), data.get("revision"))
            else:
                return jsonify(error="未知推理操作"), 404
            return jsonify(engine.status())
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 409

    @app.get("/api/devices/frame")
    def device_frame() -> Response:
        frame = manager.frame(request.args.get("id", ""), request.args.get("revision", ""))
        if frame is None:
            return Response(status=204)
        return Response(frame, mimetype="image/jpeg")

    return app
