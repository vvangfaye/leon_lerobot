"""Small HTTP layer; configuration and detached training jobs remain independent."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from . import configuration
from .jobs import Jobs

ROOT = Path(__file__).resolve().parents[3]


def create_app(*, root: Path = ROOT, state_dir: Path | None = None, jobs: Jobs | None = None,
               collection_url: str = "http://127.0.0.1:5000", inference_url: str = "http://127.0.0.1:7862") -> Flask:
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=2 * 1024 * 1024, JSON_SORT_KEYS=False)
    manager = jobs or Jobs(root, state_dir)
    app.extensions["jobs"] = manager

    @app.before_request
    def local_requests():
        if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
            return jsonify(error="仅允许本机访问"), 403
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if (request.headers.get("X-Platform-Request") != "1" or not request.is_json
                    or (origin and urlsplit(origin).netloc != request.host)):
                return jsonify(error="请从本机训练平台提交操作"), 403

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, KeyError):
            return jsonify(error=f"未找到任务或字段：{exc}"), 404
        if isinstance(exc, (ValueError, TypeError, FileNotFoundError)):
            return jsonify(error=str(exc)), 400
        if isinstance(exc, RuntimeError):
            return jsonify(error=str(exc)), 409
        app.logger.exception("Training platform request failed")
        return jsonify(error=f"操作失败：{exc}"), 500

    def body() -> dict:
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("请求内容必须是 JSON 对象")
        return value

    @app.get("/")
    def index():
        return render_template("index.html", collection_url=collection_url, inference_url=inference_url)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", root=str(root), api_version=1)

    @app.get("/api/catalog")
    def catalog():
        return jsonify(configuration.catalog())

    @app.get("/api/defaults")
    def defaults():
        return jsonify(config=configuration.default_config(request.args.get("model", "act"),
                                                            request.args.get("kind", "policy")))

    @app.get("/api/datasets")
    def datasets():
        return jsonify(datasets=configuration.discover_datasets())

    @app.post("/api/datasets/inspect")
    def dataset():
        return jsonify(dataset=configuration.inspect_dataset(body().get("path", "")))

    @app.get("/api/checkpoints")
    def checkpoints():
        return jsonify(checkpoints=configuration.discover_checkpoints(root / "outputs"))

    @app.post("/api/checkpoints/inspect")
    def checkpoint():
        return jsonify(checkpoint=configuration.inspect_checkpoint(body().get("path", "")))

    @app.post("/api/pick")
    def pick():
        payload = body()
        labels = {"dataset": "选择 LeRobot 数据集文件夹", "checkpoint": "选择 checkpoint 文件夹",
                  "model": "选择预训练模型文件夹"}
        kind = payload.get("kind", "checkpoint")
        if kind not in labels:
            raise ValueError("未知的文件夹用途")
        # Reuse the existing desktop picker without importing inference or hardware code.
        sibling = str(root / "examples" / "inference_platform")
        if sibling not in sys.path:
            sys.path.insert(0, sibling)
        from inference_platform.checkpoint_picker import DirectoryPicker

        picker = app.extensions.setdefault("directory_picker", DirectoryPicker())
        initial = root / "outputs" if kind == "checkpoint" else Path.home()
        selected = picker(initial, title=labels[kind])
        return jsonify(path=str(selected) if selected else None, cancelled=selected is None)

    @app.post("/api/validate")
    def validate():
        payload = body()
        return jsonify(configuration.validate_config(payload.get("config", {}), payload.get("mode", "new"),
                                                       payload.get("checkpoint")))

    @app.get("/api/jobs")
    def job_list():
        return jsonify(jobs=manager.list())

    @app.get("/api/jobs/<job_id>")
    def job_get(job_id):
        return jsonify(job=manager.get(job_id))

    @app.post("/api/jobs")
    def job_create():
        payload = body()
        return jsonify(job=manager.create(payload.get("config", {}), payload.get("name", ""),
                                         checkpoint=payload.get("checkpoint")))

    @app.put("/api/jobs/<job_id>")
    def job_update(job_id):
        payload = body()
        options = {"checkpoint": payload["checkpoint"]} if "checkpoint" in payload else {}
        return jsonify(job=manager.update(job_id, payload.get("config", {}), **options))

    @app.post("/api/jobs/<job_id>/start")
    def job_start(job_id):
        payload = body()
        job = manager.get(job_id)
        resume_path = payload.get("checkpoint", job.get("draft_checkpoint", job.get("resume_from")))
        config = job.get("draft_config") or job["config"]
        checked = configuration.validate_config(config, "resume" if resume_path else "new", resume_path)
        if not checked["valid"]:
            return jsonify(error="；".join(checked["errors"]), validation=checked), 400
        return jsonify(job=manager.launch(job_id, checkpoint=resume_path))

    @app.post("/api/jobs/<job_id>/stop")
    def job_stop(job_id):
        body()
        return jsonify(job=manager.stop(job_id))

    @app.post("/api/jobs/<job_id>/command")
    def job_command(job_id):
        payload = body()
        job = manager.get(job_id)
        command = [sys.executable, str(root / "examples/training_platform/run.py"),
                   "--state-dir", str(manager.state_dir), "--start", job_id]
        checkpoint = payload.get("checkpoint", job.get("draft_checkpoint", job.get("resume_from")))
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        return jsonify(command=command)

    return app
