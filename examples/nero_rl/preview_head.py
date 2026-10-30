"""Independent, authenticated head-camera preview; never imports a robot driver.

Explicit MJPEG capture avoids OpenCV's initial unsupported-format probes on
some UVC devices. The existing collection page can be embedded beneath it;
this preview does not add a camera to an already-running dataset.
"""

import argparse
import html
import logging
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from flask import Flask, Response, jsonify, request


class MJPEGFrames:
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.timestamp = 0.0
        self.count = 0
        self.error: str | None = None
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        self.buffer.extend(chunk)
        if len(self.buffer) > 8 * 1024 * 1024:
            self.buffer.clear()
            raise ValueError("Invalid or oversized MJPEG frame")
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                # Keep a possible first marker byte split across chunks.
                self.buffer[:] = self.buffer[-1:]
                return
            if start:
                del self.buffer[:start]
            end = self.buffer.find(b"\xff\xd9", 2)
            if end < 0:
                return
            frame = bytes(self.buffer[: end + 2])
            del self.buffer[: end + 2]
            with self.lock:
                self.frame, self.timestamp = frame, time.monotonic()
                self.count += 1

    def latest(self) -> tuple[bytes | None, float, int]:
        with self.lock:
            return self.frame, time.monotonic() - self.timestamp, self.count

    def consume(self, process: subprocess.Popen) -> None:
        try:
            while chunk := os.read(process.stdout.fileno(), 65536):
                self.feed(chunk)
            self.error = "顶部相机视频流已结束"
        except (OSError, ValueError) as exc:
            self.error = str(exc)


def create_app(frames: MJPEGFrames, token: str, control_url: str) -> Flask:
    app = Flask(__name__)

    @app.before_request
    def authenticate():
        supplied = request.args.get("token") if request.path == "/" else request.cookies.get("nero_head")
        if not supplied or not secrets.compare_digest(supplied, token):
            return "请使用本次采集的预览链接", 403

    @app.get("/")
    def index():
        page = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>NERO 顶部相机与采集</title>
<style>body{font:17px sans-serif;margin:20px auto;max-width:1200px}img{width:100%;max-width:960px}iframe{width:100%;height:1400px;border:1px solid #ccc}</style>
<h2>顶部相机实时预览</h2><p>顶部画面仅用于查看；当前训练数据仍记录左右腕相机。</p>
<p id="status">正在读取顶部相机…</p><img id="head" hidden alt="顶部相机">
<h2>原采集控制页</h2><p><a href="CONTROL_URL" target="_blank" rel="noopener">单独打开采集控制页</a></p>
<iframe title="右臂采集控制" src="CONTROL_URL"></iframe>
<script>setInterval(async()=>{try{const r=await fetch('/status');if(!r.ok)throw Error('连接失败');
const s=await r.json();const im=document.getElementById('head');im.hidden=!s.live;
document.getElementById('status').textContent=s.live?'实时画面 · 已接收 '+s.frames+' 帧':(s.error||'等待新画面');
if(s.live)im.src='/frame.jpg?t='+Date.now();}catch(e){document.getElementById('status').textContent='顶部预览服务连接已断开';document.getElementById('head').hidden=true;}},400);</script></html>"""
        response = Response(
            page.replace("CONTROL_URL", html.escape(control_url, quote=True)), mimetype="text/html"
        )
        response.set_cookie("nero_head", token, httponly=True, samesite="Strict")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/status")
    def status():
        frame, age, count = frames.latest()
        return jsonify(live=frame is not None and age < 1, age_s=age, frames=count, error=frames.error)

    @app.get("/frame.jpg")
    def frame():
        value, age, _ = frames.latest()
        if value is None or age >= 1:
            return "顶部相机暂无新画面", 503
        return Response(value, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--control-url-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control_url = args.control_url_file.read_text().strip()
    parsed = urlsplit(control_url)
    token = parse_qs(parsed.query).get("token", [""])[0]
    if parsed.scheme != "http" or len(token) != 32:
        parser.error("Invalid existing collection URL")
    if not args.device.exists():
        parser.error("Head camera device does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        "1280x720",
        "-framerate",
        "30",
        "-i",
        str(args.device),
        "-c:v",
        "copy",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    frames = MJPEGFrames()
    with (
        (args.output / "ffmpeg.log").open("a") as error_log,
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log) as process,
    ):
        reader = threading.Thread(target=frames.consume, args=(process,), daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 15
            while frames.latest()[0] is None:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Top camera did not produce a frame; see ffmpeg.log")
                time.sleep(0.05)
            preview_url = f"http://{args.public_host}:{args.port}/?token={token}"
            path = args.output / "preview_url.txt"
            path.write_text(preview_url + "\n")
            path.chmod(0o600)
            print(f"Head preview: {preview_url}", flush=True)
            logging.getLogger("werkzeug").setLevel(logging.WARNING)
            create_app(frames, token, control_url).run(host=args.host, port=args.port, threaded=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            reader.join(timeout=1)


if __name__ == "__main__":
    main()
