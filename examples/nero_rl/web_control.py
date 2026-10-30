"""Small authenticated local operator console for the standalone RL process."""

import json
import logging
import secrets
import stat
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import parse_qs, urlsplit

import cv2
from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server


class WebControls(AbstractContextManager):
    def __init__(self, host: str, port: int, token_file: Path, public_host: str = "127.0.0.1"):
        self.host, self.port = host, port
        self.token = secrets.token_urlsafe(24)
        self.token_file = token_file
        self.public_host = public_host
        # Keep the same private control link when restarting this experiment.
        # Browser cookies remain valid, while a new output directory gets a new token.
        if token_file.exists() and stat.S_IMODE(token_file.stat().st_mode) == 0o600:
            try:
                previous = urlsplit(token_file.read_text().strip())
                token = parse_qs(previous.query).get("token", [""])[0]
                if (
                    previous.scheme == "http"
                    and previous.netloc == f"{public_host}:{port}"
                    and len(token) == 32
                    and all(c.isalnum() or c in "-_" for c in token)
                ):
                    self.token = token
            except (OSError, ValueError):
                pass
        self.events: Queue[str] = Queue()
        self.abort = threading.Event()
        self.cancel = threading.Event()
        self.status = {"phase": "starting", "message": "正在准备相机和控制器"}
        self.frames: dict = {}
        self.event_log = token_file.parent / "control_events.jsonl"
        self.event_lock = threading.Lock()
        self.app = Flask(__name__)
        self._routes()

    def _record_command(self, key: str) -> None:
        """Trace accepted button presses without putting credentials in logs."""
        entry = {
            "timestamp": time.time(),
            "event": "operator_command",
            "key": key,
            "phase": self.status.get("phase"),
            "mode": self.status.get("mode"),
            "episode": self.status.get("episode"),
            "steps": self.status.get("steps"),
        }
        try:
            with self.event_lock:
                self.event_log.parent.mkdir(parents=True, exist_ok=True)
                with self.event_log.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
        except OSError:
            # Logging failure must never prevent a pause or emergency stop.
            logging.getLogger(__name__).warning("Could not record operator command %s", key, exc_info=True)
        logging.getLogger(__name__).info("Operator command: %s", entry)

    def _routes(self) -> None:
        @self.app.before_request
        def authenticate():
            token = request.args.get("token") if request.path == "/" else request.cookies.get("nero_rl")
            if not token or not secrets.compare_digest(token, self.token):
                return "需要本次实验的控制链接", 403
            if (
                request.method == "POST"
                and request.headers.get("Origin")
                and request.headers["Origin"] != request.host_url.rstrip("/")
            ):
                return "Invalid origin", 403

        @self.app.get("/")
        def index():
            html = r"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>NERO RL</title><style>body{font:18px sans-serif;max-width:1000px;margin:30px auto}
button{font-size:18px;padding:14px;margin:6px}img{width:46%;margin:1%}#state{white-space:pre-wrap}
.stop{background:#b00;color:white}</style><h1 id="title">NERO 右臂 RL 实验</h1>
<p id="task"></p>
<p>移动右主臂操作。每集先复原物体，按“开始本集”。成功需确认全部目标完成、物件放稳、夹爪松开并撤离。</p>
<p id="help">先采集多条成功示范，再尝试“策略执行”；模型会在每集结束后更新。</p>
<div><button onclick="send('n')">开始本集</button><button onclick="send('i')">主臂跟随 / 恢复</button>
<button onclick="send('h')">相对接管（无需对齐）</button>
<button onclick="send('p')">暂停跟随</button><button id="policy" onclick="send('a')">策略执行</button></div>
<div><button id="probe" onclick="send('b')" hidden>从当前位置开始限时策略测试</button></div>
<div id="master-sync-controls" hidden><button onclick="send('k')">主臂使能并原位保持</button>
<button onclick="send('s')">同步主臂到从臂</button>
策略运行或同步期间主臂会主动运动；需要接管时先点击“相对接管”，等就绪后再拖动。</div>
<div><button onclick="send('1')">成功，保存本集</button><button onclick="send('0')">失败，保存本集</button>
<button class="stop" onclick="send('x')">请求控制器停止</button></div>
<details><summary>服务管理</summary><p>完成一集请使用上方“成功 / 失败，保存本集”，保存后可继续下一集。</p>
<button onclick="send('q')">关闭采集服务（网页将断开）</button>
<p>关闭服务会把进行中的片段按未完成保存，并结束本次控制连接。</p></details>
<p>软件停止不是机械锁定；物理急停仍以设备按钮为准。</p>
<div id="state"></div><div id="images"></div><script>
let serviceClosed=false;
async function send(key){try{const r=await fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});
if(!r.ok) alert(await r.text());
else if(key==='q'){serviceClosed=true;document.getElementById('state').textContent='已请求关闭采集服务，正在保存未完成片段并结束控制连接。';}}
catch(e){document.getElementById('state').textContent='无法连接采集服务，指令是否收到尚未确认。';}}
setInterval(async()=>{try{const s=await(await fetch('/status')).json();
const phases={starting:'正在准备',aligning:'正在准备人工控制，请保持主臂不动',master_syncing:'正在同步主臂，请勿扶住',waiting:'未录制',human:'人工示范 / 纠正',policy:'策略执行',paused:'已暂停',updating:'正在训练'};
if(s.mode==='eval') phases.updating='正在保存测试结果';
if(s.mode==='record') phases.updating='正在保存示范';
document.getElementById('task').textContent=s.task||'';
document.getElementById('title').textContent=s.mode==='record'?'NERO 右臂示范采集':'NERO 右臂 RL 实验';
document.getElementById('policy').hidden=s.mode==='record';
if(s.mode==='record') document.getElementById('help').textContent='当前仅采集人工示范。保存不会训练模型，也不会自动执行策略。';
if(s.mode==='eval') document.getElementById('help').textContent='当前为独立测试：已加载模型，开始本集后可选择策略执行，结束时保存成功或失败结果。';
document.getElementById('probe').hidden=s.mode!=='eval'||!s.policy_burst_seconds;
document.getElementById('probe').disabled=s.phase!=='waiting';
document.getElementById('master-sync-controls').hidden=!s.active_master_sync;
let lines=[phases[s.phase]||s.phase,s.message||''];
if(s.checkpoint_name) lines.push('当前模型：'+s.checkpoint_name);
if(s.policy_workspace_enabled===false) lines.push('自主 XYZ 工作区限制：已关闭');
if(s.active_master_sync&&s.master_sync){lines.push('主臂同步：'+(s.master_sync.phase||'尚未准备'));
if(s.master_sync.pair_gap_deg!==undefined) lines.push('主从最大关节差：'+s.master_sync.pair_gap_deg.toFixed(2)+'°');}
if(s.control_hz_measured!==undefined) lines.push('实测跟随频率：'+s.control_hz_measured+' Hz（目标 '+s.control_hz_target+' Hz）');
if(s.tracking_error_deg!==undefined) lines.push('指令与实测最大差：'+s.tracking_error_deg+'°');
if(s.tracking_limited) lines.push('已按从臂反馈减慢目标推进，等待跟上');
if(s.workspace_limited) lines.push('目标已限制在工作范围内；往区域内移动主臂可继续跟随');
if(s.manual_control==='relative_takeover') lines.push('相对接管：从当前位置跟随主臂增量，无需手动对齐');
else if(s.human_follow==='vla_native') lines.push('人工跟随：原 VLA 直接关节跟随；RL 旁路记录');
if(s.policy_burst_seconds) lines.push('本次自主试动上限：'+s.policy_burst_seconds+' 秒，随后自动保持');
lines.push('第 '+(s.episode||1)+' 集 · 已记录 '+(s.steps||0)+' 步'+(s.mode==='record'?'':' · SAC 已更新 '+(s.training_updates||0)+' 次'));
if(s.saved_episodes!==undefined) lines.push((s.mode==='eval'?'累计测试 ':'累计保存 ')+s.saved_episodes+' 集：成功 '+s.successful_episodes+'，失败 '+s.failed_episodes);
document.getElementById('state').textContent=lines.join('\n');
const box=document.getElementById('images');for(const name of s.cameras||[]){
let im=document.getElementById('camera-'+name);if(!im){im=document.createElement('img');im.id='camera-'+name;box.appendChild(im);}
im.src='/camera/'+encodeURIComponent(name)+'?t='+Date.now();}}catch(e){
document.getElementById('state').textContent=serviceClosed?'采集服务已关闭。重新启动后可继续采集。':'采集服务连接已断开，页面将重试。当前反馈已过期，请勿据此判断机械臂状态。';}},500);
</script></html>"""
            response = Response(html, mimetype="text/html")
            response.set_cookie("nero_rl", self.token, httponly=True, samesite="Strict")
            return response

        @self.app.get("/status")
        def status():
            return jsonify(**self.status, cameras=list(self.frames))

        @self.app.get("/camera/<name>")
        def camera(name):
            frame = self.frames.get(name)
            if frame is None:
                return "No frame", 404
            ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return Response(data.tobytes(), mimetype="image/jpeg") if ok else ("Encode failed", 500)

        @self.app.post("/key")
        def key():
            key = (request.get_json(silent=True) or {}).get("key")
            if key not in ("n", "i", "h", "p", "a", "b", "s", "k", "1", "0", "q", "x"):
                return "Unknown command", 400
            phase = self.status.get("phase")
            if key in ("s", "k") and (
                not self.status.get("active_master_sync") or phase not in ("waiting", "paused", "human")
            ):
                return "请先暂停策略，再准备主臂同步", 409
            if phase == "master_syncing" and key not in ("p", "q", "x", "h"):
                return "正在同步主臂；可暂停或切换相对接管", 409
            if key == "b" and (
                phase != "waiting"
                or self.status.get("mode") != "eval"
                or not self.status.get("policy_burst_seconds")
            ):
                return "当前位置测试需要独立测试模式、空闲状态和限时时长", 409
            if key == "n" and phase != "waiting":
                return "当前不能开始新一集", 409
            if key in ("1", "0") and phase not in ("human", "policy", "paused"):
                return "当前没有进行中的示范", 409
            if key in ("i", "h", "p", "a") and phase in ("starting", "updating"):
                return "请等待当前操作完成", 409
            if key in ("n", "i", "h", "a", "1", "0") and phase == "aligning":
                return "正在对齐，请保持主臂不动；可按暂停取消", 409
            if key == "a" and self.status.get("mode") == "record":
                return "示范录制模式不执行策略", 409
            self._record_command(key)
            if key == "x":
                self.abort.set()
            if key in ("p", "q", "x", "h"):
                self.cancel.set()
            self.events.put(key)
            return jsonify(ok=True)

    def publish(self, *, frames: dict | None = None, **state) -> None:
        self.status = {**self.status, **state}
        if frames is not None:
            self.frames = frames

    def poll(self) -> list[str]:
        if self.abort.is_set():
            raise KeyboardInterrupt("Operator stop")
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except Empty:
                return result

    def __enter__(self):
        self.server = make_server(self.host, self.port, self.app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(f"http://{self.public_host}:{self.port}/?token={self.token}\n")
        self.token_file.chmod(0o600)
        print(f"Operator console: {self.token_file.read_text().strip()}", flush=True)
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
