"""Small persistent process supervisor; imports no model or GPU libraries."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .configuration import inspect_checkpoint, inspect_dataset

ACTIVE = {"running", "stopping", "starting"}
_UNSET = object()
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC = re.compile(rf"([a-zA-Z_][\w/]*):({NUMBER})([KMB]?)\b")
PROGRESS = re.compile(r"Training:.*?\|\s*(\d+)/(\d+)\s*\[")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


def cli_values(argv: list[str]) -> dict[str, Any]:
    values = {}
    for i, arg in enumerate(argv):
        if not arg.startswith("--"):
            continue
        key, sep, raw = arg[2:].partition("=")
        if not sep:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                continue
            raw = argv[i + 1]
        try:
            values[key] = json.loads(raw)
        except ValueError:
            values[key] = raw
    return values


def set_nested(config: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(config.get(part), dict):
            config[part] = {}
        config = config[part]
    config[parts[-1]] = value


def checkpoint_info(path: Path) -> dict:
    info = inspect_checkpoint(str(path))
    return {
        "path": info["checkpoint_path"],
        "step": int(info.get("step") or 0),
        "config_path": info["train_config_path"],
        "resumable": info["resumable"],
        "resume_errors": info["resume_errors"],
        "modified_at": info["modified_at"],
    }


class LogReader:
    """Incremental, bounded log tail and scalar history; understands carriage-return tqdm."""

    def __init__(self) -> None:
        self.offset = 0
        self.pending = b""
        self.history: list[dict] = []
        self.logs: list[str] = []
        self.metrics: dict[str, float] = {}
        self.step: int | None = None
        self.num_frames: int | None = None
        self.num_episodes: int | None = None
        self.last_progress_at: float | None = None
        self.finished = False

    def read(self, path: Path, record: dict) -> None:
        try:
            size = path.stat().st_size
            if size < self.offset:
                self.__init__()
            # First read limited to 8 MiB, subsequent reads bounded similarly.
            if self.offset == 0 and size > 8 * 1024 * 1024:
                with path.open("rb") as header:
                    for raw in re.split(rb"[\r\n]", header.read(256 * 1024))[:-1]:
                        self.parse(raw.decode("utf-8", errors="replace"), record)
                self.offset = size - 8 * 1024 * 1024
            with path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read(8 * 1024 * 1024)
                self.offset = stream.tell()
        except OSError:
            return
        parts = re.split(rb"[\r\n]", self.pending + data)
        self.pending = parts.pop()[-65536:]
        # tqdm often leaves its latest line unterminated. Parse it for progress only.
        for raw in parts:
            self.parse(raw.decode("utf-8", errors="replace"), record)
        if self.pending:
            self.parse(self.pending.decode("utf-8", errors="replace"), record, progress_only=True)

    def parse(self, line: str, record: dict, progress_only: bool = False) -> None:
        line = ANSI.sub("", line).strip()
        match = PROGRESS.search(line)
        if match:
            current, remaining = map(int, match.groups())
            # tqdm is local to this resumed invocation; its total determines exact offset.
            total = int(record.get("config", {}).get("steps") or 0)
            offset = total - remaining if total >= remaining else int(record.get("start_step", 0))
            self.step = offset + current
            self.last_progress_at = time.time()
        if progress_only:
            return
        for key in ("num_frames", "num_episodes"):
            found = re.search(rf"dataset\.{key}=(\d+)", line)
            if found:
                setattr(self, key, int(found.group(1)))
        if "End of training" in line:
            self.finished = True
        if " loss:" in line and "step:" in line:
            values = {
                key: float(value) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[suffix]
                for key, value, suffix in METRIC.findall(line[line.index("step:") :])
            }
            # Prefer the preceding exact tqdm step, never promote rounded '31K' to exact.
            step_match = re.search(r"\bstep:(\d+)([KMB]?)\b", line)
            exact = int(step_match.group(1)) if step_match and not step_match.group(2) else self.step
            if exact is not None:
                self.step = exact
                row = {
                    "step": exact,
                    **{k: v for k, v in values.items() if k not in {"step", "smpl", "ep", "epch"}},
                }
                self.metrics.update({k: v for k, v in row.items() if k != "step"})
                if not self.history or self.history[-1]["step"] != exact:
                    self.history.append(row)
                    self.history = self.history[-3000:]
        evaluation = re.search(rf"step (\d+): eval_loss=({NUMBER})", line)
        if evaluation:
            self.history.append({"step": int(evaluation.group(1)), "eval_loss": float(evaluation.group(2))})
            self.metrics["eval_loss"] = float(evaluation.group(2))
        if line and (not line.startswith("Training:") or "INFO " in line or "ERROR " in line):
            if "INFO " in line and line.startswith("Training:"):
                line = line[line.index("INFO ") :]
            self.logs.append(line[-4000:])
            self.logs = self.logs[-250:]


class Jobs:
    def __init__(self, root: Path, state_dir: Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.state_dir = Path(state_dir or self.root / "outputs/training_platform").resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {p.parent.name: read_json(p, {}) for p in self.state_dir.glob("*/job.json")}
        self.readers: dict[str, LogReader] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.last_discovery = 0.0
        self.dataset_metadata: dict[str, dict] = {}
        boot = re.search(r"^btime (\d+)$", Path("/proc/stat").read_text(), re.MULTILINE)
        self.boot_time = int(boot.group(1)) if boot else time.time()

    def _save(self, record: dict) -> None:
        write_json(self.state_dir / record["id"] / "job.json", record)
        self.records[record["id"]] = record

    @contextmanager
    def _launch_lock(self):
        # All UI instances sharing this project serialize launch + PID publication.
        with (self.state_dir / ".launch.lock").open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _reload(self) -> None:
        for path in self.state_dir.glob("*/job.json"):
            record = read_json(path, {})
            if record.get("id") == path.parent.name:
                self.records[record["id"]] = record

    def _discover(self) -> None:
        if time.monotonic() - self.last_discovery < 3:
            return
        self.last_discovery = time.monotonic()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = [s.decode() for s in (entry / "cmdline").read_bytes().split(b"\0") if s]
                # Require the executable/module argument, not a shell script mentioning a trainer.
                if not any(
                    Path(a).name == "lerobot-train" or a == "lerobot.scripts.lerobot_train" for a in argv[:4]
                ):
                    continue
                identity = process_identity(int(entry.name))
                if not identity:
                    continue
                identifier = f"external-{entry.name}-{identity}"
                started_at = self.boot_time + int(identity) / os.sysconf("SC_CLK_TCK")
                if identifier in self.records:
                    if not self.records[identifier].get("started_at"):
                        self.records[identifier]["started_at"] = started_at
                        self._save(self.records[identifier])
                    continue
                args = cli_values(argv)
                config_path = Path(str(args.pop("config_path", ""))).expanduser()
                cwd = Path(os.readlink(entry / "cwd"))
                if not config_path.is_absolute():
                    config_path = cwd / config_path
                if config_path.is_dir():
                    config_path /= "train_config.json"
                config = read_json(config_path, {})
                for key, value in args.items():
                    set_nested(config, key, value)
                log_path = os.readlink(entry / "fd/1")
                if not Path(log_path).is_file():
                    log_path = ""
                output = Path(str(config.get("output_dir") or ""))
                if not output.is_absolute():
                    output = cwd / output
                start = checkpoint_info(config_path).get("step", 0) if config.get("resume") else 0
                self._save(
                    {
                        "id": identifier,
                        "name": config.get("job_name") or output.name,
                        "managed": False,
                        "readonly": True,
                        "status": "running",
                        "pid": int(entry.name),
                        "process_identity": identity,
                        "created_at": time.time(),
                        "started_at": started_at,
                        "config": config,
                        "output_dir": str(output),
                        "log_path": log_path,
                        "start_step": start,
                        "resume_from": str(config_path) if config.get("resume") else None,
                        "command": argv,
                    }
                )
            except (OSError, ValueError, TypeError):
                continue

    def _status(self, record: dict, detail: bool = True) -> dict:
        identifier = record["id"]
        process = self.processes.get(identifier)
        if process is not None:
            process.poll()  # Reap exited managed children.
        alive = bool(record.get("pid") and process_identity(record["pid"]) == record.get("process_identity"))
        outcome = read_json(self.state_dir / identifier / "result.json", {}) if record.get("managed") else {}
        starting = record["status"] == "starting" and time.time() - record.get("started_at", 0) < 30
        if record["status"] in ACTIVE and not alive and not starting:
            record.update(
                status=outcome.get("status", "exited"), ended_at=outcome.get("ended_at", time.time())
            )
            if outcome.get("error"):
                record["error"] = outcome["error"]
            self._save(record)
        result = copy.deepcopy(record)
        reader = self.readers.setdefault(identifier, LogReader())
        if record.get("log_path"):
            reader.read(Path(record["log_path"]), record)
        config = record.get("config", {})
        dataset = config.get("dataset") or {}
        source = dataset.get("root") or dataset.get("repo_id")
        source_key = json.dumps(source)
        if source_key not in self.dataset_metadata:
            try:
                self.dataset_metadata[source_key] = inspect_dataset(str(source)) if source else {}
            except (OSError, ValueError, TypeError):
                self.dataset_metadata[source_key] = {}
        meta = self.dataset_metadata[source_key]
        subset = dataset.get("episodes") or dataset.get("exclude_episodes") or dataset.get("eval_split")
        frames = reader.num_frames or (meta.get("total_frames") if not subset else None)
        episodes = reader.num_episodes or (meta.get("total_episodes") if not subset else None)
        step = reader.step if reader.step is not None else record.get("start_step", 0)
        total = int(config.get("steps") or 0)
        topology = config.get("parallelism") or {}
        batch = (
            int(config.get("batch_size") or 1)
            * int(topology.get("dp_replicate", 1))
            * int(topology.get("dp_shard", 1))
        )
        metrics = reader.metrics.copy()
        speed = 1 / metrics["step_s"] if metrics.get("step_s", 0) > 0 else None
        result.update(
            step=step,
            total_steps=total,
            progress=min(step / total, 1) if total else 0,
            epoch=step * batch / frames if frames else None,
            num_frames=frames,
            num_episodes=episodes,
            metrics=metrics,
            steps_per_second=speed,
            eta_seconds=(total - step) / speed if speed else None,
            last_log_at=Path(record["log_path"]).stat().st_mtime
            if record.get("log_path") and Path(record["log_path"]).exists()
            else None,
        )
        if not record.get("managed") and not alive and reader.finished:
            result["status"] = "completed"
        if detail:
            result.update(history=reader.history, logs=reader.logs, checkpoints=[])
            output = Path(record.get("output_dir") or "/nonexistent")
            for checkpoint in sorted((output / "checkpoints").glob("*/pretrained_model/train_config.json")):
                if checkpoint.parent.parent.is_symlink():
                    continue
                try:
                    result["checkpoints"].append(checkpoint_info(checkpoint))
                except (OSError, ValueError):
                    continue  # A native checkpoint may still be being written.
        return result

    def list(self) -> list[dict]:
        with self.lock:
            self._reload()
            self._discover()
            return sorted(
                [self._status(r, detail=False) for r in self.records.values()],
                key=lambda r: r.get("created_at", 0),
                reverse=True,
            )

    def get(self, identifier: str) -> dict:
        with self.lock:
            self._reload()
            self._discover()
            if identifier not in self.records:
                raise ValueError("训练任务不存在")
            return self._status(self.records[identifier])

    def create(self, config: dict, name: str = "", checkpoint: str | None = None) -> dict:
        if not isinstance(config, dict):
            raise ValueError("配置必须是 JSON 对象")
        with self.lock:
            identifier = uuid.uuid4().hex[:12]
            record = {
                "id": identifier,
                "name": name.strip() or config.get("job_name") or f"训练 {identifier}",
                "managed": True,
                "readonly": False,
                "status": "draft",
                "config": copy.deepcopy(config),
                "draft_checkpoint": checkpoint or None,
                "created_at": time.time(),
                "start_step": 0,
            }
            self._save(record)
            return self._status(record)

    def update(self, identifier: str, config: dict, checkpoint: Any = _UNSET) -> dict:
        if not isinstance(config, dict):
            raise ValueError("配置必须是 JSON 对象")
        with self.lock:
            self.get(identifier)
            record = self.records[identifier]
            field = "config" if record["status"] == "draft" else "draft_config"
            record[field] = copy.deepcopy(config)
            if checkpoint is not _UNSET:
                record["draft_checkpoint"] = checkpoint or None
            self._save(record)
            return self._status(record)

    def launch(self, identifier: str, checkpoint: Any = _UNSET) -> dict:
        with self.lock, self._launch_lock():
            current = self.get(identifier)
            if checkpoint is _UNSET:
                checkpoint = current.get("draft_checkpoint")
            if any(r["status"] in ACTIVE for r in self.list()):
                raise ValueError(
                    "已有训练正在运行。为避免抢占显存，请等待其结束后再启动；终端训练仅只读监控。"
                )
            config = copy.deepcopy(current.get("draft_config") or current["config"])
            if (config.get("job") or {}).get("target", "local") not in {None, "local"}:
                raise ValueError("平台启动仅支持本地训练；远程任务请使用原生 LeRobot 命令。")
            parallel = config.get("parallelism") or {}
            degrees = [parallel.get("dp_replicate", 1), parallel.get("dp_shard", 1)]
            degrees += list((parallel.get("context_parallel") or {}).values())
            if (
                any(value not in {None, 1} for value in degrees)
                or int(os.environ.get("WORLD_SIZE", "1")) != 1
            ):
                raise ValueError("平台托管仅支持单进程训练；分布式配置可编辑，请用 torchrun 启动。")
            if config.get("save_checkpoint") is False:
                raise ValueError("平台的保存并停止需要 save_checkpoint=true。")
            info = checkpoint_info(Path(checkpoint)) if checkpoint else None
            if info and not info["resumable"]:
                raise ValueError("所选 checkpoint 缺少完整训练状态，不能断点续训；可将权重作为新训练起点。")
            if info and int(config.get("steps", 0)) <= info["step"]:
                raise ValueError("目标总步数必须大于 checkpoint 已完成的步数。")
            if current["status"] != "draft" or checkpoint:
                created = self.create(
                    config, current["name"] + (" · 续训" if checkpoint else " · 新训练"), checkpoint
                )
                identifier = created["id"]
            record = self.records[identifier]
            output = Path(
                str(config.get("output_dir") or self.root / "outputs/train" / identifier)
            ).expanduser()
            if not output.is_absolute():
                output = self.root / output
            if info or current["status"] != "draft":
                output = output.parent / f"{output.name}-{identifier}"
            if output.exists():
                raise ValueError(f"输出目录已存在，不能覆盖：{output}；请修改 output_dir 或选择断点续训。")
            config["output_dir"] = str(output)
            config["resume"] = bool(info)
            if info:
                for target in ("policy", "reward_model"):
                    if isinstance(config.get(target), dict):
                        config[target].pop("path", None)
            config.pop("checkpoint_path", None)
            run_dir = self.state_dir / identifier
            write_json(run_dir / "config.json", config)
            native = [f"--config_path={info['config_path'] if info else run_dir / 'config.json'}"]
            if info:
                native.append("--resume=true")
            command = [sys.executable, str(Path(__file__).with_name("worker.py")), str(run_dir), *native]
            record.update(
                config=config,
                output_dir=str(output),
                command=command,
                log_path=str(run_dir / "train.log"),
                resume_from=info["path"] if info else None,
                start_step=info["step"] if info else 0,
                status="starting",
                started_at=time.time(),
            )
            record.pop("draft_config", None)
            self._save(record)
            try:
                with (run_dir / "train.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        command,
                        cwd=self.root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "PYTHONPATH": str(self.root / "src")
                            + os.pathsep
                            + os.environ.get("PYTHONPATH", ""),
                        },
                    )
            except OSError as exc:
                record.update(status="failed", error=str(exc), ended_at=time.time())
                self._save(record)
                raise ValueError(f"训练进程未能启动：{exc}") from exc
            self.processes[identifier] = process
            record.update(pid=process.pid, process_identity=process_identity(process.pid), status="running")
            self._save(record)
            return self._status(record)

    def stop(self, identifier: str) -> dict:
        with self.lock:
            result = self.get(identifier)
            if not result.get("managed"):
                raise ValueError("终端启动的训练仅只读监控，请在原终端管理。")
            if result["status"] not in ACTIVE:
                raise ValueError("任务当前没有运行")
            record = self.records[identifier]
            (self.state_dir / identifier / "stop.request").touch()
            record["status"] = "stopping"
            self._save(record)
            return self._status(record)
