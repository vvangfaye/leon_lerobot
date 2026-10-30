"""python -m examples.nero_rl.run --help"""

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from contextlib import AbstractContextManager
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import torch

from .config import TaskConfig
from .environment import MockHardware, NeroEnv
from .hardware import NeroHardware
from .learning import Trainer, load_episodes, save_episode
from .master_input import MasterInput, WorkbenchGuard
from .web_control import WebControls

HELP = """
n: begin next episode after restoring arm/object/packaging; q: quit
i: human control; a: autonomous policy (train/eval only); p: pause
w/s: +/- Y; d/a is NOT used for jogging (a selects policy); d/v: +/- X
r/f: +/- Z; j/k: +/- yaw; o/c: open/close gripper
1: confirmed success; 0: confirmed failure; x or Ctrl+C: controller stop & exit
Between episodes: the same jog keys reposition the arm (not recorded).
Keys apply one small increment per keypress/repeat. Hands off keys means hold.
"""
MASTER_HELP = """
Right master: relative XYZ (default 0.25x), base yaw and gripper increments.
n: begin at the validated follower start; i: engage/rebase master; p: disengage
a: policy (train/eval); 1/0: success/failure; q: save & quit; x/Ctrl+C: stop
After p, reposition the master freely, then press i: no absolute alignment.
Move slowly: excess per-step displacement is discarded, never queued.
Between episodes press i to reposition the follower, p to hold, n to begin.
"""
JOG = {
    "d": (0, 1),
    "v": (0, -1),
    "w": (1, 1),
    "s": (1, -1),
    "r": (2, 1),
    "f": (2, -1),
    "j": (3, 1),
    "k": (3, -1),
    "o": (4, 1),
    "c": (4, -1),
}


class Keyboard(AbstractContextManager):
    """TTY reader also exposes an interrupt event during bounded hardware calls."""

    def __init__(self):
        self.events: Queue[str] = Queue()
        self.abort = threading.Event()
        self.closed = threading.Event()

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Live control requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self) -> None:
        while not self.closed.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                data = os.read(self.fd, 64)
                if not data:
                    self.abort.set()
                for key in data.decode("utf-8", errors="ignore"):
                    if key in ("x", "\x03"):
                        self.abort.set()
                    self.events.put(key)

    def poll(self) -> list[str]:
        if self.abort.is_set():
            raise KeyboardInterrupt("Operator stop")
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except Empty:
                return result

    def __exit__(self, *args):
        self.closed.set()
        self.thread.join(timeout=0.2)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class MockTeacher:
    """Scripted fixture to exercise the pipeline, not an actual trained policy."""

    def __init__(self):
        self.phase = 0

    def action(self, backend: MockHardware) -> np.ndarray:
        above = np.array([0, 0, 0.025])
        targets = [
            backend.object + above,
            backend.object.copy(),
            None,
            backend.object + above,
            backend.goal + above,
            backend.goal,
            None,
        ]
        # The lifted object's position follows the gripper, so lift to a fixed height.
        targets[3] = np.array([backend.q[0], backend.q[1], 0.025])
        result = np.zeros(5)
        if self.phase in (2, 6):
            result[4] = -1 if self.phase == 2 else 1
            if (self.phase == 2 and backend.grasped) or (self.phase == 6 and backend.success()):
                self.phase += 1
        elif self.phase < 7:
            delta = targets[self.phase] - backend.q[:3]
            result[:3] = np.clip(delta / backend.cfg.translation_step_m, -1, 1)
            if np.linalg.norm(delta) < 0.001:
                self.phase += 1
        return result


def transition(
    obs: dict,
    after: dict,
    info: dict,
    reward: float,
    done: bool,
    truncated: bool,
    intervention: bool,
    source: str | None = None,
) -> dict:
    return {
        "state": obs,
        "next_state": after,
        "action": info["executed_action"],
        "requested_action": info["requested_action"],
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "intervention": intervention,
        "elapsed_s": info["elapsed_s"],
        "safety_rejection": info["safety_rejection"] is not None,
        "action_source": source or ("keyboard" if intervention else "policy"),
    }


def finish_episode(args: argparse.Namespace, cfg: TaskConfig, rows: list[dict]) -> Path:
    path = save_episode(args.output / "episodes", rows, cfg, mock=args.mock, kind=args.mode)
    metrics = {
        "episode": path.name,
        "mode": args.mode,
        "mock": args.mock,
        "success": rows[-1]["reward"] == 1,
        "steps": len(rows),
        "human_steps": sum(row["intervention"] for row in rows),
        "safety_rejections": sum(row["safety_rejection"] for row in rows),
        "action_time_s": sum(row["elapsed_s"] for row in rows),
        "truncated": bool(rows[-1]["truncated"]),
        "action_sources": sorted({row["action_source"] for row in rows}),
    }
    with (args.output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics) + "\n")
    return path


def preview(obs: dict, enabled: bool) -> None:
    if not enabled:
        return
    for name, value in obs.items():
        if name.startswith("observation.images."):
            cv2.imshow(name, cv2.cvtColor(value.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def mock_config() -> TaskConfig:
    return TaskConfig(
        workspace_min_m=[-0.06, -0.06, -0.002],
        workspace_max_m=[0.06, 0.06, 0.08],
        cameras={"head": "synthetic"},
        translation_step_m=0.004,
        max_joint_step_rad=0.02,
        hz=10,
        learning_starts=16,
        batch_size=8,
        replay_capacity=512,
        updates_per_episode=3,
    )


def smoke(cfg: TaskConfig, output: Path) -> dict:
    backend = MockHardware(cfg)
    env = NeroEnv(cfg, backend, mock=True)
    trainer = Trainer(cfg, mock=True)
    teacher = MockTeacher()
    obs, _ = env.reset()
    rows = []
    for _ in range(500):
        action = teacher.action(backend)
        outcome = "success" if backend.success() else None
        after, reward, done, truncated, info = env.step(action, outcome=outcome)
        row = transition(obs, after, info, reward, done, truncated, True)
        rows.append(row)
        obs = after
        if done or truncated:
            break
    if rows[-1]["reward"] != 1:
        raise RuntimeError("Synthetic pick/place fixture did not succeed")
    path = save_episode(output / "episodes", rows, cfg, mock=True, kind="demo")
    n = load_episodes(output / "episodes", trainer.replay, cfg, mock=True)
    initial = next(trainer.policy.actor.parameters()).detach().clone()
    losses = trainer.update(cfg.updates_per_episode)
    if not losses or torch.equal(initial, next(trainer.policy.actor.parameters()).detach()):
        raise RuntimeError("SAC did not update actor parameters")
    checkpoint = output / "checkpoint.pt"
    trainer.save(checkpoint)
    expected = trainer.act(obs, deterministic=True)
    restored = Trainer(cfg, mock=True)
    restored.load(checkpoint)
    if not np.allclose(expected, restored.act(obs, deterministic=True), atol=1e-6):
        raise RuntimeError("Checkpoint did not restore identical inference")
    result = {
        "mock_only": True,
        "episode": str(path),
        "transitions": n,
        "updates": trainer.updates,
        "checkpoint_restored": True,
        "losses": losses,
        "note": "Synthetic fixture only; no real robot movement or task-performance claim.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2))
    env.close()
    return result


def run_interactive(args: argparse.Namespace, cfg: TaskConfig) -> None:
    if cfg.action_mode == "joint_target":
        from .native_run import run_native

        run_native(args, cfg, keyboard_factory=Keyboard, episode_writer=finish_episode)
        return
    cfg.validate(live=not args.mock)
    if cfg.action_mode == "joint_delta" and args.human_input != "master":
        raise ValueError("Joint-delta live control requires --human-input master")
    master = (
        MasterInput(cfg, translation_scale=args.master_translation_scale)
        if args.human_input == "master"
        else None
    )
    guard = WorkbenchGuard(args.workbench_url) if master else None
    last_guard = -float("inf")

    def check_controller() -> None:
        nonlocal last_guard
        if guard and time.monotonic() - last_guard >= 0.5:
            guard.check()
            last_guard = time.monotonic()

    backend = MockHardware(cfg) if args.mock else NeroHardware(cfg)
    trainer = Trainer(cfg, mock=args.mock) if args.mode != "record" else None
    if args.mode == "eval" and not args.resume:
        raise ValueError("Evaluation requires --resume checkpoint.pt")
    if trainer:
        for root in sorted({root.resolve() for root in [args.output / "episodes", *args.data]}):
            if root.exists():
                print(
                    f"Loaded {load_episodes(root, trainer.replay, cfg, mock=args.mock)} transitions from {root}"
                )
        if args.resume:
            trainer.load(args.resume)
        elif (args.output / "checkpoint.pt").exists():
            raise ValueError("Checkpoint already exists: specify --resume or a new output directory")
        if args.mode == "train" and not trainer.replay.demos:
            print("No demos loaded: collect human episodes before switching to autonomous actions.")
    env, rows = None, []
    try:
        check_controller()
        backend.open()
        if master:
            master.open()
        env = NeroEnv(cfg, backend, mock=args.mock)
        print(
            "Joint-delta master control: 7 joints + gripper; n/i/p/a/1/0/q/x."
            if cfg.action_mode == "joint_delta"
            else MASTER_HELP
            if master
            else HELP
        )
        controls = (
            WebControls(args.web_host, args.web_port, args.output / "control_url.txt", args.web_public_host)
            if getattr(args, "web", False)
            else Keyboard()
        )
        with controls as keys:
            cycle_ends: deque[float] = deque(maxlen=200)
            diagnostics: dict = {}

            def report_cycle(info: dict) -> None:
                cycle_ends.append(time.monotonic())
                diagnostics["tracking_error_deg"] = round(info.get("tracking_error_deg", 0.0), 3)
                diagnostics["tracking_limited"] = info.get("tracking_limited", False)
                diagnostics["workspace_limited"] = info.get("workspace_limited", False)
                if len(cycle_ends) > 1:
                    diagnostics["control_hz_measured"] = round(
                        (len(cycle_ends) - 1) / (cycle_ends[-1] - cycle_ends[0]), 1
                    )

            def hold() -> dict:
                cycle_ends.clear()
                diagnostics["last_control_hz_measured"] = diagnostics.get("control_hz_measured", 0.0)
                diagnostics["control_hz_measured"] = 0.0
                return env.hold()

            def rejection_message(info: dict) -> str:
                reason = info["safety_rejection"]
                details = info.get("tracking_diagnostics", {})
                event = {"time": time.time(), "reason": reason, **details}
                with (args.output / "control_events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                descriptions = {
                    "tracking_error": "指令与实测关节差超过 3°，已保持当前位置",
                    "tracking_stall": "从臂持续 1 秒没有明显跟进，已保持当前位置",
                    "workspace": "当前关节路径无法保持在工作范围内，已保持当前位置",
                }
                message = descriptions.get(reason, f"动作被约束拒绝：{reason}")
                if details.get("stalled_joints"):
                    message += "（" + "/".join(f"J{j}" for j in details["stalled_joints"]) + "）"
                return message + "。调整主臂后按主臂跟随可恢复。"

            def publish(phase: str, message: str = "") -> None:
                if hasattr(keys, "publish"):
                    keys.publish(
                        phase=phase,
                        message=message,
                        task=cfg.task,
                        steps=env.steps,
                        episode=episode + 1,
                        training_updates=trainer.updates if trainer else 0,
                        mode=args.mode,
                        control_hz_target=cfg.hz,
                        action_mode=cfg.action_mode,
                        **diagnostics,
                        frames=getattr(backend, "last_frames", {}),
                    )

            def master_action() -> np.ndarray:
                if cfg.action_mode == "joint_delta":
                    return master.joint_action(*env.joint_reference())
                return master.action()

            backend.abort_requested = keys.abort.is_set
            if master:
                master.abort_requested = keys.abort.is_set
            for episode in range(args.episodes):
                print(
                    f"Episode {episode + 1}: restore scene/start pose; press n. Between episodes jog keys are unrecorded."
                )
                repositioning = False
                last_preview = 0.0
                wait_message = "请恢复物体；本轮从空夹爪放置位开始。按开始本集后用主臂示范。"
                while True:
                    check_controller()
                    if time.monotonic() - last_preview >= 0.2:
                        backend.read()
                        publish("waiting", wait_message)
                        last_preview = time.monotonic()
                    events = keys.poll()
                    if "q" in events:
                        return
                    if "n" in events:
                        if repositioning:
                            hold()
                            repositioning = False
                        try:
                            obs, _ = env.reset()
                        except ValueError as exc:
                            wait_message = str(exc)
                            print(wait_message)
                            publish("waiting", wait_message)
                            continue
                        break
                    if master:
                        if "p" in events:
                            repositioning = False
                            if backend.prepared:
                                hold()
                            wait_message = "跟随已暂停；恢复物体和起点后可开始本集。"
                        elif "i" in events:
                            env.prepare_repositioning()
                            master.prepare()
                            master.rebase()
                            repositioning = True
                            cycle_ends.clear()
                            wait_message = "仅跟随调试／返回起点中，不录制数据；按暂停可停下。"
                    for key in events:
                        if key in JOG and backend.prepared and cfg.action_mode == "cartesian":
                            action = np.zeros(5)
                            axis, sign = JOG[key]
                            action[axis] = sign
                            env.jog(action)
                            if master and master.prepared:
                                master.rebase()
                    if repositioning:
                        info = env.jog(master_action())
                        report_cycle(info)
                        if info["safety_rejection"]:
                            hold()
                            repositioning = False
                            wait_message = rejection_message(info)
                            publish("waiting", wait_message)
                    else:
                        time.sleep(0.01)
                if master:
                    master.prepare()
                    master.rebase()
                rows, human, paused = [], True, False
                last_preview = 0.0
                pause_message = "跟随已暂停。"
                cycle_ends.clear()
                while True:
                    cycle_started = time.monotonic()
                    check_controller()
                    action, outcome = np.zeros(len(cfg.scales)), None
                    keyboard_action = False
                    rebase = False
                    events = keys.poll()
                    if "q" in events:
                        if rows:
                            rows[-1]["truncated"] = True
                            finish_episode(args, cfg, rows)
                            rows = []
                        return
                    for key in events:
                        if key == "i":
                            human, paused = True, False
                            rebase = True
                        elif key == "a" and trainer is not None:
                            if args.mode == "train" and not any(
                                r["reward"] > 0 for r in trainer.replay.demos
                            ):
                                print("Collect/load a successful human demonstration first.")
                            else:
                                human, paused = False, False
                        elif key == "p":
                            paused = True
                            pause_message = "跟随已暂停。"
                            obs = hold()
                        elif key in ("1", "0"):
                            outcome = "success" if key == "1" else "failure"
                        elif key in JOG and cfg.action_mode == "cartesian":
                            # Any jog key immediately takes control from the policy.
                            human, paused = True, False
                            keyboard_action = True
                            rebase = True
                            axis, sign = JOG[key]
                            action[axis] = sign
                    preview(obs, args.preview)
                    if time.monotonic() - last_preview >= 0.2:
                        publish(
                            "paused" if paused else "human" if human else "policy",
                            pause_message if paused else "主臂示范中" if human else "策略运行中",
                        )
                        last_preview = time.monotonic()
                    if paused and outcome is None:
                        # Refresh after an operator pause; do not infer from an old camera image.
                        obs = env.observation(backend.read())
                        time.sleep(0.02)
                        continue
                    if master and rebase:
                        master.rebase()
                    if not human and outcome is None:
                        action = trainer.act(obs, deterministic=args.mode == "eval")
                    elif master and human and not keyboard_action and outcome is None:
                        action = master_action()
                    if args.mock:
                        time.sleep(1 / cfg.hz)
                    after, reward, done, truncated, info = env.step(
                        action, outcome=outcome, autonomous=not human, cycle_started=cycle_started
                    )
                    report_cycle(info)
                    source = "right_master" if master and human and not keyboard_action else None
                    if outcome is not None:
                        source = "outcome"
                    rows.append(transition(obs, after, info, reward, done, truncated, human, source))
                    obs = after
                    if info["safety_rejection"]:
                        paused = True
                        obs = hold()
                        pause_message = rejection_message(info)
                        publish("paused", pause_message)
                        print(f"Action rejected ({info['safety_rejection']}); paused. Press i to intervene.")
                    if done or truncated:
                        hold()
                        break
                path = finish_episode(args, cfg, rows)
                completed_rows, rows = rows, []
                if trainer and args.mode == "train":
                    for row in completed_rows:
                        trainer.replay.add(row)
                    print("Updating SAC while the arm holds its settled position...")
                    publish("updating", "本集已保存，正在更新 SAC；从臂保持位置。")
                    losses = trainer.update(cfg.updates_per_episode)
                    trainer.save(args.output / "checkpoint.pt")
                    print(json.dumps({"updates": trainer.updates, "losses": losses}))
                print(
                    f"Saved {path}; reward={completed_rows[-1]['reward']}, human_steps={sum(r['intervention'] for r in completed_rows)}"
                )
    except BaseException:
        if not args.mock and backend.prepared:
            print("Stopping controller. This is a damped stop; the arm may descend.", file=sys.stderr)
            backend.stop()
        if rows:
            rows[-1]["truncated"] = True
            finish_episode(args, cfg, rows)
        raise
    finally:
        try:
            if env is not None and backend.prepared and cfg.action_mode == "joint_delta":
                env.hold()
        finally:
            if master:
                try:
                    master.close()
                finally:
                    backend.close()
            else:
                backend.close()
        if args.preview:
            cv2.destroyAllWindows()
        print("Connections closed. No automatic home or torque-disable was issued.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NERO single-arm SAC: demonstrations + human corrections + RL"
    )
    parser.add_argument("mode", choices=["smoke", "inspect", "record", "train", "eval"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mock", action="store_true", help="Synthetic fixture; never connect hardware")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow right-follower control and optional active right-master synchronization",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/nero_rl"))
    parser.add_argument(
        "--data", type=Path, action="append", default=[], help="Directory of demonstration NPZ episodes"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--human-input", choices=["keyboard", "master"], default="keyboard")
    parser.add_argument("--master-translation-scale", type=float, default=0.25)
    parser.add_argument(
        "--active-master-sync",
        action="store_true",
        help="Mirror measured follower motion on the right master",
    )
    parser.add_argument("--workbench-url", default="http://127.0.0.1:5000")
    parser.add_argument("--web", action="store_true", help="Use local browser controls instead of a terminal")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-public-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=5001)
    parser.add_argument(
        "--policy-burst-seconds",
        type=float,
        help="Pause autonomous motion after this duration (native runner)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Open camera windows if a desktop display is available"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mock and args.execute:
        parser.error("Choose --mock OR --execute")
    if args.human_input == "master" and (args.mock or args.mode in ("smoke", "inspect")):
        parser.error("Master input is for live record/train/eval with --execute")
    if args.mode == "smoke":
        if args.execute or args.config:
            parser.error("smoke uses its own synthetic fixture; omit --execute and --config")
        print(json.dumps(smoke(mock_config(), args.output), indent=2))
        return
    if args.config is None and not args.mock:
        parser.error("Live modes require --config")
    cfg = TaskConfig.load(args.config) if args.config else mock_config()
    if args.active_master_sync and (
        args.human_input != "master"
        or cfg.action_mode != "joint_target"
        or args.mode not in ("train", "eval")
    ):
        parser.error("Active master sync requires native joint_target train/eval with master input")
    if args.mode == "inspect":
        cfg.validate()
        if cfg.channel != "can1":
            parser.error("Inspect is restricted to right follower can1")
        backend = MockHardware(cfg) if args.mock else NeroHardware(cfg)
        try:
            backend.open()
            snap = backend.read()
            args.output.mkdir(parents=True, exist_ok=True)
            report = {
                "read_only": True,
                "mock": args.mock,
                "joints_rad": snap.joints.tolist(),
                "gripper_m": snap.width,
                "pose": snap.pose.tolist(),
                "pose_frame": "TCP" if cfg.tcp_offset_m is not None else "SDK link7 (TCP not calibrated)",
            }
            (args.output / "inspection.json").write_text(json.dumps(report, indent=2))
            for name, frame in snap.images.items():
                cv2.imwrite(str(args.output / f"{name}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(json.dumps(report, indent=2))
        finally:
            backend.close()
        return
    if not args.mock and not args.execute:
        parser.error("Use --mock for offline tests; live motion requires --execute")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.policy_burst_seconds is not None and (
        not np.isfinite(args.policy_burst_seconds) or args.policy_burst_seconds <= 0
    ):
        parser.error("--policy-burst-seconds must be finite and positive")
    run_interactive(args, cfg)


if __name__ == "__main__":
    main()
