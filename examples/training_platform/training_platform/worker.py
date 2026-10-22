"""Detached native trainer with a cooperative checkpoint-boundary stop hook.

No trainer loop is copied. The stop hook runs after native optimizer, RNG, processor,
training metadata and EMA serialization, then exits before another training update.
Only single-process local runs are accepted by Jobs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any


class SavedStopError(Exception):
    """A requested stop reached a complete native checkpoint."""


def install_stop_hooks(trainer: Any, request: Path, accumulation_steps: int = 1) -> None:
    original_should_save = trainer.should_save_checkpoint
    original_update_last = trainer.update_last_checkpoint
    stopping = False

    def should_save(step: int, save_freq: int, total_steps: int) -> bool:
        nonlocal stopping
        # Never stop between accumulation micro-batches: in-flight gradients are not serialized.
        stopping = request.exists() and step % accumulation_steps == 0
        return stopping or original_should_save(step, save_freq, total_steps)

    def update_last(checkpoint_dir: Path) -> None:
        original_update_last(checkpoint_dir)
        if stopping:
            logging.info("TRAINING_PLATFORM saved checkpoint and stopped: %s", checkpoint_dir)
            raise SavedStopError(str(checkpoint_dir))

    trainer.should_save_checkpoint = should_save
    trainer.update_last_checkpoint = update_last


def parse_resume_config(config_path: Path, native_args: list[str]) -> Any:
    """Parse complete edited config natively, while resume keeps the original state location.

    `parser.wrap` accepts a typed config directly. The trainer can therefore use the
    complete snapshot (including arbitrary nested map fields) without flattening it
    into lossy CLI overrides. Its validate() still sees the original checkpoint argv.
    """
    from lerobot.configs import parser  # noqa: PLC0415
    from lerobot.configs.train import TrainPipelineConfig  # noqa: PLC0415

    def capture(cfg: Any) -> Any:
        return cfg

    capture.__annotations__["cfg"] = TrainPipelineConfig
    try:
        sys.argv = ["lerobot-train", f"--config_path={config_path}"]
        return parser.wrap()(capture)()
    finally:
        sys.argv = ["lerobot-train", *native_args]


def main() -> int:
    run_dir = Path(sys.argv[1])
    native_args = sys.argv[2:]
    config = json.loads((run_dir / "config.json").read_text())
    result: dict[str, Any] = {}
    try:
        # Heavy imports belong to this child, never the web server.
        from lerobot.scripts import lerobot_train  # noqa: PLC0415

        accumulation = int(
            ((config.get("accelerator") or {}).get("gradient_accumulation") or {}).get("steps", 1)
        )
        install_stop_hooks(lerobot_train, run_dir / "stop.request", accumulation)
        sys.argv = ["lerobot-train", *native_args]
        if config.get("resume"):
            lerobot_train.register_third_party_plugins()
            parsed = parse_resume_config(run_dir / "config.json", native_args)
            lerobot_train.train(parsed)
        else:
            lerobot_train.main()
        result["status"] = "completed"
        return 0
    except SavedStopError as exc:
        result.update(status="stopped", checkpoint=str(exc))
        return 0
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        result["ended_at"] = time.time()
        temporary = run_dir / "result.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False))
        temporary.replace(run_dir / "result.json")


if __name__ == "__main__":
    raise SystemExit(main())
