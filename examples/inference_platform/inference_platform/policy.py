"""An isolated π0.5 process. Loading/predicting never opens hardware."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from typing import Any


def policy_worker(pipe: Any, path: str) -> None:
    # Everything must be local: the selected checkpoint includes its processors
    # and tokenizer. An incomplete checkpoint fails instead of fetching a model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import numpy as np
        import torch

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pretrained import PreTrainedPolicy

        torch.set_num_threads(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = PI05Config.from_pretrained(path, local_files_only=True)
        cfg.device, cfg.compile_model, cfg.gradient_checkpointing = device, False, False
        cfg.pretrained_path = path
        # PI05's conversion loader catches weight errors and can return random
        # weights. Native saved checkpoints must use the strict base loader.
        policy = PreTrainedPolicy.from_pretrained.__func__(
            PI05Policy,
            path,
            config=cfg,
            local_files_only=True,
            strict=True,
        ).eval()
        pre, post = make_pre_post_processors(
            cfg,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        pipe.send({"ok": True, "device": device})
        while True:
            request = pipe.recv()
            if request["op"] == "close":
                break
            if request["op"] == "reset":
                policy.reset()
                pre.reset()
                post.reset()
                pipe.send({"ok": True})
                continue
            obs = {"observation.state": torch.from_numpy(request["state"]), "task": request["task"]}
            for key, rgb in request["images"].items():
                obs[key] = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
            started = time.monotonic()
            with torch.inference_mode():
                batch = pre(obs)
                actions = policy.predict_action_chunk(batch)[:, : cfg.n_action_steps]
                # Run the saved postprocessor over the full chunk, including its
                # shared relative-action anchor, before returning physical units.
                actions = post(actions).squeeze(0).float().cpu().numpy()
            pipe.send(
                {"ok": True, "actions": actions, "inference_ms": round((time.monotonic() - started) * 1000)}
            )
    except EOFError:
        pass
    except BaseException as exc:
        with contextlib.suppress(BrokenPipeError, OSError):
            pipe.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        pipe.close()


class PolicyProcess:
    def __init__(self, path: str) -> None:
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(
            target=policy_worker, args=(child, path), daemon=True, name="pi05-policy"
        )
        self.process.start()
        child.close()

    def receive(self, check: Callable, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            check()
            if self.pipe.poll(0.05):
                try:
                    result = self.pipe.recv()
                except EOFError as exc:
                    raise ValueError("模型进程退出，请查看启动终端并重新加载") from exc
                if not result.get("ok"):
                    raise ValueError(result.get("error", "模型进程异常"))
                return result
            if not self.process.is_alive():
                raise ValueError(f"模型进程退出（{self.process.exitcode}），请重新加载")
            if time.monotonic() >= deadline:
                raise ValueError("模型操作超时，请重新加载")

    def load(self, check: Callable) -> dict:
        return self.receive(check, 300)

    def call(self, request: dict, check: Callable) -> dict:
        check()
        self.pipe.send(request)
        return self.receive(check, 60)

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1)
        self.pipe.close()
