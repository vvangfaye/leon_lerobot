"""An isolated policy process. Loading/predicting never opens hardware."""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
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

        from lerobot.inference.contracts import ExecutionMode
        from lerobot.inference.prediction import chunk_inference_context
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
        from lerobot.processor import (
            NormalizerProcessorStep,
            RelativeActionsProcessorStep,
            TransitionKey,
            create_transition,
        )

        torch.set_num_threads(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = json.loads((Path(path) / "config.json").read_text())["type"]
        if model_type == "pi05":
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            policy_class = PI05Policy
            cfg = PI05Config.from_pretrained(path, local_files_only=True)
            cfg.compile_model, cfg.gradient_checkpointing = False, False
        elif model_type == "fastwam":
            from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
            from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

            policy_class = FastWAMPolicy
            cfg = FastWAMConfig.from_pretrained(path, local_files_only=True)
            cfg.compile_action_infer, cfg.use_gradient_checkpointing = False, False
        else:
            raise ValueError(f"Unsupported policy: {model_type}")
        cfg.device = device
        cfg.pretrained_path = path
        # PI05's conversion loader catches weight errors and can return random
        # weights. Native saved checkpoints must use the strict base loader.
        policy = PreTrainedPolicy.from_pretrained.__func__(
            policy_class,
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
        normalizer = next((s for s in pre.steps if isinstance(s, NormalizerProcessorStep)), None)
        relative = next(
            (s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled), None
        )
        active_rtc = None
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
            options = request.get("rtc")
            if options and model_type != "pi05":
                raise ValueError("FastWAM 尚未接入 RTC，请关闭 RTC 后运行")
            if options != active_rtc:
                cfg.rtc_config = RTCConfig(**options) if options else None
                policy.init_rtc_processor()
                active_rtc = options
            mode = (
                ExecutionMode.RTC_TRAINED if options["mode"] == "trained" else ExecutionMode.RTC_GUIDED
            ) if options else ExecutionMode.CHUNK
            started = time.monotonic()
            # Guided RTC needs ordinary tensors: inference_mode would prevent
            # the denoising correction from temporarily enabling autograd.
            with chunk_inference_context(mode):
                batch = pre(obs)
                kwargs = {}
                if options:
                    prefix = request.get("prefix")
                    if prefix is not None:
                        prefix = torch.as_tensor(prefix, dtype=torch.float32)
                        if prefix.ndim != 2 or prefix.shape[1] != cfg.output_features["action"].shape[0] \
                                or not torch.isfinite(prefix).all():
                            raise ValueError("RTC prefix must contain finite canonical actions")
                        # Prefixes are the actual planned (including clipped)
                        # targets, not unexecuted raw model proposals.
                        if relative is not None:
                            prefix = reanchor_relative_rtc_prefix(
                                prefix, relative.get_cached_state(), relative, normalizer, device
                            )
                        elif normalizer is not None:
                            prefix = normalizer(create_transition(action=prefix))[TransitionKey.ACTION]
                        prefix = prefix.to(device)
                    kwargs = {"prev_chunk_left_over": prefix,
                              "inference_delay": int(request.get("inference_delay", 0)),
                              "execution_horizon": options["execution_horizon"]}
                actions = policy.predict_action_chunk(batch, **kwargs)[:, : cfg.n_action_steps]
                # Run the saved postprocessor over the full chunk, including its
                # shared relative-action anchor, before returning physical units.
                actions = post(actions).squeeze(0).float().cpu().numpy()
                if not np.isfinite(actions).all():
                    raise ValueError("Policy returned non-finite actions")
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
            target=policy_worker, args=(child, path), daemon=True, name="inference-policy"
        )
        self.process.start()
        child.close()
        self.call_lock = threading.Lock()

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
        with self.call_lock:
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
