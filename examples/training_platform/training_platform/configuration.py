"""Read-only adapters for LeRobot's native configs and local metadata.

Discovery never constructs a policy, reads weights, downloads data, or calls
TrainPipelineConfig.validate (which resolves checkpoints and changes paths).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.metadata
import json
import math
import os
import re
import tomllib
from contextlib import suppress
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f"配置文件过大：{path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 JSON 对象：{path}")
    return value


@lru_cache(maxsize=1)
def _native() -> tuple:
    # Keep the web server importable even when its interpreter lacks training extras.
    import draccus

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.rewards import RewardModelConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    importlib.import_module("lerobot.policies")
    importlib.import_module("lerobot.rewards")
    register_third_party_plugins()
    return draccus, TrainPipelineConfig, {"policy": PreTrainedConfig, "reward_model": RewardModelConfig}


def _fields(cls: type) -> list[dict]:
    return [
        {
            "name": item.name,
            "type": str(item.type),
            "required": item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING,
        }
        for item in dataclasses.fields(cls)
        if item.init
    ]


def _dependency_status(model_id: str) -> tuple[list[str], list[str]]:
    """Distribution metadata only; matching packages does not guarantee binary compatibility."""
    from packaging.requirements import Requirement

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    extra = {"pi0": "pi", "pi05": "pi", "pi0_fast": "pi", "wall_x": "wallx"}.get(model_id, model_id)
    missing, mismatched, seen = [], [], set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for raw in extras.get(name, []):
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate():
                continue
            if req.name == "lerobot":
                for child in req.extras:
                    visit(child)
                continue
            try:
                version = importlib.metadata.version(req.name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(str(req))
                continue
            if req.specifier and version not in req.specifier:
                mismatched.append(f"{req.name}={version}，项目要求 {req.specifier}")

    visit(extra)
    return sorted(set(missing)), sorted(set(mismatched))


@lru_cache(maxsize=1)
def _catalog() -> dict:
    models = []
    try:
        draccus, train_cls, registries = _native()
    except Exception as exc:
        return {"models": [], "errors": [f"LeRobot 配置加载失败：{exc}"], "entrypoint": "lerobot-train"}
    for kind, registry in registries.items():
        for name, cls in sorted(registry.get_known_choices().items()):
            item = {
                "id": name,
                "label": name,
                "kind": kind,
                "fields": _fields(cls),
                "defaults": {},
                "available": True,
                "reason": "",
                "trainable": True,
                "entrypoint": "lerobot-train",
                "warnings": [],
            }
            try:
                # Read declared dataclass defaults without __post_init__: some native
                # configs resolve Hub backbone configs in their constructors.
                config = object.__new__(cls)
                for field in dataclasses.fields(cls):
                    if field.default is not dataclasses.MISSING:
                        value = copy.deepcopy(field.default)
                    elif field.default_factory is not dataclasses.MISSING:
                        value = field.default_factory()
                    else:
                        value = None
                    object.__setattr__(config, field.name, value)
                item["defaults"] = _jsonable(draccus.encode(config, registry))
                item["defaults"]["push_to_hub"] = False
                missing, mismatched = _dependency_status(name)
                item["warnings"] = mismatched
                item["missing_dependencies"] = missing
                if missing:
                    item.update(available=False, reason="缺少可选依赖：" + "; ".join(missing))
            except Exception as exc:
                item.update(available=False, reason=str(exc))
            if name == "gaussian_actor":
                item.update(
                    trainable=False,
                    entrypoint="python -m lerobot.rl.learner",
                    reason="RL actor 需要 LeRobot RL 算法和环境配置，不能用离线 lerobot-train 训练。",
                )
            elif kind == "reward_model" and name == "topreward":
                item.update(
                    trainable=False, entrypoint=None, reason="TOPReward 是零样本奖励模型，没有可训练参数。"
                )
            elif kind == "reward_model" and name == "robometer":
                item.update(
                    trainable=False,
                    entrypoint=None,
                    reason="当前 LeRobot 的 Robometer 实现仅支持推理，未实现训练 forward。",
                )
            item["capabilities"] = {
                "offline_training": item["trainable"],
                "dedicated_rl": name == "gaussian_actor",
            }
            models.append(item)
    return {
        "models": models,
        "errors": [],
        "entrypoint": "lerobot-train",
        "fields": _fields(train_cls),
        "dependency_check": "仅检查包元数据；模型初始化时仍可能报告版本或系统依赖问题。",
    }


def catalog() -> dict:
    return copy.deepcopy(_catalog())


def default_config(model_id: str, kind: str = "policy") -> dict:
    draccus, train_cls, registries = _native()
    if kind not in registries:
        raise ValueError(f"未知模型类别：{kind}")
    item = next(
        (entry for entry in _catalog()["models"] if entry["id"] == model_id and entry["kind"] == kind), None
    )
    if item is None:
        raise ValueError(f"未注册的 LeRobot 模型：{kind}/{model_id}")
    if not item["defaults"]:
        raise ValueError(item["reason"])
    from lerobot.configs.default import DatasetConfig

    config = _jsonable(draccus.encode(train_cls(dataset=DatasetConfig(repo_id=""))))
    config[kind] = copy.deepcopy(item["defaults"])
    config.update(output_dir=None, job_name=None, env_eval_freq=0, resume=False)
    return config


def _check_local_features(decoded: Any, errors: list, warnings: list) -> None:
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features

    source = decoded.dataset.root or decoded.dataset.repo_id
    try:
        info = inspect_dataset(str(source))
    except ValueError:
        warnings.append("本地尚无此数据集元数据；特征兼容性将在训练启动后由 LeRobot 检查。")
        return
    features = dataset_to_policy_features(info["features"])
    features = {decoded.rename_map.get(key, key): value for key, value in features.items()}
    model = decoded.trainable_config
    # Reward models have their own label/output contracts; do not replace them with actions.
    if not decoded.is_reward_model_training:
        for key, expected in {**(model.input_features or {}), **(model.output_features or {})}.items():
            actual = features.get(key)
            if actual is None:
                if ".empty_camera_" not in key:
                    errors.append(f"数据集缺少模型特征 {key}；检查数据或 rename_map。")
            elif expected.type != actual.type:
                errors.append(f"数据集特征 {key} 类型与模型不一致。")
            elif expected.type is not FeatureType.VISUAL and tuple(expected.shape) != tuple(actual.shape):
                errors.append(f"数据集特征 {key} 维度 {actual.shape} 与模型 {expected.shape} 不一致。")
        model.output_features = {
            key: value for key, value in features.items() if value.type is FeatureType.ACTION
        }
        if not model.input_features:
            model.input_features = {
                key: value for key, value in features.items() if key not in model.output_features
            }
        for feature_name, limit_name in (
            ("action", "max_action_dim"),
            ("observation.state", "max_state_dim"),
        ):
            feature, limit = features.get(feature_name), getattr(model, limit_name, None)
            if feature and limit is not None and feature.shape[-1] > limit:
                errors.append(f"数据集 {feature_name} 维度超过模型 {limit_name}={limit}。")
    model.validate_features()


def _check_resume_config(config: dict, info: dict, errors: list, warnings: list) -> None:
    saved = info.get("train_config") or {}
    kind = info["kind"]
    before, after = saved.get(kind) or {}, config.get(kind) or {}
    runtime_options = {
        "device",
        "dtype",
        "use_amp",
        "compile_model",
        "compile_mode",
        "gradient_checkpointing",
        "push_to_hub",
        "repo_id",
        "private",
        "tags",
        "license",
    }
    changed = [
        key
        for key in set(before) | set(after)
        if key not in runtime_options and before.get(key) != after.get(key)
    ]
    if changed:
        errors.append(
            "完整续训不能改变模型结构、动作约定或模型训练参数："
            + ", ".join(sorted(changed))
            + "。需要修改时请作为新任务微调。"
        )
    old_data, new_data = saved.get("dataset") or {}, config.get("dataset") or {}
    changed_data = [
        key
        for key in ("repo_id", "revision", "episodes", "exclude_episodes", "eval_split")
        if old_data.get(key) != new_data.get(key)
    ]
    if changed_data or saved.get("rename_map", {}) != config.get("rename_map", {}):
        errors.append("续训的数据集、集数划分和特征映射必须与 checkpoint 一致；更换数据请新建微调任务。")
    if old_data.get("root") != new_data.get("root"):
        warnings.append("续训的数据根目录发生变化，请确认只是同一数据集移动了存储位置。")
    for key in ("optimizer", "scheduler", "peft", "sample_weighting"):
        if saved.get(key) != config.get(key):
            errors.append(f"完整续训必须保留 checkpoint 的 {key} 配置。")
    if saved.get("batch_size") != config.get("batch_size"):
        warnings.append("续训修改了 batch_size；累计样本数与采样进度将按 LeRobot 的新配置计算。")


def validate_config(config: dict, mode: str = "new", checkpoint: str | None = None) -> dict:
    """Decode using native dataclasses; return errors without I/O side effects or downloads."""
    errors, warnings = [], []
    result = {"valid": False, "errors": errors, "warnings": warnings, "config": copy.deepcopy(config)}
    if not isinstance(config, dict):
        errors.append("训练配置必须是 JSON 对象。")
        return result
    if mode not in ("new", "resume"):
        errors.append(f"未知训练模式：{mode}")
        return result
    try:
        draccus, train_cls, _ = _native()
        payload = copy.deepcopy(config)
        active = [key for key in ("policy", "reward_model") if payload.get(key) is not None]
        if len(active) != 1:
            raise ValueError("必须且只能配置 policy 或 reward_model 中的一个。")
        kind = active[0]
        model = payload[kind]
        if not isinstance(model, dict):
            raise ValueError(f"{kind} 必须是 JSON 对象。")
        item = next(
            (
                entry
                for entry in _catalog()["models"]
                if entry["id"] == model.get("type") and entry["kind"] == kind
            ),
            None,
        )
        if item and not item["trainable"]:
            errors.append(item["reason"])
        if item and not item["available"]:
            errors.append(item["reason"])
        if item:
            warnings.extend(item["warnings"])
        # Preserve the user's device/AMP settings. Validation must not silently rewrite them
        # because GPU availability differs between the web process and the eventual trainer.
        device = model.get("device")
        if device is not None and not re.fullmatch(r"(?:cpu|cuda(?::\d+)?|mps|xpu(?::\d+)?)", str(device)):
            errors.append(f"{kind}.device 无效：{device}")
        model["device"] = "cpu"
        if item and not item["trainable"]:
            return result
        if kind == "policy" and model.get("type") == "eo1" and model.get("vlm_config") is None:
            # Only the validation copy skips backbone resolution; the saved configuration
            # remains unchanged and native training resolves the real config when launched.
            model["vlm_config"] = {}
            warnings.append("EO-1 的基础模型配置将在正式启动时由 LeRobot 解析。")
        decoded = draccus.decode(train_cls, payload)
        if not decoded.dataset.repo_id.strip():
            errors.append("请填写 dataset.repo_id。")
        for name in ("steps", "batch_size", "log_freq"):
            if getattr(decoded, name) <= 0:
                errors.append(f"{name} 必须大于 0。")
        if decoded.num_workers < 0:
            errors.append("num_workers 不能小于 0。")
        if decoded.dataset.root and not Path(decoded.dataset.root).expanduser().is_dir():
            errors.append(f"本地数据目录不存在：{decoded.dataset.root}")
        if decoded.eval_steps > 0 and decoded.dataset.eval_split == 0:
            errors.append("eval_steps > 0 需要 dataset.eval_split > 0。")
        _check_local_features(decoded, errors, warnings)
        if decoded.dataset.eval_split == 0:
            warnings.append("未启用验证集；训练 loss 不能用于判断泛化能力。")
        if not decoded.use_policy_training_preset and decoded.optimizer is None:
            errors.append("关闭 use_policy_training_preset 时必须设置 optimizer。")
        if getattr(decoded.trainable_config, "push_to_hub", False) and not decoded.trainable_config.repo_id:
            errors.append("push_to_hub=true 需要模型 repo_id。")
        if decoded.env is not None and decoded.env_eval_freq > 0:
            warnings.append("此配置包含仿真评估；训练平台不执行真机验证。")
        if decoded.job.is_remote:
            errors.append("当前平台只管理本机训练，请将 job.target 设为空。")
        if mode == "resume":
            if not checkpoint:
                errors.append("断点续训需要选择本地完整 checkpoint。")
            else:
                info = inspect_checkpoint(checkpoint)
                if not info["resumable"]:
                    errors.append("checkpoint 不能完整续训：" + "；".join(info["resume_errors"]))
                _check_resume_config(config, info, errors, warnings)
                if info["kind"] != kind or info["model"] != model.get("type"):
                    errors.append("续训模型类别必须与 checkpoint 一致。")
                if info.get("step") is not None and decoded.steps <= info["step"]:
                    errors.append(f"目标总步数必须大于 checkpoint 步数 {info['step']}。")
        elif decoded.resume:
            errors.append("配置 resume=true，请切换断点续训模式并选择 checkpoint。")
        output = decoded.output_dir
        if mode == "new" and output is not None and Path(output).expanduser().exists():
            errors.append(f"新训练输出目录已存在：{output}；请换一个目录或选择断点续训。")
    except Exception as exc:
        errors.append(str(exc))
    result["valid"] = not errors
    return result


def _dataset_root(path_or_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    path = Path(path_or_id).expanduser()
    if path.is_file() and path.name == "info.json":
        return path.resolve().parent.parent
    if (path / "meta" / "info.json").is_file():
        return path.resolve()
    cached = HF_LEROBOT_HOME / path_or_id
    if (cached / "meta" / "info.json").is_file():
        return cached.resolve()
    raise ValueError(f"未找到本地数据集元数据：{path_or_id}。可填写数据集绝对路径；不会自动下载。")


def inspect_dataset(path_or_id: str) -> dict:
    path = _dataset_root(path_or_id)
    info = _read_json(path / "meta" / "info.json")
    features = info.get("features", {})
    names = features.get("action", {}).get("names") or []
    if isinstance(names, dict):
        names = list(names.values())
    names = [str(name).lower() for name in names]
    space = "unknown"
    if names and any("joint" in name for name in names):
        space = "joint"
    elif names and any("eef" in name or "rot6d" in name for name in names):
        space = "eef"
    left, right = any("left" in name for name in names), any("right" in name for name in names)
    arms = "dual" if left and right else "single" if left or right else "unknown"
    if arms == "unknown" and info.get("robot_type") in {"nero_follower", "piper_follower"}:
        arms = "single"
    return {
        "path": str(path),
        "repo_id": info.get("repo_id") or "/".join(path.parts[-2:]),
        "frames": info.get("total_frames"),
        "episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_episodes": info.get("total_episodes"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "features": features,
        "cameras": [name for name, feature in features.items() if feature.get("dtype") in ("video", "image")],
        "space": space,
        "arms": arms,
        "codebase_version": info.get("codebase_version"),
    }


def discover_datasets() -> list:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    result, seen = [], set()
    for root in (HF_LEROBOT_HOME, ROOT / "datasets", ROOT / "data", ROOT / "outputs"):
        if not root.is_dir():
            continue
        for directory, dirs, _files in os.walk(root, followlinks=False):
            path = Path(directory)
            dirs[:] = [
                name
                for name in dirs
                if name not in {"videos", "images", ".git", "checkpoints", "blobs"}
                and not name.startswith(".")
            ]
            if len(path.relative_to(root).parts) > 8:
                dirs[:] = []
                continue
            if (path / "meta" / "info.json").is_file():
                dirs[:] = []
                if path.resolve() not in seen:
                    seen.add(path.resolve())
                    with suppress(ValueError, OSError):
                        result.append(inspect_dataset(str(path)))
    return sorted(result, key=lambda item: item["path"])


def inspect_checkpoint(path: str) -> dict:
    model = Path(path).expanduser().resolve()
    if model.is_file():
        model = model.parent
    if (model / "pretrained_model").is_dir():
        model /= "pretrained_model"
    if not (model / "config.json").is_file():
        raise ValueError(f"目录缺少 config.json：{model}")
    config = _read_json(model / "config.json")
    train_path = model / "train_config.json"
    train = _read_json(train_path) if train_path.is_file() else None
    state = model.parent / "training_state"
    errors = []
    weight_files = ("model.safetensors", "adapter_model.safetensors")
    weights = any((model / name).is_file() for name in weight_files)
    shards = (model / "pytorch_model_fsdp_0").is_dir()
    if not weights and not shards:
        errors.append("缺少模型权重")
    if train is None:
        errors.append("缺少 train_config.json")
    for name in ("training_step.json", "rng_state.safetensors"):
        if not (state / name).is_file():
            errors.append(f"缺少 training_state/{name}")
    if not (state / "optimizer_0").is_dir():
        for name in ("optimizer_state.safetensors", "optimizer_param_groups.json"):
            if not (state / name).is_file():
                errors.append(f"缺少 training_state/{name}")
    if train and train.get("scheduler") and not (state / "scheduler_state.json").is_file():
        errors.append("缺少 training_state/scheduler_state.json")
    step = None
    if (state / "training_step.json").is_file():
        step = _read_json(state / "training_step.json").get("step")
    kind = "reward_model" if train and train.get("reward_model") else "policy"
    return {
        "path": str(model),
        "checkpoint_path": str(model.parent),
        "model": config.get("type"),
        "kind": kind,
        "step": step,
        "config": config,
        "train_config": train,
        "train_config_path": str(train_path) if train else None,
        "resumable": not errors,
        "resume_errors": errors,
        "weights_available": weights or shards,
        "modified_at": model.stat().st_mtime,
    }


def discover_checkpoints(root: str | None = None) -> list:
    scan_root = Path(root).expanduser().resolve() if root else ROOT / "outputs"
    if not scan_root.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("自动扫描仅限 LeRobot 项目的 outputs 文件夹；其他路径请手动选择。")
    result, seen = [], set()
    for directory, dirs, files in os.walk(scan_root, followlinks=False):
        path = Path(directory)
        dirs[:] = [
            name
            for name in dirs
            if name not in {"training_state", "tokenizer", "videos", "data", "wandb"}
            and not name.startswith(".")
            and not (path / name).is_symlink()
        ]
        if "config.json" in files and (path.name == "pretrained_model" or "train_config.json" in files):
            dirs[:] = []
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            with suppress(ValueError, OSError):
                result.append(inspect_checkpoint(str(path)))
    return sorted(result, key=lambda item: item["modified_at"], reverse=True)
