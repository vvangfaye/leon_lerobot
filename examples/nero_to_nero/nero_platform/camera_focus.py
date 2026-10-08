"""V4L2 对焦设置与首尾审计。只在启动/集边界调用，不进入逐帧采样。"""

import errno
import os
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from fcntl import ioctl
from pathlib import Path

DEFAULT_FOCUS_VALUE = 128
_QUERY_CTRL = 0xC0445624
_GET_CTRL = 0xC008561B
_SET_CTRL = 0xC008561C
_QUERY_FORMAT = "II32siiiiI2I"
_VALUE_FORMAT = "Ii"
_CONTROLS = {
    "focus_auto": 0x009A090C,
    "focus_absolute": 0x009A090A,
    "zoom_absolute": 0x009A090D,
    "exposure_auto": 0x009A0901,
    "exposure_absolute": 0x009A0902,
}
_FOCUS_KEYS = ("focus_auto", "focus_absolute")


@contextmanager
def _device(path: str | Path, *, writable: bool = False) -> Iterator[int]:
    fd = os.open(str(path), (os.O_RDWR if writable else os.O_RDONLY) | os.O_NONBLOCK)
    try:
        yield fd
    finally:
        os.close(fd)


def _query(fd: int, name: str) -> dict | None:
    try:
        raw = ioctl(fd, _QUERY_CTRL, struct.pack(_QUERY_FORMAT, _CONTROLS[name], 0, b"", 0, 0, 0, 0, 0, 0, 0))
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ENOTTY):
            return None  # 查询明确不提供此控制；不是把任意读取错误当成定焦镜头。
        raise
    fields = struct.unpack(_QUERY_FORMAT, raw)
    if fields[7] & 0x0001:  # V4L2_CTRL_FLAG_DISABLED
        return None
    return dict(zip(("min", "max", "step", "default", "flags"), fields[3:8], strict=True))


def _get(fd: int, name: str) -> int:
    return struct.unpack(_VALUE_FORMAT, ioctl(fd, _GET_CTRL, struct.pack(_VALUE_FORMAT, _CONTROLS[name], 0)))[
        1
    ]


def _validate_value(name: str, value: int, limits: dict) -> None:
    if limits["flags"] & 0x0004:  # V4L2_CTRL_FLAG_READ_ONLY
        raise ValueError(f"{name} 为只读控制，不能写入")
    if not limits["min"] <= value <= limits["max"] or (value - limits["min"]) % max(1, limits["step"]):
        raise ValueError(
            f"{name} 不支持 {value}，范围 {limits['min']}～{limits['max']}，步长 {limits['step']}"
        )


def _set_and_confirm(fd: int, name: str, value: int, limits: dict) -> None:
    _validate_value(name, value, limits)
    ioctl(fd, _SET_CTRL, struct.pack(_VALUE_FORMAT, _CONTROLS[name], value))
    deadline = time.monotonic() + 0.25
    while True:
        actual = _get(fd, name)
        if actual == value:
            return
        if time.monotonic() >= deadline:
            raise ValueError(f"{name} 写入 {value}，读回仍为 {actual}")
        time.sleep(0.025)


def _result(path: str | Path, stage: str, target: dict) -> dict:
    return {
        "device_path": str(path),
        "stage": stage,
        "checked_at_unix": time.time(),
        "target": dict(target),
        "values": dict.fromkeys(_CONTROLS),
        "supported": [],
        "limits": {},
        "errors": {},
    }


def _read(fd: int, result: dict) -> None:
    result["supported"] = []
    result["limits"] = {}
    result["values"] = dict.fromkeys(_CONTROLS)
    for name in _CONTROLS:
        try:
            limits = _query(fd, name)
            if limits is not None:
                result["supported"].append(name)
                result["limits"][name] = limits
                result["values"][name] = _get(fd, name)
        except OSError as exc:
            result["errors"][name] = str(exc)


def _describe(result: dict) -> dict:
    errors = result["errors"]
    problems = [
        f"{key}: {value}"
        for key, value in errors.items()
        if key not in ("zoom_absolute", "exposure_auto", "exposure_absolute")
    ]
    focus_keys = [key for key in _FOCUS_KEYS if key in result["supported"]]
    for key in result["target"]:
        actual, expected = result["values"][key], result["target"][key]
        if key not in errors and actual != expected:
            problems.append(f"{key} 期望 {expected}，读回 {actual}")
    if problems:
        result.update(state="warning", message="对焦未确认：" + "；".join(problems))
    elif not focus_keys:
        result.update(state="unsupported", message="设备未提供 V4L2 对焦控制，已跳过固定对焦")
    else:
        details = []
        if "focus_auto" in focus_keys:
            details.append("自动对焦已开启" if result["values"]["focus_auto"] else "自动对焦已关闭")
        if "focus_absolute" in focus_keys and result["target"].get("focus_auto") != 1:
            details.append(f"手动对焦 {result['values']['focus_absolute']}")
        result.update(state="verified", message="读回确认：" + "，".join(details))
    return result


def validate_setting(setting: dict) -> None:
    if not isinstance(setting, dict) or set(setting) - {"mode", "value"}:
        raise ValueError("对焦设置必须为 mode/value 对象")
    if setting.get("mode") not in ("manual", "auto"):
        raise ValueError("请选择定焦或自动对焦")
    value = setting.get("value")
    if (setting["mode"] == "manual" or value is not None) and (type(value) is not int or value < 0):
        raise ValueError("定焦值必须为非负整数")


def validate_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("相机对焦配置必须按设备标识保存")
    for device_id, setting in settings.items():
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("对焦配置的相机标识无效")
        validate_setting(setting)


def default_setting(result: dict) -> dict | None:
    """新设备优先自动对焦，不将其他型号的数值套用到它的标尺上。"""
    manual = result["limits"].get("focus_absolute")
    value = manual["default"] if manual else None
    if "focus_auto" in result["supported"]:
        return {"mode": "auto", "value": value}
    if manual:
        return {
            "mode": "manual",
            "value": result["values"]["focus_absolute"]
            if result["values"]["focus_absolute"] is not None
            else value,
        }
    return None


def _target(setting: dict | None, result: dict) -> dict:
    if setting is None:
        return {}
    validate_setting(setting)
    if setting["mode"] == "auto":
        return {"focus_auto": 1}
    return {
        **({"focus_auto": 0} if "focus_auto" in result["supported"] else {}),
        "focus_absolute": setting["value"],
    }


def inspect_focus(path: str | Path, *, stage: str, setting: dict | None = None) -> dict:
    """只读控制值；读回确认不等于光学清晰度检测，也不证明集内全程不变。"""
    result = _result(path, stage, {})
    try:
        with _device(path) as fd:
            _read(fd, result)
    except OSError as exc:
        result["errors"]["device"] = str(exc)
    result["setting"] = setting or default_setting(result)
    result["target"] = _target(result["setting"], result)
    return _describe(result)


def fix_focus(path: str | Path, focus_value: int = DEFAULT_FOCUS_VALUE) -> dict:
    """显式固定某个数值的便捷入口；平台设备设置走 configure_focus。"""
    return configure_focus(path, {"mode": "manual", "value": focus_value})


def configure_focus(path: str | Path, setting: dict | None = None) -> dict:
    """先验证所有目标，再写设备。失败时恢复原模式，不能关掉 AF 后才发现值越界。"""
    if setting is not None:
        validate_setting(setting)
    result = _result(path, "open", {})
    try:
        with _device(path, writable=True) as fd:
            _read(fd, result)
            result["setting"] = dict(setting) if setting is not None else default_setting(result)
            result["target"] = _target(result["setting"], result)
            # 不提供任何对焦控制的设备正常跳过，不误称为定焦镜头。
            if not any(key in result["supported"] for key in _FOCUS_KEYS) and not result["errors"]:
                result["target"] = {}
                return _describe(result)
            original = dict(result["values"])
            written = []
            try:
                if "focus_absolute" in result["target"] and "focus_auto" in result["errors"]:
                    raise ValueError("自动对焦状态未确认，暂不写入手动对焦值")
                for name, value in result["target"].items():
                    if name not in result["limits"] or name in result["errors"]:
                        raise ValueError(f"设备不提供可用的 {name} 控制")
                    _validate_value(name, value, result["limits"][name])
                for name, value in result["target"].items():
                    limits = _query(fd, name)
                    if limits is None:
                        raise ValueError(f"{name} 控制已不可用")
                    written.append(name)
                    _set_and_confirm(fd, name, value, limits)
            except (OSError, ValueError) as exc:
                result["errors"]["setting"] = str(exc)
                for name in reversed(written):
                    if original[name] is None:
                        continue
                    try:
                        _set_and_confirm(fd, name, original[name], result["limits"][name])
                    except (OSError, ValueError) as restore:
                        result["errors"]["restore_" + name] = str(restore)
            _read(fd, result)
    except OSError as exc:
        result["errors"]["device"] = str(exc)
    return _describe(result)


def audit_warnings(checks: dict[str, dict]) -> list[str]:
    """首尾不一致或读回失败需要复核；曝光只记录，不将自动曝光变化当成对焦错误。"""
    warnings = []
    labels = {"left": "左相机", "right": "右相机", "head": "头部相机"}
    stages = {"opened": "打开时", "before": "本集开始前", "after": "本集结束后"}
    for role, entry in checks.items():
        label = labels.get(role, role)
        for stage, title in stages.items():
            result = entry.get(stage)
            if result and result["state"] == "warning":
                warnings.append(f"{label}{title}：{result['message']}")
        before, after = entry.get("before"), entry.get("after")
        if not before or not after:
            continue
        # 自动模式允许镜头焦点变化；仍检查 AF 开关和变焦是否发生意外改变。
        keys = ("focus_auto", "zoom_absolute")
        if (before.get("setting") or {}).get("mode") != "auto":
            keys += ("focus_absolute",)
        for key in keys:
            a, b = before["values"].get(key), after["values"].get(key)
            if a != b:
                warnings.append(f"{label}首尾 {key} 读数变化：{a} → {b}，请复核画面")
    return warnings
