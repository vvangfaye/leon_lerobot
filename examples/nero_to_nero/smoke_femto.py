"""显式运行的 Femto RGBD 冒烟测试：只连接相机，不导入机械臂模块、不保存图像。"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from nero_platform import femto, resolution, rgbd_timing


def run_smoke(camera: femto.FemtoCamera, seconds: float) -> dict:
    report = {
        "frames": 0,
        "metadata": {},
        "timestamp_stats": {},
        "frame_index_issues": 0,
        "errors": [],
        "samples": [],
        "depth_range_m": None,
        "sensor_sync_verified": False,
    }
    previous_capture, previous_indices = None, None
    minimum, maximum = math.inf, -math.inf
    try:
        camera.connect()
        report["metadata"] = dict(camera.metadata)
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            sample = camera.read_snapshot()
            if sample["capture_timestamp"] == previous_capture:
                time.sleep(min(0.01, 0.5 / camera.fps))
                continue
            previous_capture = sample["capture_timestamp"]
            width, height = camera.output_size
            if sample["rgb"].shape != (height, width, 3) or sample["rgb"].dtype != np.uint8:
                raise ValueError("RGB 尺寸或类型错误")
            depth = sample["depth"]
            if (
                depth.shape != (height, width, 1)
                or depth.dtype != np.float32
                or not np.isfinite(depth).all()
                or np.any(depth < 0)
            ):
                raise ValueError("深度尺寸、类型或米值无效")
            valid = depth[depth > 0]
            if valid.size:
                minimum, maximum = min(minimum, float(valid.min())), max(maximum, float(valid.max()))
            indices = (sample.get("rgb_frame_index"), sample.get("depth_frame_index"))
            if previous_indices is not None and any(
                current is None or old is None or current <= old
                for current, old in zip(indices, previous_indices, strict=True)
            ):
                report["frame_index_issues"] += 1
            previous_indices = indices
            check = rgbd_timing.checked_sample(sample)
            if check is None:
                raise ValueError("快照缺少新版时间戳检查结果")
            rgbd_timing.add_stats(report["timestamp_stats"], check)
            rgbd_timing.add_receive_stats(report["timestamp_stats"], sample)
            report["frames"] += 1
            details = {key: value for key, value in sample.items() if key not in ("rgb", "depth")}
            if len(report["samples"]) < 6:
                report["samples"].append(details)
            else:
                report["samples"][-1] = details
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if report["frames"] < 3:
            raise RuntimeError("完整且持续更新的 RGBD 帧不足 3 帧")
    except Exception as exc:
        report["errors"].append(str(exc))
    finally:
        try:
            camera.disconnect()
        except Exception as exc:
            report["errors"].append("关闭相机失败：" + str(exc))
    report["depth_range_m"] = [minimum, maximum] if math.isfinite(minimum) else None
    report["warnings"] = rgbd_timing.warnings(report["timestamp_stats"], camera.fps)
    if report["frame_index_issues"]:
        report["warnings"].append("帧号缺失、重复或倒退，请检查设备元数据")
    if report["depth_range_m"] is None:
        report["warnings"].append("未收到非零深度，请检查场景与测距")
    # 初始一帧等待时钟推进是正常预热；其他未验证帧需要人工复核。
    stats = report["timestamp_stats"]
    checked = stats.get("timestamp_checked_frames", 0)
    warmup_only = (
        stats.get("timestamp_unverified_frames", 0) <= 1
        and report["samples"]
        and report["samples"][0]["timestamp_status"] == "warming_up"
    )
    needs_review = (
        not checked
        or (stats.get("timestamp_unverified_frames", 0) and not warmup_only)
        or report["frame_index_issues"]
        or report["depth_range_m"] is None
        or (stats.get("max_rgb_depth_skew_ms") or 0) > 1000 / camera.fps
        or (stats.get("max_rgb_depth_receive_skew_ms") or 0) > rgbd_timing.receive_skew_limit_ms(camera.fps)
    )
    report["result"] = "failed" if report["errors"] else ("review" if needs_review else "passed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="明确打开真实头部相机；请先结束平台采集")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--serial", default="")
    parser.add_argument("--output", type=Path, help="可选：保存元数据 JSON 到新文件，不覆盖已有报告")
    args = parser.parse_args(argv)
    if not args.run:
        print("未连接设备。真机检查请显式加 --run；只打开 Femto，不连接机械臂或左右相机。")
        return 0
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60 or args.fps <= 0:
        parser.error("seconds 需要在 1～60 秒，fps 需要为正整数")
    settings = resolution.defaults()
    settings["head"] = {"width": args.width, "height": args.height, "scale": args.scale}
    try:
        resolution.validate(settings)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output and args.output.exists():
        parser.error("报告文件已存在，请换新路径；尚未连接相机")
    camera = femto.FemtoCamera(
        color_size=(args.width, args.height),
        output_size=resolution.output_size(settings["head"]),
        fps=args.fps,
        serial=args.serial,
    )
    report = run_smoke(camera, args.seconds)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)
    print(
        "结果："
        + {
            "passed": "图像与时间戳检查通过；不等同于同步曝光认证",
            "review": "图像已检查，元数据或测距仍需复核",
            "failed": "相机检查失败，见 errors",
        }[report["result"]]
    )
    return {"passed": 0, "failed": 1, "review": 2}[report["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
