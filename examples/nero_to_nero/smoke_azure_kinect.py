"""显式运行的 Azure Kinect 相机冒烟；不连接机械臂，不改平台配置。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from nero_platform import azure_kinect, depth_images, resolution


def run_smoke(camera: azure_kinect.AzureKinectCamera, seconds: float) -> tuple[dict, dict | None]:
    report: dict = {"result": "failed", "frames": 0, "metadata": {}, "errors": []}
    sample = None
    stamps, skew, ages = [], [], []
    minimum, maximum = float("inf"), 0.0
    try:
        camera.connect()
        report["metadata"] = dict(camera.metadata)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            current = camera.read_snapshot()
            stamp = current["capture_timestamp"]
            if stamps and stamp == stamps[-1]:
                time.sleep(0.002)
                continue
            sample = current
            stamps.append(stamp)
            ages.append(current["age_ms"])
            depth = current["depth"]
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError("深度不是有效米值")
            valid = depth[depth > 0]
            if valid.size:
                minimum, maximum = min(minimum, float(valid.min())), max(maximum, float(valid.max()))
            if current.get("rgb_depth_skew_ms") is not None:
                skew.append(current["rgb_depth_skew_ms"])
        if len(stamps) < 3 or not skew or not np.isfinite(minimum):
            raise RuntimeError("持续更新的完整 RGBD、有效深度或时间戳不足")
        report["result"] = "passed"
    except Exception as exc:
        report["errors"].append(str(exc) or type(exc).__name__)
    finally:
        report.update(
            metadata=dict(camera.metadata),
            frames=len(stamps),
            effective_rgbd_fps=(len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
            max_age_ms=max(ages, default=None),
            max_rgb_depth_skew_ms=max(skew, default=None),
            depth_range_m=[minimum, maximum] if np.isfinite(minimum) else None,
        )
        try:
            camera.disconnect()
        except Exception as exc:
            report["errors"].append("关闭失败：" + str(exc))
            report["result"] = "failed"
    return report, sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--serial", default="")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, choices=(5, 15, 30), default=30)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview", type=Path, help="可选：保存最后一组 RGB/深度伪彩色并排图")
    args = parser.parse_args()
    if not args.run:
        parser.error("真机操作必须显式传入 --run")
    if not 0 < args.seconds <= 60 or args.scale not in resolution.scales(args.width, args.height):
        parser.error("seconds 必须在 0～60 内，scale 必须是该尺寸支持的逐次减半档位")
    setting = {"width": args.width, "height": args.height, "scale": args.scale}
    camera = azure_kinect.AzureKinectCamera(
        color_size=(args.width, args.height),
        fps=args.fps,
        serial=args.serial,
        output_size=resolution.output_size(setting),
    )
    report, sample = run_smoke(camera, args.seconds)
    if sample is not None and args.preview:
        image = np.concatenate((sample["rgb"], depth_images.depth_preview(sample["depth"])), axis=1)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("预览图保存失败")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if report["result"] == "passed" else 1)


if __name__ == "__main__":
    main()
