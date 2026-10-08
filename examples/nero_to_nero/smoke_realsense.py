"""RealSense 真机冒烟：仅相机，不连接 CAN、不修改平台配置或正式数据集。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from nero_platform import depth_images, realsense, resolution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    if not args.run or not 0 < args.seconds <= 60:
        parser.error("请显式传入 --run；seconds 必须在 0～60 内")
    if args.width < 2 or args.height < 2 or args.scale not in resolution.scales(args.width, args.height):
        parser.error("分辨率/缩放无效")
    setting = {"width": args.width, "height": args.height, "scale": args.scale}
    cam = realsense.RealSenseCamera(
        serial=args.serial,
        fps=args.fps,
        color_size=(args.width, args.height),
        output_size=resolution.output_size(setting),
    )
    report: dict = {"result": "failed", "errors": []}
    samples = []
    last = None
    try:
        cam.connect()
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            current = cam.read_snapshot()
            if last is None or current["capture_timestamp"] != last["capture_timestamp"]:
                depth = current["depth"]
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError("深度不是有效米值")
                samples.append({k: v for k, v in current.items() if k not in ("rgb", "depth")})
                last = current
            time.sleep(0.002)
        if len(samples) < 3 or not np.any(last["depth"] > 0):
            raise RuntimeError("持续更新的 RGBD 或有效深度不足")
        measured = (len(samples) - 1) / (samples[-1]["capture_timestamp"] - samples[0]["capture_timestamp"])
        report["effective_rgbd_fps"] = measured
        if measured < args.fps * 0.9:
            raise RuntimeError(f"实测 RGBD {measured:.1f} FPS，低于目标 {args.fps} FPS 的 90%")
        report["result"] = "passed"
    except Exception as exc:
        report["errors"].append(str(exc) or type(exc).__name__)
    finally:
        report.update(
            metadata=cam.metadata.copy(),
            frames=len(samples),
            max_age_ms=max((s["age_ms"] for s in samples), default=None),
            samples=samples,
        )
        try:
            cam.disconnect()
        except Exception as exc:
            report["errors"].append("关闭失败：" + str(exc))
            report["result"] = "failed"
    if last is not None and args.preview:
        paired = np.concatenate((last["rgb"], depth_images.depth_preview(last["depth"])), axis=1)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), cv2.cvtColor(paired, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("预览图保存失败")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["result"] == "passed" else 1)


if __name__ == "__main__":
    main()
