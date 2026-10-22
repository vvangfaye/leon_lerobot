#!/usr/bin/env python3
"""Run the local LeRobot training platform without owning trainer lifetimes."""

import argparse
import sys
from pathlib import Path

from training_platform import configuration
from training_platform.app import create_app
from training_platform.jobs import Jobs
from werkzeug.serving import WSGIRequestHandler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


class QuietRequests(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.command == "GET" and self.path.startswith("/api/") and str(code) == "200":
            return
        super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--state-dir", type=Path, help="任务记录目录；默认 outputs/training_platform")
    parser.add_argument("--collection-url", default="http://127.0.0.1:5000")
    parser.add_argument("--inference-url", default="http://127.0.0.1:7862")
    parser.add_argument("--start", metavar="TASK_ID", help="使用已保存配置启动任务，不启动网页服务")
    parser.add_argument("--checkpoint", help="--start 时恢复的完整 checkpoint")
    args = parser.parse_args()
    if args.start:
        manager = Jobs(ROOT, args.state_dir)
        job = manager.get(args.start)
        config = job.get("draft_config") or job["config"]
        checkpoint = args.checkpoint or job.get("draft_checkpoint", job.get("resume_from"))
        checked = configuration.validate_config(config, "resume" if checkpoint else "new", checkpoint)
        if not checked["valid"]:
            parser.error("；".join(checked["errors"]))
        try:
            result = manager.launch(args.start, checkpoint=checkpoint)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(f"训练任务 {result['id']} 已启动；日志：{result['log_path']}", flush=True)
        return
    app = create_app(state_dir=args.state_dir, collection_url=args.collection_url, inference_url=args.inference_url)
    print(f"Training platform: http://127.0.0.1:{args.port}", flush=True)
    print("关闭网页或平台不会停止后台训练；外部终端训练只读监控。", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False,
            threaded=True, request_handler=QuietRequests)


if __name__ == "__main__":
    main()
