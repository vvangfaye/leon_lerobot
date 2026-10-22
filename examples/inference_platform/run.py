#!/usr/bin/env python3
"""Launch the NERO / Piper inference platform (π0.5 inference, devices and recording)."""

import argparse
import atexit
import signal

from inference_platform.app import create_platform_app
from werkzeug.serving import WSGIRequestHandler


class PreviewRequestHandler(WSGIRequestHandler):
    """Keep frame/status polling out of the terminal; retain errors and actions."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if self.path.split("?")[0] in (
            "/api/devices", "/api/devices/frame", "/api/devices/jog-heartbeat",
            "/api/inference", "/api/inference/heartbeat",
        ) and str(code) in ("200", "204"):
            return
        super().log_request(code, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    app = create_platform_app()
    manager = app.extensions["devices"]
    engine = app.extensions["inference"]
    atexit.register(manager.close)
    atexit.register(engine.close)

    def shutdown(signum: int, frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    print(f"Inference platform: http://127.0.0.1:{args.port}", flush=True)
    print("π0.5: connect → confirm devices → load checkpoint → single / continuous execution. Stop: Esc.", flush=True)
    try:
        app.run(
            host="127.0.0.1", port=args.port, debug=False, use_reloader=False,
            threaded=True, request_handler=PreviewRequestHandler,
        )
    finally:
        engine.close()
        manager.close()


if __name__ == "__main__":
    main()
