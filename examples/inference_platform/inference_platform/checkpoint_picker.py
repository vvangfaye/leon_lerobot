"""本地桌面目录选择；返回路径，不上传 checkpoint 文件。"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path


class DirectoryPicker:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __call__(self, initial: Path, *, title: str = "选择 checkpoint 文件夹") -> Path | None:
        if not self.lock.acquire(blocking=False):
            raise ValueError("文件夹选择窗口已打开，请先完成或取消选择")
        try:
            executable = shutil.which("zenity")
            if not executable or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise ValueError("无法打开文件夹窗口：请在本机桌面启动平台，并安装 zenity")
            start = initial if initial.is_dir() else Path.home()
            try:
                result = subprocess.run(
                    [executable, "--file-selection", "--directory",
                     f"--title={title}", f"--filename={start.resolve()}{os.sep}"],
                    capture_output=True, text=True, timeout=120, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError("文件夹选择超时，请重新点击手动选择") from exc
            if result.returncode == 1:
                return None
            if result.returncode != 0 or not result.stdout.strip():
                raise ValueError("文件夹窗口未能完成选择，请检查本机桌面后重试")
            return Path(result.stdout.rstrip("\r\n"))
        finally:
            self.lock.release()
