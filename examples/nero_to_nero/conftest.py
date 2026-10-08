"""平台测试永远不枚举或连接真实 Femto；用例可显式注入模拟 SDK。"""

import errno
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from nero_platform import azure_kinect, camera_focus, femto, realsense


@pytest.fixture(autouse=True)
def simulated_focus_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认模拟无 V4L2 对焦控制，禁止测试为真实相机改参数；专门用例可注入模拟 ioctl。"""

    @contextmanager
    def device(path, *, writable=False):
        yield -1

    monkeypatch.setattr(camera_focus, "_device", device)
    monkeypatch.setattr(camera_focus, "ioctl", Mock(side_effect=OSError(errno.EINVAL, "unsupported control")))


@pytest.fixture(autouse=True)
def block_real_femto(monkeypatch: pytest.MonkeyPatch) -> None:
    # 检测通过 sysfs，可被每个用例替换；默认任何真实 SDK 句柄都禁止打开。
    monkeypatch.setattr(azure_kinect, "USB_ROOT", Path("/nonexistent-test-usb"))
    if azure_kinect.k4a is not None:
        monkeypatch.setattr(
            azure_kinect.k4a, "PyK4A", Mock(side_effect=AssertionError("测试禁止打开真实 Kinect"))
        )
    if femto.ob is not None:
        monkeypatch.setattr(femto.ob, "Context", Mock(side_effect=AssertionError("测试禁止访问真实 Femto")))
        monkeypatch.setattr(femto.ob, "Pipeline", Mock(side_effect=AssertionError("测试禁止打开真实 Femto")))

    if realsense.rs is not None:
        monkeypatch.setattr(
            realsense.rs, "context", Mock(side_effect=AssertionError("测试禁止访问真实 RealSense"))
        )
        monkeypatch.setattr(
            realsense.rs, "pipeline", Mock(side_effect=AssertionError("测试禁止打开真实 RealSense"))
        )
