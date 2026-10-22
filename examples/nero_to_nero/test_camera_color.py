"""白平衡只修改支持的开关，写入失败不能误报成功。"""

import errno
import struct

import pytest
from nero_platform import camera_color


@pytest.fixture
def white_balance(monkeypatch):
    state = {"automatic": 0, "flags": 0, "ignore_write": False}
    writes = []

    def ioctl(fd, request, raw):
        cid = struct.unpack_from("I", raw)[0]
        assert cid == camera_color.AUTO_WHITE_BALANCE
        if request == camera_color.QUERY_CTRL:
            return struct.pack("II32siiiiI2I", cid, 2, b"AWB", 0, 1, 1, 0, state["flags"], 0, 0)
        if request == camera_color.GET_CTRL:
            return struct.pack("Ii", cid, state["automatic"])
        assert request == camera_color.SET_CTRL
        writes.append(struct.unpack("Ii", raw)[1])
        if not state["ignore_write"]:
            state["automatic"] = writes[-1]
        return raw

    monkeypatch.setattr(camera_color, "ioctl", ioctl)
    return state, writes


def test_disabled_automatic_white_balance_is_enabled_and_read_back(white_balance):
    state, writes = white_balance
    result = camera_color.configure_white_balance("/fake/camera")
    assert result["state"] == "verified" and result["automatic"] == 1
    assert writes == [1] and state["automatic"] == 1
    camera_color.configure_white_balance("/fake/camera")
    assert writes == [1]  # 已开启时不反复重置相机算法。


def test_ignored_write_is_a_warning(white_balance):
    state, _ = white_balance
    state["ignore_write"] = True
    result = camera_color.configure_white_balance("/fake/camera")
    assert result["state"] == "warning" and result["automatic"] == 0


def test_readonly_and_unsupported_controls_are_not_written(white_balance, monkeypatch):
    state, writes = white_balance
    state["flags"] = 4
    assert camera_color.configure_white_balance("/fake/camera")["state"] == "warning"
    state["flags"] = 1
    assert camera_color.configure_white_balance("/fake/camera")["state"] == "unsupported"
    assert not writes

    def unsupported(*args):
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(camera_color, "ioctl", unsupported)
    assert camera_color.configure_white_balance("/fake/camera")["state"] == "unsupported"


def test_permission_failure_is_reported_and_fd_is_closed(white_balance, monkeypatch):
    def denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(camera_color, "ioctl", denied)
    assert camera_color.configure_white_balance("/fake/camera")["state"] == "warning"
    camera_color.os.close.assert_called_once_with(-1)
