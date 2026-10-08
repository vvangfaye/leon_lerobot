"""RGBD 分配预览：同帧拼接、独占管线、失败清理及采集互斥。"""

import threading
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from nero_platform import camera_preview, cameras
from nero_platform.app import create_app
from nero_platform.runtime import Runtime


@pytest.fixture
def preview(monkeypatch):
    runtime = Runtime()
    runtime.camera_inventory = {"femto:A": {"type": "rgbd", "backend": "orbbec", "serial": "A"}}
    runtime.cameras_scanned = True
    cam = Mock()
    rgb = np.full((36, 64, 3), [255, 0, 0], dtype=np.uint8)
    depth = np.full((36, 64, 1), 2.0, dtype=np.float32)
    cam.read_snapshot.return_value = {"rgb": rgb, "depth": depth}
    factory = Mock(return_value=cam)
    monkeypatch.setitem(camera_preview.FACTORIES, "orbbec", factory)
    return runtime, create_app(runtime), cam, factory


def test_side_by_side_is_one_snapshot_and_never_opens_from_image_get(preview):
    runtime, app, cam, factory = preview
    client = app.test_client()
    url = "/api/camera/femto:A/rgbd/frame"
    assert client.get(url).status_code == 404
    factory.assert_not_called()
    assert client.post("/api/camera/femto:A/preview/start").json["phase"] == "ready"
    assert client.post("/api/camera/femto:A/preview/start").json["phase"] == "ready"
    factory.assert_called_once()
    cam.read_snapshot.reset_mock()
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    decoded = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (36, 128, 3)
    assert np.allclose(decoded[18, 24], [0, 0, 255], atol=3), "左侧 RGB 红色不能反成蓝色"
    expected = camera_preview.depth_images.depth_preview(cam.read_snapshot.return_value["depth"])[18, 24]
    assert np.allclose(decoded[18, 88], expected[::-1], atol=3), "右侧使用同一帧深度伪彩色"
    cam.read_snapshot.assert_called_once()
    assert not runtime.dataset


def test_warmup_is_idempotent_across_windows_and_blocks_session_commands(preview):
    runtime, app, cam, factory = preview
    entered, finish = threading.Event(), threading.Event()

    def connect():
        entered.set()
        assert finish.wait(5)

    cam.connect.side_effect = connect
    results = []
    thread = threading.Thread(
        target=lambda: results.append(app.test_client().post("/api/camera/femto:A/preview/start"))
    )
    thread.start()
    try:
        assert entered.wait(5)
        client = app.test_client()
        assert client.get("/api/status").json["camera_previews"]["femto:A"]["phase"] == "opening"
        assert client.post("/api/camera/femto:A/preview/start").json["phase"] == "opening"
        assert client.post("/api/cameras/detect/depth").status_code == 409
        assert runtime.operation_busy()
        factory.assert_called_once()
    finally:
        finish.set()
        thread.join(5)
    assert results[0].status_code == 200
    assert not runtime.device_busy


def test_failed_warmup_cleans_handle_and_retry_can_reopen(preview):
    runtime, app, cam, factory = preview
    client = app.test_client()
    cam.connect.side_effect = RuntimeError("相机被其他程序占用")
    result = client.post("/api/camera/femto:A/preview/start")
    assert result.status_code == 503
    assert "其他程序占用" in result.json["msg"]
    assert not runtime.preview_cams and not runtime.device_busy
    cam.disconnect.assert_called_once()
    cam.connect.side_effect = None
    assert client.post("/api/camera/femto:A/preview/start").status_code == 200
    assert factory.call_count == 2
    assert client.post("/api/camera/femto:A/preview/start", json={"restart": True}).status_code == 200
    assert cam.disconnect.call_count == 2, "手动重新打开先释放旧管线"


def test_close_failure_retains_owner_and_prevents_duplicate_pipeline(preview):
    runtime, app, cam, factory = preview
    client = app.test_client()
    client.post("/api/camera/femto:A/preview/start")
    cam.disconnect.side_effect = RuntimeError("管线未退出")
    with pytest.raises(RuntimeError, match="管线未退出"):
        cameras.close_preview_cams(runtime)
    assert runtime.preview_cams["femto:A"] is cam
    assert client.post("/api/camera/femto:A/preview/start").status_code == 503
    factory.assert_called_once()
    cam.disconnect.side_effect = None
    cameras.close_preview_cams(runtime)
    assert not runtime.preview_cams
    assert runtime.preview_states["femto:A"]["phase"] == "closed"
    assert client.get("/api/camera/femto:A/rgbd/frame").status_code == 404


def test_selective_rescan_does_not_close_other_type_and_session_blocks_open(preview):
    runtime, app, cam, _ = preview
    client = app.test_client()
    client.post("/api/camera/femto:A/preview/start")
    rgb = Mock()
    runtime.preview_cams["usb"] = rgb
    cameras.close_preview_cams(runtime, kind="rgb")
    rgb.disconnect.assert_called_once()
    cam.disconnect.assert_not_called()
    cameras.close_preview_cams(runtime, kind="depth")
    cam.disconnect.assert_called_once()
    runtime.state["connected"] = True
    assert client.post("/api/camera/femto:A/preview/start").status_code == 409
    runtime.state["connected"] = False
    runtime.camera_inventory["future:R"] = {"type": "rgbd", "backend": "future"}
    assert not client.get("/api/cameras").json["cameras"]["future:R"]["preview_supported"]
    assert client.post("/api/camera/future:R/preview/start").status_code == 400
