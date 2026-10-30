import time

from examples.nero_rl.preview_head import MJPEGFrames, create_app


def test_mjpeg_chunk_boundaries_and_latest_frame():
    frames = MJPEGFrames()
    frames.feed(b"garbage\xff")
    frames.feed(b"\xd8first\xff")
    assert frames.latest()[0] is None
    frames.feed(b"\xd9\xff\xd8second\xff\xd9")
    value, age, count = frames.latest()
    assert value == b"\xff\xd8second\xff\xd9"
    assert count == 2 and age < 1


def test_preview_requires_auth_and_rejects_stale_images():
    frames = MJPEGFrames()
    token = "x" * 32
    app = create_app(frames, token, f"http://localhost:5001/?token={token}")
    client = app.test_client()
    assert client.get("/frame.jpg").status_code == 403
    assert client.get("/?token=wrong").status_code == 403
    page = client.get(f"/?token={token}")
    assert page.status_code == 200 and "原采集控制页" in page.text
    assert client.get("/frame.jpg").status_code == 503
    frames.feed(b"\xff\xd8frame\xff\xd9")
    assert client.get("/frame.jpg").data == b"\xff\xd8frame\xff\xd9"
    assert client.get("/status").json["live"]
    frames.timestamp = time.monotonic() - 2
    assert not client.get("/status").json["live"]
    assert client.get("/frame.jpg").status_code == 503
