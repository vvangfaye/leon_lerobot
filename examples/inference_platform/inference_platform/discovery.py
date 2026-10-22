"""Linux attachment identities, independent of display order and role choices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def usb_identity(path: Path) -> dict:
    for parent in (path, *path.parents):
        if (parent / "idVendor").is_file():
            return {"usb_path": parent.name, **{key: read(parent / key) for key in
                    ("serial", "idVendor", "idProduct", "busnum", "devnum")}}
    return {"usb_path": str(path)}


def can_inventory() -> dict:
    devices = {}
    for path in sorted(Path("/sys/class/net").iterdir()):
        if read(path / "type") != "280":
            continue
        identity = {"ifindex": read(path / "ifindex"), **usb_identity((path / "device").resolve())}
        devices[path.name] = {
            "id": path.name, "kind": "arm", "identity": identity,
            "attachment": fingerprint(identity), "up": bool(int(read(path / "flags") or "0", 16) & 1),
        }
    return devices


def camera_attachment(info: dict) -> str:
    if info["backend"] == "opencv":
        path = Path(info["path"])
        stat = path.stat()
        device = Path("/sys/class/video4linux") / path.resolve().name / "device"
        return fingerprint({"device": str(path.resolve()), "inode": stat.st_ino,
                            "ctime": stat.st_ctime_ns, **usb_identity(device.resolve())})
    serial = info["serial"]
    # Device numbers change after USB reconnection even if its serial is stable.
    identities = [usb_identity(path) for path in Path("/sys/bus/usb/devices").glob("*")
                  if read(path / "serial") == serial]
    return fingerprint({"backend": info["backend"], "serial": serial, "usb": identities})


def rgb_inventory() -> dict:
    devices = {}
    paths = sorted(Path("/dev/v4l/by-id").glob("*video-index0"))
    resolved = {path.resolve() for path in paths}
    # Some cameras expose no USB serial/by-id link. Index 0 is the capture node.
    for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
        path = Path("/dev") / entry.name
        if read(entry / "index") == "0" and path.resolve() not in resolved:
            paths.append(path)
    for path in paths:
        if any(word in path.name.lower() for word in ("realsense", "femto", "azure_kinect")):
            continue
        identity = path.name.removesuffix("-video-index0")
        info = {"id": identity, "kind": "camera", "type": "rgb", "backend": "opencv", "path": str(path)}
        usb = usb_identity((Path("/sys/class/video4linux") / path.resolve().name / "device").resolve())
        info["serial"] = usb.get("serial", "")
        try:
            info["attachment"] = camera_attachment(info)
            devices[identity] = info
        except OSError:
            continue  # Unplugged during enumeration.
    return devices


def attachment_present(info: dict) -> bool:
    try:
        if info["kind"] == "arm":
            current = can_inventory().get(info["id"])
            return current is not None and current["up"] and current["attachment"] == info["attachment"]
        return camera_attachment(info) == info["attachment"]
    except (OSError, ValueError):
        return False
