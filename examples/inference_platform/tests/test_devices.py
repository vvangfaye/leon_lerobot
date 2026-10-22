"""Device lifecycle and failure paths exercised without opening physical devices."""

import copy
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_platform.app import create_platform_app
from inference_platform.devices import DeviceManager
from inference_platform.hardware import ArmReader, FreshFeedback


class FakeReader:
    def __init__(self, info: dict, robot: str) -> None:
        self.info, self.robot = info, robot
        self.connected = False
        self.closed = False
        self.fail = False
        self.fail_connect = False
        self.fail_close = False

    def connect(self) -> None:
        self.connected = True
        if self.fail_connect:
            raise RuntimeError("busy camera")

    def sample(self) -> tuple[dict, bytes | None]:
        if self.fail:
            raise RuntimeError("lost feedback")
        return {
            "ready": True, "message": "fresh", "joint_degrees": [0] * (7 if self.robot == "nero" else 6),
            "gripper_m": 0.01, "width": 640, "height": 480,
        }, b"jpeg-fixture" if self.info["kind"] == "camera" else None

    def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")
        self.closed = True


class DeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.lease = Path(temporary.name) / "devices.lock"
        self.inventory = {
            key: {"id": key, "kind": kind, "up": True, "backend": "fake", "attachment": "one"}
            for key, kind in (("can0", "arm"), ("can1", "arm"), ("cam0", "camera"), ("cam1", "camera"))
        }
        self.present = True
        self.readers: dict[str, FakeReader] = {}
        self.config = {"robot": "nero", "arms": "single", "space": "eef"}
        self.manager = self.make_manager()
        self.client = create_platform_app(candidates=[], devices=self.manager).test_client()

    def make_manager(self) -> DeviceManager:
        manager = DeviceManager(
            scanner=lambda: (copy.deepcopy(self.inventory), {}),
            factory=self.factory, present=lambda info: self.present, lease_path=self.lease,
        )
        self.addCleanup(manager.close)
        return manager

    def factory(self, info: dict, robot: str) -> FakeReader:
        reader = FakeReader(info, robot)
        self.readers[info["id"]] = reader
        return reader

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail(f"condition timed out: {self.manager.status()}")
            time.sleep(0.01)

    def scan(self) -> None:
        self.manager.scan(self.config)
        self.wait_for(lambda: not self.manager.status()["scanning"])

    def open(self, identity: str) -> None:
        self.manager.open(identity, self.manager.revision)
        self.wait_for(lambda: self.manager.workers[identity].snapshot()["phase"] == "ready")

    def assign(self, identity: str, role: str) -> None:
        self.manager.assign(identity, role, self.manager.revision)

    def ready_mapping(self) -> None:
        self.scan()
        self.open("can0")
        self.open("cam0")
        self.assign("can0", "left_follower")
        self.assign("cam0", "head")

    def post(self, action: str, **data: object) -> object:
        return self.client.post(
            f"/api/devices/{action}", json={"revision": self.manager.revision, **data},
            headers={"X-Platform-Request": "1"},
        )

    def test_startup_and_scan_do_not_open_hardware(self) -> None:
        self.assertEqual(self.client.get("/api/devices").json["devices"], [])
        self.assertEqual(self.readers, {})
        self.scan()
        self.assertEqual(len(self.manager.status()["devices"]), 4)
        self.assertEqual(self.readers, {})

    def test_role_conflicts_stale_revision_and_incomplete_mapping_rejected(self) -> None:
        self.scan()
        old = self.manager.revision
        self.assign("can0", "left_follower")
        with self.assertRaisesRegex(ValueError, "列表已变化"):
            self.manager.assign("cam0", "head", old)
        with self.assertRaisesRegex(ValueError, "另一台"):
            self.assign("can1", "left_follower")
        with self.assertRaises(ValueError):
            self.assign("can0", "right")
        with self.assertRaises(ValueError):
            self.manager.confirm(self.manager.revision, True)
        with self.assertRaises(ValueError):
            self.manager.confirm(self.manager.revision, False)

    def test_confirmation_and_disconnect_release_and_require_new_confirmation(self) -> None:
        self.ready_mapping()
        self.manager.confirm(self.manager.revision, True)
        self.assertTrue(self.manager.status()["confirmed"])
        response = self.client.get("/api/devices/frame", query_string={"id": "cam0", "revision": self.manager.revision})
        self.assertEqual(response.data, b"jpeg-fixture")
        self.assertEqual(response.mimetype, "image/jpeg")
        self.manager.disconnect(self.manager.revision)
        self.assertIsNone(self.manager.frame("cam0", self.manager.revision))
        self.wait_for(lambda: not self.manager.status()["active"])
        self.assertTrue(all(reader.closed for reader in self.readers.values()))
        self.assertIsNone(self.manager.lease)
        self.open("cam0")
        self.assertFalse(self.manager.status()["confirmed"])

    def test_transient_failure_invalidates_confirmation_even_if_it_recovers_between_polls(self) -> None:
        self.ready_mapping()
        self.manager.confirm(self.manager.revision, True)
        self.readers["cam0"].fail = True
        worker = self.manager.workers["cam0"]
        self.wait_for(lambda: worker.snapshot()["phase"] == "error")
        self.assertIsNone(worker.frame())
        self.readers["cam0"].fail = False
        self.wait_for(lambda: worker.snapshot()["phase"] == "ready")
        self.assertFalse(self.manager.status()["confirmed"])

    def test_hot_unplug_drops_frames_and_releases_reader(self) -> None:
        self.ready_mapping()
        self.manager.confirm(self.manager.revision, True)
        self.present = False
        self.wait_for(lambda: not self.manager.status()["active"])
        self.assertFalse(self.manager.status()["confirmed"])
        self.assertIsNone(self.manager.frame("cam0", self.manager.revision))
        self.assertTrue(self.readers["cam0"].closed)

    def test_second_manager_cannot_open_owned_devices(self) -> None:
        self.ready_mapping()
        other = self.make_manager()
        other.scan(self.config)
        self.wait_for(lambda: not other.scanning)
        with self.assertRaisesRegex(ValueError, "另一推理平台"):
            other.open("can0", other.revision)
        self.manager.close()
        other.open("can0", other.revision)

    def test_failed_connect_is_closed_and_can_be_retried(self) -> None:
        self.scan()
        def failing_factory(info: dict, robot: str) -> FakeReader:
            reader = self.factory(info, robot)
            reader.fail_connect = True
            return reader
        self.manager.factory = failing_factory
        self.manager.open("cam0", self.manager.revision)
        self.wait_for(lambda: self.manager.workers["cam0"].closed)
        self.assertTrue(self.readers["cam0"].closed)
        self.manager.disconnect(self.manager.revision, "cam0")
        self.assertEqual(self.manager.workers["cam0"].snapshot()["phase"], "closed")
        self.manager.factory = self.factory
        self.open("cam0")

    def test_close_failure_retains_lease_and_blocks_reopening(self) -> None:
        self.ready_mapping()
        self.readers["cam0"].fail_close = True
        self.manager.disconnect(self.manager.revision)
        self.wait_for(lambda: self.manager.workers["cam0"].snapshot()["phase"] == "release_failed")
        self.assertTrue(self.manager.status()["active"])
        self.assertIsNotNone(self.manager.lease)
        with self.assertRaises(ValueError):
            self.manager.open("cam0", self.manager.revision)
        # Test cleanup emulates process exit after a driver release failure.
        self.manager.lease.close()
        self.manager.lease = None

    def test_slow_device_does_not_block_status_or_other_devices(self) -> None:
        self.scan()
        gate = threading.Event()
        self.addCleanup(gate.set)
        original = self.factory
        def slow_factory(info: dict, robot: str) -> FakeReader:
            reader = original(info, robot)
            if info["id"] == "cam0":
                reader.connect = lambda: gate.wait(2)
            return reader
        self.manager.factory = slow_factory
        self.manager.open("cam0", self.manager.revision)
        self.open("can0")
        start = time.monotonic()
        self.assertEqual(self.client.get("/api/devices").status_code, 200)
        self.assertLess(time.monotonic() - start, 0.2)
        self.manager.disconnect(self.manager.revision)
        gate.set()
        self.wait_for(lambda: not self.manager.status()["active"])

    def test_dual_arm_requires_both_roles_and_rescan_requires_release(self) -> None:
        self.config.update(robot="piper", arms="dual", space="joint")
        self.scan()
        self.open("can0")
        self.open("cam0")
        self.assign("can0", "left_follower")
        self.assign("cam0", "head")
        with self.assertRaises(ValueError):
            self.manager.confirm(self.manager.revision, True)
        with self.assertRaises(ValueError):
            self.manager.scan({**self.config, "robot": "nero"})
        self.open("can1")
        self.assign("can1", "right_follower")
        self.manager.confirm(self.manager.revision, True)
        self.assertTrue(self.manager.status()["confirmed"])
        self.assertEqual(self.readers["can0"].robot, "piper")

    def test_all_four_identities_are_available_even_for_single_arm(self) -> None:
        self.inventory.update({
            key: {"id": key, "kind": "arm", "up": True, "backend": "fake", "attachment": "one"}
            for key in ("can2", "can3")
        })
        self.scan()
        roles = ("left_leader", "right_leader", "left_follower", "right_follower")
        for i, role in enumerate(roles):
            self.open(f"can{i}")
            self.assign(f"can{i}", role)
        self.open("cam0")
        self.assign("cam0", "head")
        self.assertEqual(set(self.manager.roles.values()), {*roles, "head"})
        with self.assertRaisesRegex(ValueError, "恰好一个"):
            self.manager.confirm(self.manager.revision, True)
        self.assign("can2", "")
        self.manager.confirm(self.manager.revision, True)
        self.assertTrue(self.manager.status()["confirmed"])
        self.assertEqual(self.manager.status()["schema_version"], 4)
        # Names record identity; assigning a leader never enables or changes mode.
        self.assertTrue(all(reader.connected for reader in self.readers.values()))

    def test_leaders_cannot_substitute_for_inference_followers(self) -> None:
        self.ready_mapping()
        self.assign("can0", "left_leader")
        self.open("can1")
        self.assign("can1", "right_leader")
        with self.assertRaisesRegex(ValueError, "从臂"):
            self.manager.confirm(self.manager.revision, True)
        for old_role in ("single", "left", "right"):
            with self.assertRaises(ValueError):
                self.assign("can0", old_role)

    def test_usage_and_space_change_keep_readers_and_roles_but_require_confirmation(self) -> None:
        self.ready_mapping()
        self.manager.confirm(self.manager.revision, True)
        original_workers = dict(self.manager.workers)
        original_roles = dict(self.manager.roles)
        old_revision = self.manager.revision
        response = self.post("configure", config={**self.config, "arms": "dual", "space": "joint"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["confirmed"])
        self.assertEqual(self.manager.workers, original_workers)
        self.assertEqual(self.manager.roles, original_roles)
        self.assertFalse(any(reader.closed for reader in self.readers.values()))
        self.assertEqual(self.manager.frame("cam0", self.manager.revision), b"jpeg-fixture")
        with self.assertRaisesRegex(ValueError, "列表已变化"):
            self.manager.configure(self.config, old_revision)
        with self.assertRaises(ValueError):
            self.manager.confirm(self.manager.revision, True)
        self.open("can1")
        self.assign("can1", "right_follower")
        self.manager.confirm(self.manager.revision, True)
        self.manager.configure(self.config, self.manager.revision)
        # Do not guess which physical follower the user wants for single-arm use.
        with self.assertRaises(ValueError):
            self.manager.confirm(self.manager.revision, True)
        self.assign("can1", "")
        self.manager.confirm(self.manager.revision, True)

    def test_robot_change_releases_only_arms_and_reconnect_uses_new_driver(self) -> None:
        self.ready_mapping()
        original_arm = self.readers["can0"]
        original_camera = self.readers["cam0"]
        roles = dict(self.manager.roles)
        response = self.post("configure", config={**self.config, "robot": "piper"})
        self.assertEqual(response.status_code, 200)
        self.wait_for(lambda: original_arm.closed)
        self.assertFalse(original_camera.closed)
        self.assertEqual(self.manager.roles, roles)
        self.assertEqual(self.manager.frame("cam0", self.manager.revision), b"jpeg-fixture")
        self.open("can0")
        self.assertEqual(self.readers["can0"].robot, "piper")
        self.assertIs(self.readers["cam0"], original_camera)
        self.manager.confirm(self.manager.revision, True)

    def test_robot_change_never_reopens_an_arm_that_failed_release(self) -> None:
        self.ready_mapping()
        self.readers["can0"].fail_close = True
        self.manager.configure({**self.config, "robot": "piper"}, self.manager.revision)
        self.wait_for(lambda: self.manager.workers["can0"].snapshot()["phase"] == "release_failed")
        with self.assertRaisesRegex(ValueError, "尚未释放"):
            self.manager.open("can0", self.manager.revision)
        self.assertFalse(self.readers["cam0"].closed)
        self.readers["can0"].fail_close = False
        self.manager.workers["can0"].closed = True  # release-failure fixture cleanup

    def test_noop_and_invalid_configuration_do_not_discard_confirmation(self) -> None:
        self.ready_mapping()
        self.manager.confirm(self.manager.revision, True)
        revision = self.manager.revision
        self.assertEqual(self.post("configure", config=self.config).status_code, 200)
        self.assertEqual(self.manager.revision, revision)
        self.assertEqual(self.post("configure", config={**self.config, "robot": "unknown"}).status_code, 409)
        self.assertTrue(self.manager.status()["confirmed"])

    def test_http_rejects_cross_origin_requests_and_malformed_actions(self) -> None:
        self.assertEqual(self.client.post("/api/devices/scan", json={}).status_code, 403)
        response = self.client.post("/api/devices/scan", json={"config": self.config},
                                    headers={"X-Platform-Request": "1", "Origin": "https://external.invalid"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.post("scan", config={**self.config, "robot": "other"}).status_code, 409)
        self.assertEqual(self.post("move").status_code, 404)
        self.assertEqual(self.post("scan", config=self.config).status_code, 200)
        self.wait_for(lambda: not self.manager.scanning)
        self.assertEqual(self.post("open", id="can0", config={}).status_code, 409)
        self.assertEqual(self.post("open", id=["can0"], config=self.config).status_code, 409)
        self.assertEqual(self.post("open", id="missing", config=self.config).status_code, 409)


class ArmFeedbackTests(unittest.TestCase):
    def test_freshness_rejects_static_reset_and_nonfinite_timestamps(self) -> None:
        feedback = FreshFeedback()
        sample = SimpleNamespace(timestamp=1, msg=[0])
        with self.assertRaises(ValueError):
            feedback.read("joint", sample, 0)
        sample.timestamp = 2
        self.assertEqual(feedback.read("joint", sample, 0.1), [0])
        with self.assertRaises(ValueError):
            feedback.read("joint", sample, 0.7)
        sample.timestamp = 1
        with self.assertRaises(ValueError):
            feedback.read("joint", sample, 0.8)
        sample.timestamp = float("nan")
        with self.assertRaises(ValueError):
            feedback.read("joint", sample, 1)

    def test_both_arm_types_only_connect_read_and_disconnect(self) -> None:
        for robot, count in (("nero", 7), ("piper", 6)):
            calls = []
            stamp = iter(range(1, 30))
            arm = SimpleNamespace(
                OPTIONS=SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="gripper")),
                connect=lambda: calls.append("connect"), disconnect=lambda: calls.append("disconnect"),
                get_joint_angles=lambda: SimpleNamespace(timestamp=next(stamp), msg=[0] * count),
                get_leader_joint_angles=lambda: None,
                init_effector=lambda kind: SimpleNamespace(get_gripper_status=lambda: SimpleNamespace(
                    timestamp=next(stamp), msg=SimpleNamespace(value=0.01, foc_status=None))),
            )
            configs = []
            sdk = SimpleNamespace(
                AgxArmFactory=SimpleNamespace(create_arm=lambda config: arm),
                create_agx_arm_config=lambda **config: configs.append(config) or config,
                NeroFW=SimpleNamespace(V121="v121"), PiperFW=SimpleNamespace(DEFAULT="default"),
            )
            with patch("inference_platform.hardware.importlib.import_module", return_value=sdk):
                reader = ArmReader({"id": "can0"}, robot)
                reader.connect()
                self.assertFalse(reader.sample()[0]["ready"])
                self.assertTrue(reader.sample()[0]["ready"])
                self.assertEqual(len(reader.sample()[0]["joint_degrees"]), count)
                reader.close()
            self.assertEqual(configs[0]["robot"], robot)
            self.assertEqual(calls, ["connect", "disconnect"])


if __name__ == "__main__":
    unittest.main()
