"""Exercise motion boundaries with synthetic feedback; never connect hardware."""

import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from inference_platform.app import create_platform_app
from inference_platform.devices import DeviceManager
from inference_platform.hardware import FreshFeedback
from inference_platform.jog import ArmJog, JogJob, JogStopped


class SimulatedArm:
    def __init__(self, count: int) -> None:
        self.pose = [0.0] * count
        self.enabled = False
        self.ctrl = NS(ctrl_mode=1, arm_status=0, err_code=0)
        self.moves: list[list[float]] = []
        self.calls: list = []
        self.stamp = 0
        self.limits = (-1.0, 1.0)
        self.fault = False
        self.fail_hold = False
        self.fail_stop = False
        self.stall_enable = False
        self.freeze = False
        self.controller_limits: list | None = None
        self.clipping = False

    def reply(self, msg: object) -> NS:
        if not self.freeze:
            self.stamp += 1
        return NS(timestamp=self.stamp, msg=msg)

    def get_joint_angles(self) -> NS:
        return self.reply(list(self.pose))

    def get_arm_status(self) -> NS:
        return self.reply(self.ctrl)

    def get_driver_states(self, index: int) -> NS:
        return self.reply(NS(foc_status=NS(driver_enable_status=self.enabled, driver_error_status=self.fault)))

    def get_joint_angle_vel_limits(self, index: int, **kwargs: object) -> NS:
        lo, hi = self.limits if self.controller_limits is None else self.controller_limits[index - 1]
        return self.reply(NS(min_angle_limit=lo, max_angle_limit=hi))

    def get_joint_limits_enabled(self) -> bool:
        return self.clipping

    def set_follower_mode(self) -> None:
        self.calls.append("follower")

    def enable(self, **kwargs: object) -> None:
        self.calls.append(("enable", kwargs))
        if not self.stall_enable:
            self.enabled = True

    def set_speed_percent(self, value: int) -> None:
        self.calls.append(("speed", value))

    def move_j(self, pose: list[float]) -> None:
        if self.fail_hold and self.moves:
            raise RuntimeError("hold send failed")
        self.moves.append(list(pose))
        self.pose = list(pose)

    def electronic_emergency_stop(self) -> None:
        self.calls.append("estop")
        if self.fail_stop:
            raise RuntimeError("CAN disconnected")


class MotionTests(unittest.TestCase):
    def engine(self, robot: str = "nero") -> ArmJog:
        self.arm = SimulatedArm(7 if robot == "nero" else 6)
        return ArmJog(self.arm, robot, [(-1, 1)] * len(self.arm.pose), FreshFeedback())

    def test_round_trip_only_changes_last_joint_and_uses_slow_planned_motion(self) -> None:
        for robot in ("nero", "piper"):
            with self.subTest(robot=robot):
                engine = self.engine(robot)
                origin = list(self.arm.pose)
                result = engine.run(lambda: None, lambda text: None)
                self.assertEqual(result[0], "done", result)
                self.assertFalse(result[2])
                self.assertEqual(len(self.arm.moves), 2)
                self.assertEqual(self.arm.moves[0][:-1], origin[:-1])
                self.assertAlmostEqual(abs(self.arm.moves[0][-1]), math.radians(5))
                self.assertEqual(self.arm.moves[1], origin)
                self.assertIn(("speed", 5), self.arm.calls)
                kwargs = next(call[1] for call in self.arm.calls if isinstance(call, tuple) and call[0] == "enable")
                self.assertEqual(kwargs, {"timeout": 0.05} if robot == "nero" else {})

    def test_positive_and_negative_boundary_move_towards_center(self) -> None:
        for position in (-0.99, 0.99):
            engine = self.engine()
            self.arm.pose[-1] = position
            self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "done")
            self.assertLess(abs(self.arm.moves[0][-1]), abs(position))

    def test_outside_sdk_or_controller_limits_never_enables_or_moves(self) -> None:
        for controller_limits, position in (((-1, 1), 1.01), ((-0.5, 0.5), 0.6)):
            engine = self.engine()
            self.arm.limits = controller_limits
            self.arm.pose[1] = position
            result = engine.run(lambda: None, lambda _: None)
            self.assertIn("J2", result[1])
            self.assertIn("允许范围", result[1])
            self.assertEqual(self.arm.calls, [])
            self.assertEqual(self.arm.moves, [])
            self.assertFalse(result[2])

    def test_missing_or_invalid_limits_reject_before_enabling(self) -> None:
        for limits in ((float("nan"), 1), (0, 0), (2, 3)):
            engine = self.engine()
            self.arm.limits = limits
            self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "error")
            self.assertEqual(self.arm.calls, [])

    def test_fault_or_external_control_never_enables(self) -> None:
        for field, value in (("arm_status", 1), ("err_code", 2), ("ctrl_mode", 8)):
            engine = self.engine()
            setattr(self.arm.ctrl, field, value)
            self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "error")
            self.assertEqual(self.arm.calls, [])
        engine = self.engine()
        self.arm.fault = True
        self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "error")
        self.assertEqual(self.arm.calls, [])

    def test_cancel_before_prepare_does_not_write(self) -> None:
        engine = self.engine()
        def cancelled() -> None:
            raise JogStopped("stop")
        self.assertEqual(engine.run(cancelled, lambda _: None)[0], "stopped")
        self.assertEqual(self.arm.calls, [])

    def test_cancel_after_outbound_holds_current_pose_without_returning(self) -> None:
        engine = self.engine()
        def stop_after_outbound() -> None:
            if self.arm.moves:
                raise JogStopped("stop")
        result = engine.run(stop_after_outbound, lambda _: None)
        self.assertEqual(result[0], "stopped", result)
        self.assertEqual(self.arm.moves[1], self.arm.moves[0])
        self.assertNotEqual(self.arm.moves[-1], [0] * 7)
        self.assertNotIn("estop", self.arm.calls)

    def test_hold_failure_sends_estop_and_requires_recovery(self) -> None:
        for fail_stop in (False, True):
            engine = self.engine()
            self.arm.fail_hold, self.arm.fail_stop = True, fail_stop
            def stopped() -> None:
                if self.arm.moves:
                    raise JogStopped("stop")
            result = engine.run(stopped, lambda _: None)
            self.assertEqual(result[0], "error")
            self.assertTrue(result[2])
            self.assertIn("estop", self.arm.calls)
            if fail_stop:
                self.assertIn("实体急停", result[1])

    def test_feedback_loss_during_movement_stops_and_latches(self) -> None:
        engine = self.engine()
        def lost() -> None:
            if self.arm.moves:
                raise ValueError("CAN disconnected")
        result = engine.run(lost, lambda _: None)
        self.assertEqual(result[0], "error")
        self.assertTrue(result[2])
        self.assertIn("estop", self.arm.calls)
        self.assertEqual(len(self.arm.moves), 1)

    def test_frozen_joint_timestamps_do_not_count_as_arrival(self) -> None:
        engine = self.engine()
        def freeze_after_move() -> None:
            if self.arm.moves:
                self.arm.freeze = True
        result = engine.run(freeze_after_move, lambda _: None)
        self.assertEqual(result[0], "error", result)
        self.assertTrue(result[2])
        self.assertEqual(len(self.arm.moves), 1)
        self.assertIn("estop", self.arm.calls)

    def test_enable_timeout_does_not_send_joint_targets(self) -> None:
        engine = self.engine()
        self.arm.stall_enable = True
        result = engine.run(lambda: None, lambda _: None)
        self.assertIn("使能超时", result[1])
        self.assertEqual(self.arm.moves, [])
        self.assertIn("estop", self.arm.calls)
        self.assertTrue(result[2])

    def test_other_axis_drift_aborts_without_returning(self) -> None:
        engine = self.engine()
        def drift() -> None:
            if self.arm.moves:
                self.arm.pose[0] = math.radians(2)
        result = engine.run(drift, lambda _: None)
        self.assertIn("其他关节偏移", result[1])
        self.assertEqual(len(self.arm.moves), 1)
        self.assertIn("estop", self.arm.calls)

    def test_heartbeat_expires_and_late_heartbeat_cannot_clear_cancel(self) -> None:
        job = JogJob("can0")
        job.created -= 2
        with self.assertRaises(JogStopped):
            job.wait_for_browser()
        job.heartbeat()
        with self.assertRaises(JogStopped):
            job.check()
        job = JogJob("can0")
        job.heartbeat()
        job.check()
        job.heartbeat_at -= 2
        with self.assertRaises(JogStopped):
            job.check()
        job.cancel.set()
        old = job.heartbeat_at
        job.heartbeat()
        self.assertEqual(job.heartbeat_at, old)
        with self.assertRaises(JogStopped):
            job.check()

    def folded_engine(self, degrees: list[float]) -> ArmJog:
        self.arm = SimulatedArm(7)
        # Actual limits returned by the installed NERO controller.
        limits = [tuple(math.radians(value) for value in pair) for pair in (
            (-155, 155), (-100, 100), (-158, 158), (-58, 123), (-158, 158), (-42, 55), (-90, 90),
        )]
        self.arm.controller_limits = limits
        self.arm.pose = [math.radians(value) for value in degrees]
        return ArmJog(self.arm, "nero", limits, FreshFeedback())

    def test_reported_nero_folded_poses_preserve_non_wrist_targets_and_return(self) -> None:
        # 限位放宽到实测范围后，J2 < -101° 的收拢位姿仍走收拢姿态专用窗口；J2 ≥ -101° 且 J7 ≤ 98.1° 走常规微动。
        cases = {
            "folded": (
                [0, -101.43, 0, 125.8, 0, 0, 96.8],
            ),
            "normal": (
                [0.1, -101, 0, 125.9, 0, 0, 97.2],
                [1.7, -100.5, 0, 125.8, -0.1, 0, 94.9],
                [0, -100.4, 0, 126, 0, 0, 94.9],
            ),
        }
        for kind, poses in cases.items():
            for degrees in poses:
                with self.subTest(kind=kind, degrees=degrees):
                    engine = self.folded_engine(degrees)
                    origin = list(self.arm.pose)
                    result = engine.run(lambda: None, lambda _: None)
                    self.assertEqual(result[0], "done", result)
                    self.assertEqual(self.arm.moves[0][:-1], origin[:-1])
                    self.assertAlmostEqual(self.arm.moves[0][-1], origin[-1] - math.radians(5))
                    self.assertEqual(self.arm.moves[1], origin)
                    if kind == "folded":
                        self.assertIn("收拢姿态", result[1])
                    else:
                        self.assertNotIn("收拢姿态", result[1])

    def test_folded_pose_does_not_expand_general_workspace_or_disable_sdk_clipping(self) -> None:
        baseline = [0, -101.43, 0, 125.8, 0, 0, 96.8]
        for index, value in ((0, 20), (1, -103), (3, 128), (6, 99)):
            degrees = list(baseline)
            degrees[index] = value
            engine = self.folded_engine(degrees)
            self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "error")
            self.assertEqual(self.arm.calls, [])
        engine = self.folded_engine(baseline)
        self.arm.clipping = True
        result = engine.run(lambda: None, lambda _: None)
        self.assertIn("SDK 已开启", result[1])
        self.assertEqual(self.arm.calls, [])
        self.assertTrue(self.arm.clipping)
        # J2/J4/J7 的控制器限位已被 NERO_LIMIT_WIDEN_DEG 覆盖放宽，收窄它们不再触发拒绝；
        # 用一个未放宽的关节（J5）验证「收窄控制器限位 → 拒绝」仍然成立。
        engine = self.folded_engine(baseline)
        self.arm.controller_limits[4] = (math.radians(0.2), math.radians(1.0))
        self.assertEqual(engine.run(lambda: None, lambda _: None)[0], "error")
        self.assertEqual(self.arm.calls, [])

    def test_folded_pose_cancel_holds_without_return_or_other_joint_correction(self) -> None:
        engine = self.folded_engine([0, -101.43, 0, 125.8, 0, 0, 96.8])
        def stop() -> None:
            if self.arm.moves:
                raise JogStopped("cancel")
        result = engine.run(stop, lambda _: None)
        self.assertEqual(result[0], "stopped", result)
        self.assertEqual(self.arm.moves[1], self.arm.moves[0])
        self.assertNotIn("estop", self.arm.calls)

    def test_folded_pose_cannot_move_wrist_further_out(self) -> None:
        engine = self.folded_engine([0, -101.43, 0, 125.8, 0, 0, 96.8])
        def drift_outward() -> None:
            if self.arm.moves:
                self.arm.pose[-1] = math.radians(97.8)
        result = engine.run(drift_outward, lambda _: None)
        self.assertTrue(result[2])
        self.assertIn("偏离收拢姿态微动范围", result[1])
        self.assertEqual(len(self.arm.moves), 1)
        self.assertIn("estop", self.arm.calls)


class JogReader:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.recovery = False
        self.owner: int | None = None

    def connect(self) -> None:
        self.owner = threading.get_ident()

    def sample(self) -> tuple[dict, None]:
        assert self.owner == threading.get_ident()
        return {"ready": True, "message": "synthetic"}, None

    def jog(self, check: object, progress: object) -> tuple[str, str, bool]:
        assert self.owner == threading.get_ident()
        self.started.set()
        self.events.append("start")
        try:
            while not self.release.is_set():
                check()
                time.sleep(0.01)
            return "error" if self.recovery else "done", "synthetic", self.recovery
        except JogStopped:
            self.events.append("hold")
            return "stopped", "synthetic stopped", False

    def close(self) -> None:
        assert self.owner == threading.get_ident()
        self.events.append("close")


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.readers: dict[str, JogReader] = {}
        def factory(info: dict, robot: str) -> JogReader:
            reader = JogReader()
            self.readers[info["id"]] = reader
            return reader
        self.manager = DeviceManager(
            scanner=lambda: ({key: {"id": key, "kind": "arm"} for key in ("can0", "can1")}, {}),
            factory=factory, present=lambda _: True, lease_path=Path(temp.name) / "lease",
        )
        self.addCleanup(self.manager.close)
        self.client = create_platform_app(candidates=[], devices=self.manager).test_client()
        self.manager.scan({"robot": "nero", "arms": "dual", "space": "eef"})
        self.wait(lambda: not self.manager.scanning)
        for identity in ("can0", "can1"):
            self.manager.open(identity, self.manager.revision)
            self.wait(lambda: self.manager.workers[identity].snapshot()["phase"] == "ready")

    def wait(self, condition: object) -> None:
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() > deadline:
                self.fail("lifecycle timed out")
            time.sleep(0.01)

    def start(self, heartbeat: bool = True) -> JogJob:
        self.manager.jog("can0", self.manager.revision, True)
        job = self.manager.jog_job
        if heartbeat:
            self.manager.heartbeat_jog(job.id)
            self.wait(self.readers["can0"].started.is_set)
        return job

    def test_exclusive_motion_locks_roles_and_confirmation(self) -> None:
        self.start()
        for action in (
            lambda: self.manager.jog("can1", self.manager.revision, True),
            lambda: self.manager.assign("can1", "right_follower", self.manager.revision),
            lambda: self.manager.confirm(self.manager.revision, True),
            lambda: self.manager.configure(
                {"robot": "piper", "arms": "single", "space": "joint"}, self.manager.revision,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "先停止"):
                action()
        self.assertFalse(self.readers["can1"].started.is_set())

    def test_stop_invalidates_delayed_start_and_does_not_require_current_revision(self) -> None:
        old = self.manager.revision
        response = self.client.post("/api/devices/jog-stop", json={}, headers={"X-Platform-Request": "1"})
        self.assertEqual(response.status_code, 200)
        with self.assertRaises(ValueError):
            self.manager.jog("can0", old, True)
        job = self.start()
        before = time.monotonic()
        self.manager.stop_jog()
        self.assertLess(time.monotonic() - before, 0.1)
        self.wait(lambda: not job.snapshot()["active"])
        self.assertEqual(job.snapshot()["phase"], "stopped")

    def test_abandoned_start_or_wrong_heartbeat_never_calls_motion_adapter(self) -> None:
        job = self.start(heartbeat=False)
        self.manager.heartbeat_jog("wrong")
        job.created -= 2
        self.wait(lambda: not job.snapshot()["active"])
        self.assertFalse(self.readers["can0"].started.is_set())
        self.assertEqual(job.snapshot()["phase"], "stopped")

    def test_expired_active_browser_lease_holds_without_a_stop_request(self) -> None:
        job = self.start()
        job.heartbeat_at -= 2
        self.manager.heartbeat_jog(job.id)
        self.wait(lambda: not job.snapshot()["active"])
        self.assertEqual(self.readers["can0"].events, ["start", "hold"])

    def test_disconnect_and_shutdown_stop_before_releasing_the_device(self) -> None:
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                job = self.start()
                if shutdown:
                    self.manager.close()
                else:
                    self.manager.disconnect(self.manager.revision, "can0")
                self.wait(lambda: self.manager.workers["can0"].closed)
                self.assertEqual(self.readers["can0"].events, ["start", "hold", "close"])
                self.assertFalse(job.snapshot()["active"])
                if not shutdown:
                    self.manager.open("can0", self.manager.revision)
                    self.wait(lambda: self.manager.workers["can0"].snapshot()["phase"] == "ready")

    def test_recovery_cannot_be_cleared_by_disconnect_or_rescan(self) -> None:
        job = self.start()
        self.readers["can0"].recovery = True
        self.readers["can0"].release.set()
        self.wait(lambda: not job.snapshot()["active"])
        self.manager.disconnect(self.manager.revision)
        self.wait(lambda: not self.manager.status()["active"])
        self.manager.scan({"robot": "piper", "arms": "single", "space": "joint"})
        self.wait(lambda: not self.manager.scanning)
        with self.assertRaisesRegex(ValueError, "检查实物"):
            self.manager.jog("can0", self.manager.revision, True)

    def test_jog_requires_explicit_confirmation_and_local_json_request(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.jog("can0", self.manager.revision, False)
        headers = {"X-Platform-Request": "1"}
        for identity in ("missing", [], None):
            response = self.client.post("/api/devices/jog",
                                        json={"id": identity, "revision": self.manager.revision,
                                              "acknowledged": True}, headers=headers)
            self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.post("/api/devices/jog-stop", json={}).status_code, 403)
