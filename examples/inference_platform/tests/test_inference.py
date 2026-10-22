"""Real runtime orchestration with simulated I/O; never opens CAN or cameras."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from inference_platform.app import create_platform_app
from inference_platform.devices import DeviceManager
from inference_platform.motion import plan
from inference_platform.runtime import InferenceRuntime


class SimReader:
    def __init__(self, info, robot):
        self.info, self.robot = info, robot
        self.q = np.zeros(7 if robot == "nero" else 6)
        self.grip = 0.02
        self.rgb = np.full((32, 48, 3), 110, dtype=np.uint8) if info["kind"] == "camera" else None
        self.captured = time.monotonic()
        self.commands = []
        self.fail = False
        self.fail_send = False
        self.fail_hold = False
        self.owner = None

    def connect(self):
        self.owner = threading.get_ident()

    def close(self):
        assert self.owner == threading.get_ident()

    def sample(self):
        assert self.owner == threading.get_ident()
        if self.fail:
            raise ValueError("disconnected")
        self.captured = time.monotonic()
        return {
            "ready": True,
            "message": "simulated",
            "joint_radians": self.q.tolist(),
            "joint_degrees": np.degrees(self.q).tolist(),
            "gripper_m": self.grip,
        }, b"jpeg"

    def control(self, operation, value, check):
        assert self.owner == threading.get_ident(), "SDK access moved to another thread"
        check()
        self.commands.append(operation)
        if operation == "prepare":
            return {"point": [*self.q, self.grip], "limits": [[-3.2, 3.2]] * len(self.q)}
        if operation == "read":
            return {"point": [*self.q, self.grip], "stamp": time.monotonic()}
        if operation == "send":
            self.q, self.grip = np.asarray(value[:-1]), value[-1]
            if self.fail_send:
                raise ValueError("partial send failed")
        if operation == "home":
            self.q = np.asarray(value)
        if operation == "hold" and self.fail_hold:
            raise ValueError("hold failed")


class SimPolicy:
    def __init__(self, path):
        self.contract = json.loads((Path(path) / "config.json").read_text())
        self.predicts = 0
        self.closed = False
        self.delay = 0
        self.bad = False
        self.calls = []

    def load(self, check):
        check()
        return {"device": "simulation"}

    def call(self, request, check):
        self.calls.append(request["op"])
        if request["op"] == "reset":
            return {"ok": True}
        self.predicts += 1
        started = time.monotonic()
        while time.monotonic() - started < self.delay:
            check()
            time.sleep(0.01)
        check()
        actions = np.tile(request["state"], (3, 1))
        if self.bad:
            actions[:] = np.nan
        return {"actions": actions, "inference_ms": 10}

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = {"robot": "nero", "arms": "single", "space": "joint", "model": "pi05"}
        self.settings = {
            **self.config,
            "path": str(self.root / "checkpoint"),
            "task": "pick",
            "record": False,
            "duration": 1,
        }
        self.checkpoint()
        inventory = {
            i: {"id": i, "kind": kind} for i, kind in [("arm", "arm"), ("other", "arm"), ("cam", "camera")]
        }
        self.readers = {}

        def factory(info, robot):
            self.readers[info["id"]] = SimReader(info, robot)
            return self.readers[info["id"]]

        self.manager = DeviceManager(
            scanner=lambda: (inventory, {}),
            factory=factory,
            present=lambda _: True,
            lease_path=self.root / "lease",
        )
        self.runtime = InferenceRuntime(self.manager, SimPolicy)
        self.addCleanup(self.manager.close)
        self.addCleanup(self.runtime.close)
        self.manager.scan(self.config)
        self.wait(lambda: not self.manager.scanning)
        for identity in ("arm", "cam"):
            self.manager.open(identity, self.manager.revision)
            self.wait(lambda identity=identity: self.manager.workers[identity].snapshot()["phase"] == "ready")
        self.manager.assign("arm", "right_follower", self.manager.revision)
        self.manager.assign("cam", "head", self.manager.revision)
        self.manager.confirm(self.manager.revision, True)
        self.heartbeat_stop = threading.Event()

        def renew():
            while not self.heartbeat_stop.wait(0.05):
                if self.runtime.job:
                    self.runtime.heartbeat(self.runtime.job.id)

        self.renew = threading.Thread(target=renew, daemon=True)
        self.renew.start()
        self.addCleanup(self.heartbeat_stop.set)

    def checkpoint(self, names=None):
        path = Path(self.settings["path"])
        path.mkdir(exist_ok=True)
        names = names or [*[f"joint{i}.pos" for i in range(1, 8)], "gripper.pos"]
        config = {
            "type": "pi05",
            "action_feature_names": names,
            "n_action_steps": 3,
            "chunk_size": 3,
            "input_features": {
                "observation.state": {"shape": [len(names)]},
                "observation.images.head": {"type": "VISUAL", "shape": [3, 32, 48]},
            },
            "output_features": {"action": {"shape": [len(names)]}},
        }
        (path / "config.json").write_text(json.dumps(config))
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (path / name).touch()

    def wait(self, predicate, timeout=6):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end:
                self.fail(str(self.runtime.status()))
            time.sleep(0.01)

    def load(self):
        self.runtime.load(self.settings, self.runtime.revision)
        self.wait(lambda: not self.runtime.thread.is_alive())
        self.assertEqual(self.runtime.model_state, "ready")

    def run_once(self, kind="single"):
        self.runtime.start(kind, self.settings, self.runtime.revision)
        self.wait(lambda: not self.runtime.thread.is_alive(), 15)
        return self.runtime.status()

    def test_single_uses_owner_thread_and_keeps_model(self):
        self.load()
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "done", result)
        self.assertGreaterEqual(result["chunks"], 1)
        self.assertIn("send", self.readers["arm"].commands)
        self.assertEqual(self.readers["arm"].commands[-1], "hold")
        self.assertFalse(self.runtime.policy.closed)
        self.assertTrue(self.manager.status()["confirmed"])
        self.assertTrue(self.runtime.job.cancel.is_set())

    def test_load_without_confirmation_never_confirms_or_controls_devices(self):
        self.manager.configure({**self.config, "arms": "dual"}, self.manager.revision)
        self.manager.configure(self.config, self.manager.revision)
        self.assertFalse(self.manager.confirmed)
        self.load()
        self.assertFalse(self.manager.confirmed)
        self.assertEqual(self.readers["arm"].commands, [])
        with self.assertRaisesRegex(ValueError, "确认设备用途"):
            self.runtime.start("single", self.settings, self.runtime.revision)
        self.assertEqual(self.readers["arm"].commands, [])

    def test_load_with_no_connected_devices(self):
        self.manager.disconnect(self.manager.revision)
        self.wait(lambda: not self.manager.status()["active"])
        self.load()
        self.assertEqual(self.runtime.model_state, "ready")
        self.assertFalse(self.manager.confirmed)
        self.assertEqual(self.readers["arm"].commands, [])

    def test_single_master_follower_ignores_two_unassigned_arms(self):
        self.manager.devices.update({
            identity: {"id": identity, "kind": "arm"} for identity in ("unused_left", "unused_right")
        })
        self.manager.open("other", self.manager.revision)
        self.wait(lambda: self.manager.workers["other"].snapshot()["phase"] == "ready")
        self.manager.assign("other", "right_leader", self.manager.revision)
        self.load()
        self.assertFalse(self.manager.confirmed)
        self.manager.confirm(self.manager.revision, True)
        arms, _ = self.runtime.contract.bind(self.manager.status())
        self.assertEqual(arms, {"right": "arm"})
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "done", result)
        self.assertEqual(self.readers["other"].commands, [])
        self.assertNotIn("unused_left", self.readers)
        self.assertNotIn("unused_right", self.readers)

    def test_continuous_stop_and_configuration_locked(self):
        self.load()
        self.runtime.start("continuous", self.settings, self.runtime.revision)
        with self.assertRaisesRegex(ValueError, "执行"):
            self.manager.configure({**self.config, "space": "eef"}, self.manager.revision)
        self.wait(lambda: self.runtime.chunks >= 2)
        self.runtime.stop()
        self.wait(lambda: not self.runtime.thread.is_alive())
        count = self.runtime.chunks
        time.sleep(0.1)
        self.assertEqual(count, self.runtime.chunks)
        self.assertFalse(self.runtime.job.active)
        self.assertEqual(self.readers["arm"].commands[-1], "hold")

    def test_stop_during_prediction_never_sends_and_keeps_model(self):
        self.load()
        policy = self.runtime.policy
        policy.delay = 5
        self.runtime.start("single", self.settings, self.runtime.revision)
        self.wait(lambda: policy.predicts > 0)
        self.runtime.stop()
        self.wait(lambda: not self.runtime.thread.is_alive())
        self.assertNotIn("send", self.readers["arm"].commands)
        self.assertFalse(policy.closed)
        self.assertEqual(self.runtime.model_state, "ready")

    def test_no_browser_heartbeat_never_prepares(self):
        self.load()
        self.heartbeat_stop.set()
        self.renew.join()
        self.runtime.start("single", self.settings, self.runtime.revision)
        self.wait(lambda: not self.runtime.thread.is_alive())
        self.assertEqual(self.readers["arm"].commands, [])
        self.assertEqual(self.runtime.job.phase, "stopped")

    def test_invalid_prediction_sends_no_model_action(self):
        self.load()
        self.runtime.policy.bad = True
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "error")
        self.assertNotIn("send", self.readers["arm"].commands)

    def test_camera_loss_during_prediction_stops(self):
        self.load()
        policy = self.runtime.policy
        policy.delay = 5
        self.runtime.start("single", self.settings, self.runtime.revision)
        self.wait(lambda: policy.predicts > 0)
        self.readers["cam"].fail = True
        self.wait(lambda: not self.runtime.thread.is_alive())
        self.assertNotIn("send", self.readers["arm"].commands)
        self.assertIn("hold", self.readers["arm"].commands)

    def test_failure_one_arm_holds_both(self):
        self.manager.configure({**self.config, "arms": "dual"}, self.manager.revision)
        self.manager.open("other", self.manager.revision)
        self.wait(lambda: self.manager.workers["other"].snapshot()["phase"] == "ready")
        self.manager.assign("other", "left_follower", self.manager.revision)
        self.manager.confirm(self.manager.revision, True)
        self.settings["arms"] = "dual"
        names = [
            f"{side}.{name}"
            for side in ("left", "right")
            for name in [*[f"joint{i}.pos" for i in range(1, 8)], "gripper.pos"]
        ]
        self.checkpoint(names)
        self.load()
        self.readers["arm"].fail_send = True
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "error")
        for side in ("arm", "other"):
            self.assertEqual(self.readers[side].commands[-1], "hold")

    def test_hold_failure_blocks_new_motion(self):
        self.load()
        self.readers["arm"].fail_hold = True
        result = self.run_once()
        self.assertTrue(result["job"]["recovery"])
        with self.assertRaisesRegex(ValueError, "停止未确认"):
            self.runtime.start("single", self.settings, self.runtime.revision)

    def test_home_without_loading_moves_only_follower(self):
        self.manager.open("other", self.manager.revision)
        self.wait(lambda: self.manager.workers["other"].snapshot()["phase"] == "ready")
        self.manager.assign("other", "right_leader", self.manager.revision)
        self.manager.confirm(self.manager.revision, True)
        result = self.run_once("home")
        self.assertEqual(result["job"]["phase"], "done", result)
        np.testing.assert_allclose(np.degrees(self.readers["arm"].q), [0, -100, 0, 120, 0, 0, 90])
        self.assertEqual(self.readers["other"].commands, [])
        self.assertNotIn("send", self.readers["arm"].commands)

    def test_changed_configuration_and_stale_start_rejected(self):
        self.load()
        revision = self.runtime.revision
        self.runtime.stop()
        with self.assertRaisesRegex(ValueError, "状态已变化"):
            self.runtime.start("single", self.settings, revision)
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.runtime.start("single", {**self.settings, "space": "eef"}, self.runtime.revision)

    def test_real_recording_creates_video_and_parquet(self):
        self.load()
        self.settings.update(record=True, datasetName="nero/test", outputDir=str(self.root / "recordings"))
        result = self.run_once()
        self.assertEqual(result["job"]["phase"], "done", result)
        root = Path(result["record_path"])
        meta = json.loads((root / "meta/info.json").read_text())
        self.assertEqual(meta["total_episodes"], 1)
        self.assertGreater(meta["total_frames"], 0)
        self.assertTrue(list(root.glob("videos/**/*.mp4")))
        self.assertTrue(list(root.glob("data/**/*.parquet")))
        manifest = json.loads((root / "inference.json").read_text())
        self.assertEqual(manifest["status"], "done")
        self.assertGreaterEqual(len(manifest["chunks"]), 1)
        import pyarrow.parquet as pq

        rows = pq.read_table(next(root.glob("data/**/*.parquet"))).to_pydict()
        self.assertIn("action.commanded", rows)
        self.assertGreater(sum(rows["action.commanded"]), 0)

    def test_api_rejects_cross_origin_and_exposes_runtime(self):
        client = create_platform_app(devices=self.manager, runtime=self.runtime, candidates=[]).test_client()
        self.assertEqual(client.post("/api/inference/home", json={}).status_code, 403)
        self.assertEqual(client.get("/api/inference").json["model_state"], "empty")
        self.assertEqual(
            client.post("/api/inference/stop", json={}, headers={"X-Platform-Request": "1"}).status_code, 200
        )


class ContractTests(unittest.TestCase):
    def test_joint_trajectory_limits_finite_and_step(self):
        from types import SimpleNamespace

        contract = SimpleNamespace(count=7, config={"space": "joint"}, actions=lambda a: a)
        initial = np.zeros((1, 8))
        limits = np.array([[[-1, 1]] * 7])
        actions = np.zeros((3, 8))
        actions[:, 0] = [0.01, 0.02, 0.03]
        points, times, _ = plan(actions, contract, initial, limits, lambda: None)
        self.assertEqual(points.shape, (4, 1, 8))
        self.assertTrue(np.all(times >= 1 / 30))
        actions[0, 0] = 0.3
        with self.assertRaisesRegex(ValueError, "0.20"):
            plan(actions, contract, initial, limits, lambda: None)

    def test_gripper_target_clamped_not_rejected(self):
        from types import SimpleNamespace

        contract = SimpleNamespace(count=7, config={"space": "joint"}, actions=lambda a: a)
        initial = np.zeros((1, 8))
        limits = np.array([[[-3.2, 3.2]] * 7])
        actions = np.zeros((1, 8))
        actions[:, -1] = 0.15
        points, times, warnings = plan(actions, contract, initial, limits, lambda: None)
        self.assertEqual(points[-1][0, -1], 0.1)
        self.assertTrue(any("夹爪" in w for w in warnings))

    def test_eef_fk_ik_roundtrip_both_robots(self):
        from inference_platform.kinematics import eef, solve

        for robot, q in [("nero", np.zeros(7)), ("piper", np.array([0, 0.3, -0.3, 0, 0, 0]))]:
            limits = np.array([[-2, 2]] * len(q))
            target = np.array(eef(robot, q, 0.03))
            solved = solve(robot, target, q, limits)
            np.testing.assert_allclose(solved, q, atol=0.002)
            self.assertAlmostEqual(target[-1], -0.4)
            target[3:9] = 0
            with self.assertRaisesRegex(ValueError, "退化"):
                solve(robot, target, q, limits)


class AdapterTests(unittest.TestCase):
    def adapter(self, robot="nero", home=False):
        from types import SimpleNamespace

        from inference_platform.motion import ArmMotion
        from test_jog import SimulatedArm

        arm = SimulatedArm(7 if robot == "nero" else 6)
        arm.move_js = arm.move_j

        class Grip:
            width = 0.02
            moves = []

            def get_gripper_status(self):
                return arm.reply(
                    SimpleNamespace(value=self.width, foc_status=SimpleNamespace(driver_error_status=False))
                )

            def move_gripper_m(self, value, force):
                self.width = value
                self.moves.append(value)

        grip = Grip()
        self.arm, self.grip = arm, grip
        return ArmMotion(
            SimpleNamespace(arm=arm, robot=robot, limits=[(-1, 1)] * len(arm.pose), gripper=grip), home
        )

    def test_both_adapters_prepare_send_and_hold(self):
        for robot in ("nero", "piper"):
            adapter = self.adapter(robot)
            result = adapter.prepare(lambda: None)
            point = result["point"]
            point[0] = 0.005
            adapter.send(point, lambda: None)
            adapter.hold()
            self.assertAlmostEqual(self.arm.pose[0], 0.005)
            self.assertNotIn("estop", self.arm.calls)

    def test_stale_feedback_stop_uses_estop(self):
        adapter = self.adapter()
        adapter.prepare(lambda: None)
        self.arm.freeze = True
        # Different methods share the synthetic clock; pin the expected last
        # joint stamp so freeze is immediately observable in this fixture.
        adapter.fresh.seen["joint"] = (self.arm.stamp, time.monotonic(), True)
        with self.assertRaises(ValueError):
            adapter.hold()
        self.assertIn("estop", self.arm.calls)

    def test_inference_rejects_folded_out_of_limit_start(self):
        adapter = self.adapter()
        # J7 已放宽到 98.1°（约 1.712 rad）；1.75 rad 仍在其外，应被拒绝。
        self.arm.pose[-1] = 1.75
        with self.assertRaisesRegex(ValueError, "超出"):
            adapter.prepare(lambda: None)
        self.assertEqual(self.arm.moves, [])

    def test_home_does_not_command_gripper(self):
        adapter = self.adapter(home=True)
        adapter.prepare(lambda: None)
        target = [0.1] * 7
        adapter.home(target, lambda: None)
        adapter.hold()
        self.assertEqual(self.grip.moves, [])
