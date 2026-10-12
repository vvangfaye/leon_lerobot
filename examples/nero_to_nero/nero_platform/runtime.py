"""应用实例持有的运行状态；导入模块不会连接任何设备。"""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import resolution

DEFAULT_CHANNELS = ["can0", "can1", "can2", "can3"]


def detect_can_interfaces() -> list[str]:
    """扫描系统里实际存在的 CAN 接口（按名字排序）；一个都没有时返回空列表。"""
    net = Path("/sys/class/net")
    if not net.exists():
        return []
    return sorted(p.name for p in net.iterdir() if p.name.startswith("can"))


def available_channels() -> list[str]:
    """实际可用通道；检测不到时回退到默认四通道（无 CAN 的测试环境）。"""
    return detect_can_interfaces() or DEFAULT_CHANNELS


def down_can_interfaces() -> list[str]:
    """当前 DOWN（或读不到状态）的 CAN 接口名，供界面提醒手动 UP；读 /sys 无需 root。"""
    down = []
    for name in detect_can_interfaces():
        operstate = Path("/sys/class/net") / name / "operstate"
        try:
            state = operstate.read_text().strip().lower()
        except OSError:
            state = ""
        if state != "up":
            down.append(name)
    return down


def default_channel_state() -> dict[str, Any]:
    """新建通道的初始状态（每次返回全新 dict）。"""
    return {
        "arm": None,
        "gripper": None,
        "status": "未连接",
        "label": "未分配",
        "detected": False,
        "feedback_mode": None,
        # 最近成功发出的主从角色请求；与收到哪类反馈分开保存。
        "requested_role": None,
    }


ARM_LABELS = ["未分配", "主左", "主右", "从左", "从右"]
JOINT_KEYS = [f"joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]
EXAMPLE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FIELDS = (
    "fps",
    "episode_sec",
    "repo_id",
    "task",
    "leader_label",
    "follower_label",
    "record_camera",
    "pair_mode",
    "camera_resolution",
    "follow_motion_mode",
    "control_frequency",
    "camera_focus_settings",
    "gripper_force",
)
DEFAULT_REPO_ID = "nero/platform_step1"
CONTROL_FREQUENCIES = (50, 100, 150, 200)


def target_control_hz(settings: dict[str, Any]) -> int:
    """解析配置的目标频率；实测频率由节拍统计给出。"""
    frequency = settings["control_frequency"]
    if frequency == "capture":
        fps = settings["fps"]
        if type(fps) is not int or fps <= 0:
            raise ValueError("FPS 必须为正整数")
        return fps
    if type(frequency) is not int or frequency not in CONTROL_FREQUENCIES:
        raise ValueError("控制频率请选择跟随采集、50、100、150 或 200 Hz")
    return frequency


def default_state() -> dict[str, Any]:
    return {
        "fps": 30,
        "follow_motion_mode": "js",
        "control_frequency": 100,
        "episode_sec": 60,
        "repo_id": DEFAULT_REPO_ID,
        "task": "nero teleop",
        "leader_label": "主左",
        "follower_label": "从左",
        "pair_mode": "single",
        "camera_resolution": resolution.defaults(),
        "camera_map": {},
        "camera_focus_settings": {},
        "record_camera": "both",
        "active_cameras": {},
        "camera_sources": {},
        "camera_config_notice": "",
        "connected": False,
        "recording": False,
        # recording 表示整个任务忙碌（含准备、从臂对齐和保存），record_phase 区分具体阶段。
        "record_phase": "idle",
        "record_message": "等待开始录制",
        "record_started_at": None,
        "record_elapsed": 0.0,
        "record_frames": 0,
        "record_frame_errors": 0,
        "record_frame_error": None,
        "record_timing": None,
        "control_timing": None,
        "follow_stop_message": "",
        "timing_report_error": None,
        "last_episode": None,
        "writer_error": None,
        "stop_requested": False,
        "shutting_down": False,
        "episode_count": 0,
        "last_error": None,
        "config_error": None,
        "recovery_required": None,
        "homing": False,
        "home_status": "",
        "home_error": None,
        "teleop_needs_restore": False,
        "disabling": False,
        "ending": False,
        "end_status": "",
        "disable_status": "",
        "disable_error": None,
        "close_error": None,
        "close_result": None,
        # 遥操作测试：只跟随主臂、不采样相机、不写数据集，用于前期训练。
        "teleop_test_active": False,
        "teleop_test_message": "",
        "teleop_test_stop": False,
        # 从臂夹爪夹持力（N），跟随循环每次下发实时读取，可采集过程中调整。
        "gripper_force": 10.0,
        # 遥操作测试专用夹持力（N），默认 3N，训练时更轻。
        "teleop_gripper_force": 3.0,
        # 返回默认位置逐个臂进行：回完一条臂后暂停，等待用户按 q 确认再回下一条。
        "home_awaiting": False,
        "home_pending": [],
    }


@dataclass
class Runtime:
    config_path: Path = field(default_factory=lambda: EXAMPLE_DIR / "collect_config.platform.json")
    dataset_root: Path = field(default_factory=lambda: Path.home() / ".cache/huggingface/lerobot")
    state: dict[str, Any] = field(default_factory=default_state)
    channels: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {ch: default_channel_state() for ch in available_channels()}
    )
    ch_locks: dict[str, Any] = field(
        default_factory=lambda: {ch: threading.Lock() for ch in available_channels()}
    )
    preview_cams: dict[str, Any] = field(default_factory=dict)
    preview_states: dict[str, dict[str, str]] = field(default_factory=dict)
    camera_focus: dict[str, dict] = field(default_factory=dict)
    preview_lock: Any = field(default_factory=threading.RLock)
    camera_inventory: dict[str, dict] = field(default_factory=dict)
    camera_detection: dict[str, dict] = field(default_factory=dict)
    cameras_scanned: bool = False
    camera_scan_lock: Any = field(default_factory=threading.Lock)
    collection_cams: dict[str, Any] = field(default_factory=dict)
    # 最新成功提交的一组 RGB 帧；整组替换，不在采样线程内编码预览。
    record_preview: dict[str, tuple[Any, Any, float]] = field(default_factory=dict)
    dataset: Any = None
    record_thread: threading.Thread | None = None
    home_thread: threading.Thread | None = None
    teleop_test_thread: threading.Thread | None = None
    home_continue_event: threading.Event = field(default_factory=threading.Event)
    device_busy: set[str] = field(default_factory=set)
    lifecycle_lock: Any = field(default_factory=threading.RLock)
    feedback_seen: dict[str, Any] = field(default_factory=dict)

    def operation_busy(self, *, include_recording: bool = True, channel: str | None = None) -> bool:
        """由调用者持有 lifecycle_lock；统一互斥判定，收尾可等待录制线程。"""
        return bool(
            self.state["shutting_down"]
            or any(self.state[key] for key in ("homing", "disabling", "ending"))
            or (include_recording and self.state["recording"])
            or self.state["teleop_test_active"]
            or (channel in self.device_busy if channel else self.device_busy)
        )
