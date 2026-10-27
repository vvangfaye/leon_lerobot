#!/usr/bin/env python
"""Expose the existing NERO stack to Codex; default to observation-only mode."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations
from nero_platform.agent_control import Controller, MockBackend, load_config
from nero_platform.agent_hardware import Cameras, LiveBackend, NeroKinematics


def create_server(config_path: Path, *, mock: bool = False, allow_motion: bool = False) -> FastMCP:
    """Construct a server without opening CAN devices or cameras."""
    config = load_config(config_path)
    backend = MockBackend() if mock else LiveBackend(config)
    controller = Controller(backend, NeroKinematics(), allow_motion=allow_motion, mock=mock)
    cameras = Cameras(config, mock=mock)

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[None]:
        try:
            yield
        finally:
            try:
                await asyncio.to_thread(controller.close)
            finally:
                await asyncio.to_thread(cameras.close)

    server = FastMCP(
        "NERO",
        lifespan=lifespan,
        instructions=(
            "Control only the configured NERO followers, one arm at a time. First call get_robot_info, "
            "look and get_robot_state. Coordinates are relative to each arm's base, not camera/world. "
            "Images are not calibrated 3D coordinates. Only call prepare_arm and motion tools when the "
            "human requests physical operation. Never change limits or launch another CAN controller. "
            "Observe after every small move; stop on missing feedback or unexpected contact. "
            "MOCK responses are simulations and never prove a real task succeeded."
        ),
    )
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    motion = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)

    @server.tool(annotations=read_only)
    def get_robot_info() -> dict:
        """Describe mode, configured hardware, limits and coordinate conventions; no connection."""
        return {
            "mock": mock,
            "allow_motion": allow_motion,
            "channels": config["channels"],
            "cameras": list(config["cameras"]),
            "joint_units": "degrees",
            "distance_units": "mm",
            "frame": "each arm's own base; no camera/world calibration or collision model",
            "max_joint_change_deg": 5,
            "max_joint_speed_deg_s": 5,
            "max_cartesian_step_mm": 10,
            "max_gripper_change_mm": 10,
            "gripper_force_n": 3,
            "requires": "Stop teleoperation/inference/recording controllers before live use",
        }

    @server.tool(annotations=read_only)
    async def get_robot_state(side: str) -> dict:
        """Read current left or right arm state; never enable or move the arm."""
        return await asyncio.to_thread(controller.state, side)

    @server.tool(annotations=read_only)
    async def look(camera: str = "head") -> Image:
        """Get a fresh configured head/left/right camera image (MOCK explicitly labelled)."""
        return Image(data=await asyncio.to_thread(cameras.jpeg, camera), format="jpeg")

    @server.tool(annotations=motion)
    async def prepare_arm(side: str) -> dict:
        """Enable the selected follower for small moves, without homing. Requires --allow-motion."""
        return await asyncio.to_thread(controller.prepare, side)

    @server.tool(annotations=motion)
    async def move_joints(side: str, target_degrees: list[float], duration_s: float = 2.0) -> dict:
        """Move seven absolute joint angles in degrees, max 5 degrees change; returns measured state."""
        return await asyncio.to_thread(controller.move_joints, side, target_degrees, duration_s)

    @server.tool(annotations=motion)
    async def move_ee_by(
        side: str, dx_mm: float, dy_mm: float, dz_mm: float, duration_s: float = 2.0
    ) -> dict:
        """Translate in the selected arm's base frame, max 10 mm total, preserving orientation."""
        return await asyncio.to_thread(controller.move_ee_by, side, [dx_mm, dy_mm, dz_mm], duration_s)

    @server.tool(annotations=motion)
    async def set_gripper(side: str, width_mm: float) -> dict:
        """Set physical jaw opening (0–70 mm), changing at most 10 mm per call, with 3 N force."""
        return await asyncio.to_thread(controller.set_gripper, side, width_mm)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    def stop_motion() -> dict:
        """Cancel active local motion without removing torque. This is not a hardware e-stop."""
        return controller.stop()

    return server


def main() -> None:
    """Start a local stdio MCP server; nothing connects until an observation tool is called."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("collect_config.platform.json")
    )
    parser.add_argument("--mock", action="store_true", help="Use simulated arm state and labelled images")
    parser.add_argument(
        "--allow-motion", action="store_true", help="Allow explicit preparation and bounded moves"
    )
    args = parser.parse_args()
    create_server(args.config, mock=args.mock, allow_motion=args.allow_motion).run(transport="stdio")


if __name__ == "__main__":
    main()
