#!/usr/bin/env python
"""Exercise the entire MCP stdio path using mock hardware only."""

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    """Verify initialization, tool discovery, images, motion and shutdown."""
    server = Path(__file__).with_name("nero_mcp.py")
    params = StdioServerParameters(command=sys.executable, args=[str(server), "--mock", "--allow-motion"])
    print("Starting mock MCP server", flush=True)
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session,
    ):
        await session.initialize()
        print("PASS initialize", flush=True)
        tool_list = await session.list_tools()
        assert len(tool_list.tools) == 8
        for name, arguments in (
            ("get_robot_info", {}),
            ("get_robot_state", {"side": "left"}),
            ("look", {"camera": "head"}),
            ("prepare_arm", {"side": "left"}),
            ("move_joints", {"side": "left", "target_degrees": [1, 0, 0, 0, 0, 0, 0]}),
            ("move_ee_by", {"side": "left", "dx_mm": 0, "dy_mm": 0, "dz_mm": 1}),
            ("set_gripper", {"side": "left", "width_mm": 35}),
            ("stop_motion", {}),
        ):
            result = await session.call_tool(name, arguments)
            assert not result.isError, (name, result)
            if name == "look":
                assert any(item.type == "image" for item in result.content)
            else:
                assert json.loads(result.content[0].text)["mock"] is True
            print(f"PASS {name}", flush=True)
        denied = await session.call_tool("move_joints", {"side": "left", "target_degrees": [0] * 7})
        assert denied.isError, "stop must require explicit re-preparation"
        print("PASS post-stop motion rejected; mock MCP smoke complete")


if __name__ == "__main__":
    asyncio.run(main())
