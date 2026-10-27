#!/usr/bin/env bash
# Keep MCP stdout exclusively for protocol messages.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec uv run --project "$repo_root" --offline --no-sync --with 'mcp==1.30.0' \
  python "$repo_root/examples/nero_to_nero/nero_mcp.py" "$@"
