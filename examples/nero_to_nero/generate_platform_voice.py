#!/usr/bin/env python
"""预生成固定中文提示音；仅此工具需要 edge-tts 和网络，采集运行时不需要。"""

import argparse
import asyncio
import json
from pathlib import Path


async def generate(voices: list[str], force: bool, cues: list[str] | None = None) -> None:
    import edge_tts  # 独立生成工具的可选依赖，不导入采集进程。

    root = Path(__file__).parent / "nero_platform/static/voice"
    manifest = json.loads((root / "prompts.json").read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(3)

    async def save(voice: str, cue: str, text: str) -> None:
        path = root / voice / f"{cue}.mp3"
        if path.is_file() and path.stat().st_size > 0 and not force:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.mp3")
        async with semaphore:
            try:
                for attempt in range(3):
                    try:
                        await asyncio.wait_for(
                            edge_tts.Communicate(
                                text, manifest["voices"][voice]["provider_voice"], rate=manifest["rate"]
                            ).save(str(temporary)),
                            timeout=45,
                        )
                        if temporary.stat().st_size == 0:
                            raise RuntimeError("语音文件为空")
                        temporary.replace(path)
                        print(f"已生成 {voice}/{cue}", flush=True)
                        return
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(1 + attempt)
            finally:
                temporary.unlink(missing_ok=True)

    selected = voices or list(manifest["voices"])
    if any(voice not in manifest["voices"] for voice in selected):
        raise ValueError("未知音色")
    prompts = cues or list(manifest["prompts"])
    if any(cue not in manifest["prompts"] for cue in prompts):
        raise ValueError("未知提示音")
    await asyncio.gather(
        *(save(voice, cue, manifest["prompts"][cue]) for voice in selected for cue in prompts)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", action="append", default=[], help="xiaoxiao 或 yunxi，默认生成两种")
    parser.add_argument("--force", action="store_true", help="文案改变后重新生成所有选中音色")
    parser.add_argument("--cue", action="append", help="仅生成指定提示音，可多次指定")
    args = parser.parse_args()
    asyncio.run(generate(args.voice, args.force, args.cue))
