"""提示音资源与页面路由验证，不建立 CAN 或相机连接。"""

import json
import re
import unittest
from pathlib import Path

from nero_platform.app import create_app
from nero_platform.runtime import Runtime


class VoiceAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent / "nero_platform/static/voice"
        self.manifest = json.loads((self.root / "prompts.json").read_text(encoding="utf-8"))
        self.client = create_app(Runtime()).test_client()

    def test_all_cues_have_both_local_voices(self) -> None:
        self.assertEqual(set(self.manifest["voices"]), {"xiaoxiao", "yunxi"})
        source = (self.root.parent / "voice.js").read_text(encoding="utf-8")
        page = self.client.get("/").text
        for cue in re.findall(r'voice.say\("([a-z_]+)"', page):
            self.assertIn(cue, self.manifest["prompts"])
        self.assertNotIn("speechSynthesis", source)
        for voice in self.manifest["voices"]:
            for cue in self.manifest["prompts"]:
                with self.subTest(voice=voice, cue=cue):
                    path = self.root / voice / f"{cue}.mp3"
                    self.assertGreater(path.stat().st_size, 1000)
                    response = self.client.get(f"/static/voice/{voice}/{cue}.mp3")
                    try:
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.mimetype, "audio/mpeg")
                    finally:
                        response.close()

    def test_page_injects_manifest_and_serves_player(self) -> None:
        page = self.client.get("/").text
        self.assertIn("zh-CN-XiaoxiaoNeural", page)
        self.assertNotIn("/*VOICE_MANIFEST*/", page)
        self.assertIn('id="safety-dialog"', page)
        self.assertIn('id="safety-confirm"', page)
        response = self.client.get("/static/voice.js")
        try:
            self.assertEqual(response.status_code, 200)
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
