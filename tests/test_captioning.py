from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from src.ai.captioning import (
    available_asr_models,
    captions_to_srt,
    captions_to_webvtt,
    load_cached_transcript,
    resolve_asr_model_id,
    safe_whisper_max_new_tokens,
    write_caption_outputs,
)
from src.schemas import CaptionSegment, repair_caption_timestamps, validate_caption_segments
from src.utils.realtime_caption import build_realtime_caption_html


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_TRANSCRIPT = REPO_ROOT / "data" / "demo" / "cached_transcript.json"
CONFIG = REPO_ROOT / "config" / "demo.yaml"


class CaptioningTests(unittest.TestCase):
    def test_cached_transcript_loads(self) -> None:
        result = load_cached_transcript(CACHED_TRANSCRIPT)
        self.assertEqual(result.language, "th")
        self.assertGreater(len(result.segments), 0)

    def test_timestamps_are_valid(self) -> None:
        result = load_cached_transcript(CACHED_TRANSCRIPT)
        validate_caption_segments(result.segments)
        for segment in result.segments:
            self.assertGreaterEqual(segment.start, 0)
            self.assertGreaterEqual(segment.end, segment.start)

    def test_webvtt_and_srt_export_all_segments(self) -> None:
        result = load_cached_transcript(CACHED_TRANSCRIPT)
        webvtt = captions_to_webvtt(result)
        srt = captions_to_srt(result)
        self.assertTrue(webvtt.startswith("WEBVTT"))
        self.assertEqual(webvtt.count("-->"), len(result.segments))
        self.assertEqual(srt.count("-->"), len(result.segments))

    def test_write_caption_outputs(self) -> None:
        result = load_cached_transcript(CACHED_TRANSCRIPT)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_caption_outputs(result, Path(temp_dir))
            for path in paths.values():
                self.assertTrue(path.exists(), path)
            metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["segment_count"], len(result.segments))

    def test_cli_smoke_uses_cached_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.pipeline.run_captioning_demo",
                    "--config",
                    str(CONFIG),
                    "--mode",
                    "cached",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["segment_count"], 22)
            self.assertEqual(summary["language"], "th")
            self.assertEqual(summary["asr_provider"], "openai_whisper")
            self.assertEqual(summary["asr_model"], "large")
            self.assertEqual(summary["asr_max_new_tokens"], 440)
            self.assertTrue(summary["audio_path"].endswith("data\\demo\\audio\\1.mp3") or summary["audio_path"].endswith("data/demo/audio/1.mp3"))
            self.assertTrue((Path(temp_dir) / "captions.json").exists())
            self.assertTrue((Path(temp_dir) / "captions.vtt").exists())

    def test_cli_accepts_asr_max_new_tokens_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.pipeline.run_captioning_demo",
                    "--config",
                    str(CONFIG),
                    "--mode",
                    "cached",
                    "--asr-max-new-tokens",
                    "256",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["asr_max_new_tokens"], 256)

    def test_cli_lists_asr_model_aliases(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.pipeline.run_captioning_demo",
                "--list-asr-models",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        models = json.loads(completed.stdout)
        self.assertEqual(
            models["typhoon_whisper"]["turbo"],
            "typhoon-ai/typhoon-whisper-turbo",
        )
        self.assertEqual(
            models["typhoon_whisper"]["large-v3"],
            "typhoon-ai/typhoon-whisper-large-v3",
        )

    def test_asr_model_alias_resolution(self) -> None:
        self.assertEqual(resolve_asr_model_id("openai_whisper", "large"), "large")
        self.assertEqual(
            resolve_asr_model_id("typhoon_whisper", "turbo"),
            "typhoon-ai/typhoon-whisper-turbo",
        )
        self.assertEqual(
            resolve_asr_model_id("typhoon_whisper", "medium"),
            "typhoon-ai/monsoon-whisper-medium-gigaspeech2",
        )
        self.assertEqual(
            resolve_asr_model_id("typhoon_whisper", "isan-medium"),
            "typhoon-ai/typhoon-isan-asr-whisper",
        )
        self.assertEqual(
            resolve_asr_model_id("typhoon_whisper", "some-org/new-thai-asr"),
            "some-org/new-thai-asr",
        )
        self.assertIn("typhoon_whisper", available_asr_models())

    def test_safe_whisper_max_new_tokens_leaves_decoder_prompt_room(self) -> None:
        self.assertEqual(safe_whisper_max_new_tokens(448, 448), 440)
        self.assertEqual(safe_whisper_max_new_tokens(999, 448), 440)
        self.assertEqual(safe_whisper_max_new_tokens(128, 448), 128)
        self.assertEqual(safe_whisper_max_new_tokens(256, None), 256)

    def test_realtime_caption_html_embeds_audio_and_segments(self) -> None:
        result = load_cached_transcript(CACHED_TRANSCRIPT)
        audio_path = REPO_ROOT / "data" / "demo" / "audio" / "1.mp3"
        html = build_realtime_caption_html(audio_path, result)
        self.assertIn("<audio controls", html)
        self.assertIn("data:audio/mpeg;base64,", html)
        self.assertIn("Generating caption", html)
        self.assertIn(result.segments[0].text, html)

    def test_repair_caption_timestamps_clamps_whisper_overlap(self) -> None:
        segments = [
            CaptionSegment(0.0, 2.0, "first", "openai_whisper"),
            CaptionSegment(1.9, 3.0, "second", "openai_whisper"),
            CaptionSegment(2.8, 2.8, "third", "openai_whisper"),
        ]
        repaired = repair_caption_timestamps(segments)
        validate_caption_segments(repaired)
        self.assertEqual(repaired[1].start, 2.0)
        self.assertGreater(repaired[2].end, repaired[2].start)


if __name__ == "__main__":
    unittest.main()
