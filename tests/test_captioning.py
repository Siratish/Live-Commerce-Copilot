from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.ai.captioning import (
    AudioWindow,
    available_asr_models,
    captions_to_srt,
    captions_to_webvtt,
    iter_pause_audio_windows_from_samples,
    load_cached_transcript,
    resolve_asr_model_id,
    safe_whisper_max_new_tokens,
    transcribe_audio_windows,
    write_caption_outputs,
)
from src.schemas import CaptionSegment, repair_caption_timestamps, validate_caption_segments
from src.utils.realtime_audio_file import (
    RealtimeAudioFileDemoConfig,
    run_realtime_audio_file_demo,
    write_audio_window_wav,
)
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
            self.assertEqual(summary["segment_count"], 25)
            self.assertEqual(summary["language"], "th")
            self.assertEqual(summary["asr_provider"], "openai_whisper")
            self.assertEqual(summary["asr_model"], "large")
            self.assertEqual(summary["asr_chunk_length_seconds"], 4)
            self.assertTrue(summary["asr_dynamic_chunking"])
            self.assertEqual(summary["asr_pause_seconds"], 0.7)
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

    def test_cli_accepts_asr_chunk_length_override(self) -> None:
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
                    "--asr-chunk-length-seconds",
                    "6",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["asr_chunk_length_seconds"], 6)

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

    def test_typhoon_audio_windows_create_timed_segments_without_text_splitting(self) -> None:
        windows = [
            AudioWindow(0.0, 4.0, [0.0], 16_000),
            AudioWindow(4.0, 8.0, [0.0], 16_000),
        ]

        class FakePipe:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, audio, generate_kwargs):
                self.calls += 1
                return {"text": f"chunk {self.calls}"}

        fake_pipe = FakePipe()
        with patch("src.ai.captioning.iter_audio_windows", return_value=iter(windows)):
            segments = transcribe_audio_windows(
                pipe=fake_pipe,
                audio_path=Path("demo.mp3"),
                language="th",
                source="typhoon_whisper:test",
                chunk_length_seconds=4,
                max_new_tokens=440,
                dynamic_chunking=True,
            )

        self.assertEqual(
            [(segment.start, segment.end, segment.text) for segment in segments],
            [(0.0, 4.0, "chunk 1"), (4.0, 8.0, "chunk 2")],
        )

    def test_typhoon_chunk_timestamps_are_offset_by_audio_window(self) -> None:
        windows = [AudioWindow(8.0, 12.0, [0.0], 16_000)]

        def fake_pipe(audio, generate_kwargs):
            return {
                "chunks": [
                    {"timestamp": (0.5, 1.5), "text": "inside window"},
                ]
            }

        with patch("src.ai.captioning.iter_audio_windows", return_value=iter(windows)):
            segments = transcribe_audio_windows(
                pipe=fake_pipe,
                audio_path=Path("demo.mp3"),
                language="th",
                source="typhoon_whisper:test",
                chunk_length_seconds=4,
                max_new_tokens=440,
                dynamic_chunking=True,
            )

        self.assertEqual(segments[0].start, 8.5)
        self.assertEqual(segments[0].end, 9.5)
        self.assertEqual(segments[0].text, "inside window")

    def test_pause_audio_windows_split_on_speaker_pause(self) -> None:
        import numpy as np

        sample_rate = 10
        samples = np.array(
            [0.0] * 5
            + [0.4] * 10
            + [0.0] * 8
            + [0.4] * 10
            + [0.0] * 5,
            dtype=np.float32,
        )
        windows = list(
            iter_pause_audio_windows_from_samples(
                samples=samples,
                sample_rate=sample_rate,
                max_chunk_seconds=10,
                min_chunk_seconds=0.5,
                pause_seconds=0.5,
                silence_threshold=0.1,
                frame_seconds=0.1,
            )
        )

        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0].start, 0.5)
        self.assertAlmostEqual(windows[0].end, 1.5)
        self.assertAlmostEqual(windows[1].start, 2.3)
        self.assertAlmostEqual(windows[1].end, 3.3)

    def test_pause_audio_windows_respect_max_chunk_without_pause(self) -> None:
        import numpy as np

        sample_rate = 10
        samples = np.array([0.4] * 35, dtype=np.float32)
        windows = list(
            iter_pause_audio_windows_from_samples(
                samples=samples,
                sample_rate=sample_rate,
                max_chunk_seconds=1.0,
                min_chunk_seconds=0.2,
                pause_seconds=0.5,
                silence_threshold=0.1,
                frame_seconds=0.1,
            )
        )

        self.assertGreaterEqual(len(windows), 3)
        self.assertAlmostEqual(windows[0].start, 0.0)
        self.assertAlmostEqual(windows[0].end, 1.0)

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

    def test_realtime_audio_file_gate_releases_chunks_before_asr(self) -> None:
        import numpy as np

        class FakePlaybackGate:
            def __init__(self) -> None:
                self.current_time = 0.0
                self.waits = []
                self.published = []

            def install(self) -> None:
                pass

            def wait_for_start(self):
                return {"currentTime": self.current_time}

            def wait_until(self, target_seconds):
                self.waits.append(float(target_seconds))
                self.current_time = float(target_seconds)
                return {"currentTime": self.current_time}

            def publish(self, payload):
                self.published.append(dict(payload))

        class FakeAsr:
            def __init__(self, gate: FakePlaybackGate) -> None:
                self.gate = gate
                self.calls = []

            def transcribe_chunk(self, chunk_path, offset_seconds, fallback_duration_seconds):
                self.calls.append((offset_seconds, fallback_duration_seconds, Path(chunk_path).exists()))
                self.assert_playback_released(offset_seconds, fallback_duration_seconds)
                return [
                    CaptionSegment(
                        start=offset_seconds,
                        end=offset_seconds + fallback_duration_seconds,
                        text=f"chunk {len(self.calls)}",
                        source="fake_stream_asr",
                    )
                ]

            def assert_playback_released(self, offset_seconds, duration_seconds) -> None:
                self_gate_time = self.gate.current_time
                if self_gate_time < offset_seconds + duration_seconds:
                    raise AssertionError("ASR started before playback passed the chunk")

        gate = FakePlaybackGate()
        asr = FakeAsr(gate)
        windows = [
            AudioWindow(0.0, 2.0, np.zeros(32, dtype=np.float32), 16_000),
            AudioWindow(2.0, 4.0, np.zeros(32, dtype=np.float32), 16_000),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_realtime_audio_file_demo(
                RealtimeAudioFileDemoConfig(
                    audio_path=REPO_ROOT / "data" / "demo" / "audio" / "1.mp3",
                    output_dir=Path(temp_dir),
                    catalog_path=REPO_ROOT / "data" / "demo" / "product_catalog.csv",
                    promotions_path=REPO_ROOT / "data" / "demo" / "promotions.csv",
                ),
                playback_gate=gate,
                asr=asr,
                windows=windows,
            )

            self.assertEqual(gate.waits, [2.0, 4.0])
            self.assertEqual(
                [(offset, duration) for offset, duration, _ in asr.calls],
                [(0.0, 2.0), (2.0, 2.0)],
            )
            self.assertTrue(all(exists for _, _, exists in asr.calls))
            self.assertEqual(summary["processed_chunks"], 2)
            self.assertTrue((Path(temp_dir) / "realtime_file_captions.json").exists())

    def test_write_audio_window_wav_outputs_mono_pcm(self) -> None:
        import numpy as np
        import wave

        window = AudioWindow(0.0, 0.1, np.array([0.0, 0.5, -0.5], dtype=np.float32), 16_000)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunk.wav"
            write_audio_window_wav(window, path)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 16_000)
                self.assertEqual(handle.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
