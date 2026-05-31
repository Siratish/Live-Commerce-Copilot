from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import shutil

from src.schemas import CaptionResult, CaptionSegment, validate_caption_segments


class CaptioningUnavailable(RuntimeError):
    """Raised when real ASR cannot run in the current environment."""


@dataclass(frozen=True)
class CaptioningSettings:
    mode: str = "cached"
    language: str = "en"
    cached_transcript_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    whisper_model: str = "tiny"
    allow_cached_fallback: bool = True


class CaptioningEngine:
    def __init__(self, settings: CaptioningSettings):
        self.settings = settings

    def transcribe(self) -> CaptionResult:
        mode = self.settings.mode.lower().strip()
        if mode == "cached":
            return self._load_cached()
        if mode in {"auto", "whisper"}:
            try:
                return self._transcribe_with_openai_whisper()
            except CaptioningUnavailable:
                if self.settings.allow_cached_fallback:
                    return self._load_cached()
                raise
        raise ValueError(f"unsupported captioning mode: {self.settings.mode}")

    def _load_cached(self) -> CaptionResult:
        if not self.settings.cached_transcript_path:
            raise CaptioningUnavailable("cached transcript path is not configured")
        return load_cached_transcript(self.settings.cached_transcript_path)

    def _transcribe_with_openai_whisper(self) -> CaptionResult:
        if not self.settings.audio_path:
            raise CaptioningUnavailable("audio path is not configured")
        if not self.settings.audio_path.exists():
            raise CaptioningUnavailable(
                f"audio path does not exist: {self.settings.audio_path}"
            )
        if shutil.which("ffmpeg") is None:
            raise CaptioningUnavailable("ffmpeg is not available on PATH")

        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise CaptioningUnavailable(
                "openai-whisper is not installed; use requirements-asr.txt"
            ) from exc

        model = whisper.load_model(self.settings.whisper_model)
        raw = model.transcribe(
            str(self.settings.audio_path),
            language=self.settings.language,
            task="transcribe",
            fp16=False,
            word_timestamps=False,
        )
        segments = [
            CaptionSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]).strip(),
                source="openai_whisper",
                confidence=None,
            )
            for item in raw.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        validate_caption_segments(segments)
        duration = segments[-1].end if segments else None
        return CaptionResult(
            language=str(raw.get("language", self.settings.language)),
            segments=segments,
            duration_seconds=duration,
        )


def load_cached_transcript(path: Path) -> CaptionResult:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return CaptionResult.from_dict(raw)


def save_caption_json(result: CaptionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def format_timestamp(seconds: float, separator: str = ".") -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def captions_to_webvtt(result: CaptionResult) -> str:
    lines = ["WEBVTT", ""]
    for segment in result.segments:
        lines.extend(
            [
                (
                    f"{format_timestamp(segment.start)} --> "
                    f"{format_timestamp(segment.end)}"
                ),
                segment.text,
                "",
            ]
        )
    return "\n".join(lines)


def captions_to_srt(result: CaptionResult) -> str:
    lines = []
    for index, segment in enumerate(result.segments, start=1):
        lines.extend(
            [
                str(index),
                (
                    f"{format_timestamp(segment.start, separator=',')} --> "
                    f"{format_timestamp(segment.end, separator=',')}"
                ),
                segment.text,
                "",
            ]
        )
    return "\n".join(lines)


def caption_metrics(result: CaptionResult) -> Dict[str, float]:
    segment_count = len(result.segments)
    coverage_seconds = sum(
        max(0.0, segment.end - segment.start) for segment in result.segments
    )
    duration = result.duration_seconds
    if duration is None and result.segments:
        duration = result.segments[-1].end
    duration = float(duration or 0.0)
    total_chars = sum(len(segment.text) for segment in result.segments)
    return {
        "segment_count": float(segment_count),
        "duration_seconds": duration,
        "coverage_seconds": coverage_seconds,
        "coverage_ratio": (coverage_seconds / duration) if duration else 0.0,
        "characters_per_second": (total_chars / coverage_seconds)
        if coverage_seconds
        else 0.0,
    }


def write_caption_outputs(
    result: CaptionResult,
    output_dir: Path,
    json_name: str = "captions.json",
    vtt_name: str = "captions.vtt",
    srt_name: str = "captions.srt",
    metrics_name: str = "caption_metrics.json",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / json_name,
        "webvtt": output_dir / vtt_name,
        "srt": output_dir / srt_name,
        "metrics": output_dir / metrics_name,
    }
    save_caption_json(result, paths["json"])
    write_text(paths["webvtt"], captions_to_webvtt(result))
    write_text(paths["srt"], captions_to_srt(result))
    with paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(caption_metrics(result), handle, indent=2)
        handle.write("\n")
    return paths
