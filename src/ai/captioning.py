from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import shutil

from src.schemas import (
    CaptionResult,
    CaptionSegment,
    repair_caption_timestamps,
    validate_caption_segments,
)


class CaptioningUnavailable(RuntimeError):
    """Raised when real ASR cannot run in the current environment."""


OPENAI_WHISPER_MODELS: Dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo": "turbo",
}

TYPHOON_WHISPER_MODELS: Dict[str, str] = {
    "large": "typhoon-ai/typhoon-whisper-large-v3",
    "large-v3": "typhoon-ai/typhoon-whisper-large-v3",
    "turbo": "typhoon-ai/typhoon-whisper-turbo",
    "medium": "typhoon-ai/monsoon-whisper-medium-gigaspeech2",
    "monsoon-medium": "typhoon-ai/monsoon-whisper-medium-gigaspeech2",
    "isan-medium": "typhoon-ai/typhoon-isan-asr-whisper",
    "isan": "typhoon-ai/typhoon-isan-asr-whisper",
}


@dataclass(frozen=True)
class CaptioningSettings:
    mode: str = "cached"
    language: str = "en"
    cached_transcript_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    asr_provider: str = "openai_whisper"
    asr_model: Optional[str] = None
    whisper_model: str = "tiny"
    asr_chunk_length_seconds: int = 30
    asr_batch_size: int = 16
    allow_cached_fallback: bool = True


class CaptioningEngine:
    def __init__(self, settings: CaptioningSettings):
        self.settings = settings

    def transcribe(self) -> CaptionResult:
        mode = self.settings.mode.lower().strip()
        if mode == "cached":
            return self._load_cached()
        if mode == "auto":
            try:
                return self._transcribe_with_selected_asr()
            except CaptioningUnavailable:
                if self.settings.allow_cached_fallback:
                    return self._load_cached()
                raise
        if mode in {"whisper", "openai_whisper"}:
            return self._transcribe_with_openai_whisper()
        if mode in {"typhoon", "typhoon_whisper"}:
            return self._transcribe_with_typhoon_whisper()
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

        model_name = resolve_asr_model_id(
            "openai_whisper",
            self.settings.asr_model or self.settings.whisper_model,
        )
        try:
            model = whisper.load_model(model_name)
            raw = model.transcribe(
                str(self.settings.audio_path),
                language=self.settings.language,
                task="transcribe",
                fp16=False,
                word_timestamps=False,
            )
        except Exception as exc:
            raise CaptioningUnavailable(
                f"OpenAI Whisper inference failed for model {model_name!r}"
            ) from exc
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
        segments = repair_caption_timestamps(segments)
        duration = segments[-1].end if segments else None
        return CaptionResult(
            language=str(raw.get("language", self.settings.language)),
            segments=segments,
            duration_seconds=duration,
        )

    def _transcribe_with_selected_asr(self) -> CaptionResult:
        provider = normalize_asr_provider(self.settings.asr_provider)
        if provider == "openai_whisper":
            return self._transcribe_with_openai_whisper()
        if provider == "typhoon_whisper":
            return self._transcribe_with_typhoon_whisper()
        raise CaptioningUnavailable(f"unsupported ASR provider: {self.settings.asr_provider}")

    def _transcribe_with_typhoon_whisper(self) -> CaptionResult:
        if not self.settings.audio_path:
            raise CaptioningUnavailable("audio path is not configured")
        if not self.settings.audio_path.exists():
            raise CaptioningUnavailable(
                f"audio path does not exist: {self.settings.audio_path}"
            )
        if shutil.which("ffmpeg") is None:
            raise CaptioningUnavailable("ffmpeg is not available on PATH")

        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except ImportError as exc:
            raise CaptioningUnavailable(
                "Typhoon Whisper requires transformers, torch, and accelerate; use requirements-asr.txt"
            ) from exc

        model_id = resolve_asr_model_id(
            "typhoon_whisper",
            self.settings.asr_model or self.settings.whisper_model,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=448,
                chunk_length_s=self.settings.asr_chunk_length_seconds,
                batch_size=self.settings.asr_batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
            raw = pipe(
                str(self.settings.audio_path),
                generate_kwargs={"language": typhoon_language(self.settings.language)},
            )
        except Exception as exc:
            raise CaptioningUnavailable(
                f"Typhoon Whisper inference failed for model {model_id!r}"
            ) from exc
        segments = _segments_from_transformers_asr_result(
            raw,
            source=f"typhoon_whisper:{model_id}",
        )
        duration = segments[-1].end if segments else None
        return CaptionResult(
            language=self.settings.language,
            segments=segments,
            duration_seconds=duration,
        )


def normalize_asr_provider(provider: str) -> str:
    normalized = (provider or "openai_whisper").strip().lower().replace("-", "_")
    if normalized in {"openai", "openai_whisper", "whisper"}:
        return "openai_whisper"
    if normalized in {"typhoon", "typhoon_whisper", "huggingface_typhoon"}:
        return "typhoon_whisper"
    return normalized


def available_asr_models() -> Dict[str, Dict[str, str]]:
    return {
        "openai_whisper": dict(OPENAI_WHISPER_MODELS),
        "typhoon_whisper": dict(TYPHOON_WHISPER_MODELS),
    }


def resolve_asr_model_id(provider: str, model: str) -> str:
    normalized_provider = normalize_asr_provider(provider)
    normalized_model = (model or "").strip()
    if not normalized_model:
        raise ValueError("ASR model must not be empty")
    if "/" in normalized_model:
        return normalized_model

    key = normalized_model.lower().replace("_", "-")
    if normalized_provider == "openai_whisper":
        if key in OPENAI_WHISPER_MODELS:
            return OPENAI_WHISPER_MODELS[key]
        raise ValueError(
            "unsupported OpenAI Whisper model "
            f"{model!r}; choose one of {sorted(OPENAI_WHISPER_MODELS)}"
        )
    if normalized_provider == "typhoon_whisper":
        if key in TYPHOON_WHISPER_MODELS:
            return TYPHOON_WHISPER_MODELS[key]
        raise ValueError(
            "unsupported Typhoon Whisper model "
            f"{model!r}; choose one of {sorted(TYPHOON_WHISPER_MODELS)} "
            "or pass a full Hugging Face model id"
        )
    raise ValueError(f"unsupported ASR provider: {provider}")


def typhoon_language(language: str) -> str:
    normalized = (language or "th").strip().lower()
    return "thai" if normalized in {"th", "tha", "thai"} else normalized


def _segments_from_transformers_asr_result(raw: Any, source: str) -> List[CaptionSegment]:
    chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
    segments = []
    for chunk in chunks:
        start, end = _coerce_timestamp_pair(chunk.get("timestamp"))
        text = str(chunk.get("text", "")).strip()
        if text:
            segments.append(
                CaptionSegment(
                    start=start,
                    end=end,
                    text=text,
                    source=source,
                    confidence=None,
                )
            )

    if not segments and isinstance(raw, dict):
        text = str(raw.get("text", "")).strip()
        if text:
            segments.append(
                CaptionSegment(
                    start=0.0,
                    end=0.05,
                    text=text,
                    source=source,
                    confidence=None,
                )
            )

    return repair_caption_timestamps(segments)


def _coerce_timestamp_pair(value: Any) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return 0.0, 0.05
    start = 0.0 if value[0] is None else float(value[0])
    end = start + 0.05 if value[1] is None else float(value[1])
    return start, end


def load_cached_transcript(path: Path) -> CaptionResult:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return CaptionResult.from_dict(raw)


def save_caption_json(result: CaptionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
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
