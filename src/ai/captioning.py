from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import json
import shutil
import subprocess

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
class AudioWindow:
    start: float
    end: float
    samples: Any
    sample_rate: int


@dataclass(frozen=True)
class CaptioningSettings:
    mode: str = "cached"
    language: str = "en"
    cached_transcript_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    asr_provider: str = "openai_whisper"
    asr_model: Optional[str] = None
    whisper_model: str = "tiny"
    asr_chunk_length_seconds: int = 4
    asr_dynamic_chunking: bool = True
    asr_min_chunk_seconds: float = 1.0
    asr_pause_seconds: float = 0.7
    asr_silence_threshold: float = 0.012
    asr_frame_seconds: float = 0.1
    asr_batch_size: int = 16
    asr_max_new_tokens: int = 440
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
            max_new_tokens = safe_whisper_max_new_tokens(
                requested=self.settings.asr_max_new_tokens,
                max_target_positions=getattr(model.config, "max_target_positions", None),
            )
            processor = AutoProcessor.from_pretrained(model_id)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=self.settings.asr_chunk_length_seconds,
                batch_size=self.settings.asr_batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
            segments = transcribe_audio_windows(
                pipe=pipe,
                audio_path=self.settings.audio_path,
                language=self.settings.language,
                source=f"typhoon_whisper:{model_id}",
                chunk_length_seconds=self.settings.asr_chunk_length_seconds,
                max_new_tokens=max_new_tokens,
                dynamic_chunking=self.settings.asr_dynamic_chunking,
                min_chunk_seconds=self.settings.asr_min_chunk_seconds,
                pause_seconds=self.settings.asr_pause_seconds,
                silence_threshold=self.settings.asr_silence_threshold,
                frame_seconds=self.settings.asr_frame_seconds,
            )
        except Exception as exc:
            raise CaptioningUnavailable(
                f"Typhoon Whisper inference failed for model {model_id!r}"
            ) from exc
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


def safe_whisper_max_new_tokens(
    requested: int,
    max_target_positions: Optional[int],
    decoder_prompt_margin: int = 8,
) -> int:
    """Keep Whisper generation under the decoder context limit."""
    requested = max(1, int(requested))
    if not max_target_positions:
        return requested
    return max(1, min(requested, int(max_target_positions) - decoder_prompt_margin))


def transcribe_audio_windows(
    pipe: Any,
    audio_path: Path,
    language: str,
    source: str,
    chunk_length_seconds: int,
    max_new_tokens: int,
    dynamic_chunking: bool = True,
    min_chunk_seconds: float = 1.0,
    pause_seconds: float = 0.7,
    silence_threshold: float = 0.012,
    frame_seconds: float = 0.1,
) -> List[CaptionSegment]:
    segments: List[CaptionSegment] = []
    for window in iter_audio_windows(
        audio_path,
        chunk_length_seconds,
        dynamic_chunking=dynamic_chunking,
        min_chunk_seconds=min_chunk_seconds,
        pause_seconds=pause_seconds,
        silence_threshold=silence_threshold,
        frame_seconds=frame_seconds,
    ):
        raw = pipe(
            {"raw": window.samples, "sampling_rate": window.sample_rate},
            generate_kwargs={
                "language": typhoon_language(language),
                "max_new_tokens": max_new_tokens,
            },
        )
        segments.extend(
            _segments_from_transformers_asr_result(
                raw,
                source=source,
                offset_seconds=window.start,
                fallback_start=window.start,
                fallback_end=window.end,
            )
        )
    return repair_caption_timestamps(segments)


def iter_audio_windows(
    audio_path: Path,
    chunk_length_seconds: int,
    sample_rate: int = 16_000,
    dynamic_chunking: bool = True,
    min_chunk_seconds: float = 1.0,
    pause_seconds: float = 0.7,
    silence_threshold: float = 0.012,
    frame_seconds: float = 0.1,
) -> Iterator[AudioWindow]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise CaptioningUnavailable(
            "Typhoon streaming-window transcription requires numpy"
        ) from exc

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr_bytes = process.communicate()
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    return_code = process.wait()
    if return_code and not stdout:
        raise CaptioningUnavailable(f"ffmpeg failed to decode audio: {stderr.strip()}")

    samples = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
    windows = (
        iter_pause_audio_windows_from_samples(
            samples=samples,
            sample_rate=sample_rate,
            max_chunk_seconds=chunk_length_seconds,
            min_chunk_seconds=min_chunk_seconds,
            pause_seconds=pause_seconds,
            silence_threshold=silence_threshold,
            frame_seconds=frame_seconds,
        )
        if dynamic_chunking
        else iter_fixed_audio_windows_from_samples(
            samples=samples,
            sample_rate=sample_rate,
            chunk_length_seconds=chunk_length_seconds,
        )
    )
    yield from windows


def iter_fixed_audio_windows_from_samples(
    samples: Any,
    sample_rate: int,
    chunk_length_seconds: float,
) -> Iterator[AudioWindow]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise CaptioningUnavailable("audio windowing requires numpy") from exc

    values = np.asarray(samples, dtype=np.float32)
    chunk_seconds = max(0.1, float(chunk_length_seconds))
    samples_per_window = max(1, int(sample_rate * chunk_seconds))
    for start_index in range(0, len(values), samples_per_window):
        end_index = min(len(values), start_index + samples_per_window)
        if end_index <= start_index:
            continue
        start = float(start_index) / float(sample_rate)
        end = float(end_index) / float(sample_rate)
        yield AudioWindow(start=start, end=end, samples=values[start_index:end_index], sample_rate=sample_rate)


def iter_pause_audio_windows_from_samples(
    samples: Any,
    sample_rate: int,
    max_chunk_seconds: float,
    min_chunk_seconds: float = 1.0,
    pause_seconds: float = 0.7,
    silence_threshold: float = 0.012,
    frame_seconds: float = 0.1,
) -> Iterator[AudioWindow]:
    """Split audio at speaker pauses using a simple frame-energy heuristic."""
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise CaptioningUnavailable("pause-aware audio windowing requires numpy") from exc

    values = np.asarray(samples, dtype=np.float32)
    if len(values) == 0:
        return

    frame_samples = max(1, int(sample_rate * max(0.02, float(frame_seconds))))
    max_samples = max(frame_samples, int(sample_rate * max(0.1, float(max_chunk_seconds))))
    min_samples = max(frame_samples, int(sample_rate * max(0.0, float(min_chunk_seconds))))
    pause_samples = max(frame_samples, int(sample_rate * max(0.0, float(pause_seconds))))
    threshold = max(0.0, float(silence_threshold))

    start_index: Optional[int] = None
    last_speech_end: Optional[int] = None
    index = 0

    while index < len(values):
        frame_end = min(len(values), index + frame_samples)
        frame = values[index:frame_end]
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        is_speech = rms >= threshold

        if is_speech:
            if start_index is None:
                start_index = index
            last_speech_end = frame_end

        if start_index is not None and last_speech_end is not None:
            window_samples = frame_end - start_index
            speech_samples = last_speech_end - start_index
            trailing_silence = frame_end - last_speech_end
            if trailing_silence >= pause_samples and speech_samples >= min_samples:
                yield _audio_window_from_slice(values, sample_rate, start_index, last_speech_end)
                start_index = None
                last_speech_end = None
            elif window_samples >= max_samples:
                yield _audio_window_from_slice(values, sample_rate, start_index, frame_end)
                start_index = None
                last_speech_end = None

        index = frame_end

    if start_index is not None and last_speech_end is not None:
        end_index = last_speech_end
        if end_index <= start_index:
            end_index = min(len(values), start_index + frame_samples)
        yield _audio_window_from_slice(values, sample_rate, start_index, end_index)


def _audio_window_from_slice(
    samples: Any,
    sample_rate: int,
    start_index: int,
    end_index: int,
) -> AudioWindow:
    start = float(start_index) / float(sample_rate)
    end = float(end_index) / float(sample_rate)
    return AudioWindow(
        start=start,
        end=end,
        samples=samples[start_index:end_index],
        sample_rate=sample_rate,
    )


def _segments_from_transformers_asr_result(
    raw: Any,
    source: str,
    offset_seconds: float = 0.0,
    fallback_start: float = 0.0,
    fallback_end: Optional[float] = None,
) -> List[CaptionSegment]:
    chunks = raw.get("chunks", []) if isinstance(raw, dict) else []
    segments = []
    for chunk in chunks:
        start, end = _coerce_timestamp_pair(chunk.get("timestamp"))
        text = str(chunk.get("text", "")).strip()
        if text:
            segments.append(
                CaptionSegment(
                    start=offset_seconds + start,
                    end=offset_seconds + end,
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
                    start=fallback_start,
                    end=fallback_end if fallback_end is not None else fallback_start + 0.05,
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
