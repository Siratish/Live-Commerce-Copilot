from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import base64
import json
import queue
import subprocess
import sys
import threading
import uuid

from src.ai.captioning import (
    CaptioningUnavailable,
    _segments_from_transformers_asr_result,
    normalize_asr_provider,
    resolve_asr_model_id,
    safe_whisper_max_new_tokens,
    save_caption_json,
    typhoon_language,
)
from src.ai.commerce_actions import generate_commerce_actions, save_commerce_actions
from src.ai.decision import CommerceDecisionProvider
from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import CaptionResult, CaptionSegment, repair_caption_timestamps
from src.utils.live_display import display_stream_state_panel


@dataclass(frozen=True)
class LiveMicDemoConfig:
    chunk_seconds: float = 4.0
    min_chunk_seconds: float = 1.0
    pause_seconds: float = 0.7
    silence_threshold: float = 0.015
    max_chunks: Optional[int] = 6
    language: str = "th"
    asr_provider: str = "openai_whisper"
    asr_model: str = "base"
    asr_max_new_tokens: int = 256
    install_asr_deps: bool = False
    show_debug_panel: bool = True
    continuous_recording: bool = True
    max_queue_chunks: int = 16
    poll_interval_seconds: float = 0.25
    output_dir: Path = Path("outputs/live_mic")
    catalog_path: Path = Path("data/demo/product_catalog.csv")
    promotions_path: Path = Path("data/demo/promotions.csv")


class BrowserMicChunkSource:
    """Callback-driven Colab mic source for background ASR workers."""

    def __init__(self, config: LiveMicDemoConfig) -> None:
        self.config = config
        suffix = uuid.uuid4().hex
        self.chunk_callback = f"live_commerce_mic_chunk_{suffix}"
        self.done_callback = f"live_commerce_mic_done_{suffix}"
        self.error_callback = f"live_commerce_mic_error_{suffix}"
        self.start_callback = f"live_commerce_mic_start_{suffix}"
        self.stop_callback = f"live_commerce_mic_stop_{suffix}"
        self._chunks: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._started = threading.Event()
        self._done = threading.Event()
        self._stopped = threading.Event()
        self.error: Optional[str] = None

    def install(self) -> None:
        self.install_recorder()
        install_colab_live_mic_debug_panel(
            config=self.config,
            callbacks=self.callbacks,
        )

    @property
    def callbacks(self) -> Dict[str, str]:
        return {
            "chunk": self.chunk_callback,
            "done": self.done_callback,
            "error": self.error_callback,
            "start": self.start_callback,
            "stop": self.stop_callback,
        }

    def install_recorder(self) -> None:
        from google.colab import output  # type: ignore

        output.register_callback(self.chunk_callback, self._handle_chunk)
        output.register_callback(self.done_callback, self._handle_done)
        output.register_callback(self.error_callback, self._handle_error)
        output.register_callback(self.start_callback, self._handle_start)
        output.register_callback(self.stop_callback, self._handle_stop)
        _install_colab_mic_recorder()

    def start(self) -> None:
        self._handle_start()
        try:
            from google.colab import output  # type: ignore

            max_milliseconds = max(500, int(float(self.config.chunk_seconds) * 1000))
            min_milliseconds = max(200, int(float(self.config.min_chunk_seconds) * 1000))
            silence_milliseconds = max(100, int(float(self.config.pause_seconds) * 1000))
            max_chunks_js = (
                "null"
                if self.config.max_chunks is None
                else str(max(1, int(self.config.max_chunks)))
            )
            output.eval_js(
                "window.liveCommerceMic.startRequested = true; "
                "window.liveCommerceMic.stopRequested = false; "
                "window.liveCommerceMic.startBufferedRecording("
                f"{max_milliseconds}, {min_milliseconds}, "
                f"{silence_milliseconds}, {float(self.config.silence_threshold)}, "
                f"{max_chunks_js}, {max(1, int(self.config.max_queue_chunks))}, "
                f"{json.dumps(self.chunk_callback)}, "
                f"{json.dumps(self.done_callback)}, "
                f"{json.dumps(self.error_callback)})"
            )
        except Exception as exc:
            self._handle_error({"error": str(exc)})

    def _handle_start(self, *_args: Any) -> Dict[str, Any]:
        self._started.set()
        return {"started": True}

    def _handle_stop(self, *_args: Any) -> Dict[str, Any]:
        self.stop(call_browser=False)
        return {"stopped": True}

    def _handle_chunk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._stopped.is_set() and isinstance(payload, dict):
            self._chunks.put(payload)
        return {"queueDepth": self._chunks.qsize()}

    def _handle_done(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._done.set()
        self._chunks.put(None)
        return {"done": True, "queueDepth": self._chunks.qsize()}

    def _handle_error(self, payload: Any = None) -> Dict[str, Any]:
        if isinstance(payload, dict):
            self.error = str(payload.get("error") or payload.get("status") or payload)
        else:
            self.error = str(payload)
        self._done.set()
        self._chunks.put(None)
        return {"error": self.error}

    def wait_for_start(self, timeout: Optional[float] = None) -> bool:
        return self._started.wait(timeout)

    def pop_chunk(self, timeout: float) -> Optional[Dict[str, Any]]:
        try:
            return self._chunks.get(timeout=max(0.05, float(timeout)))
        except queue.Empty:
            return None

    def is_done(self) -> bool:
        return self._done.is_set() or self._stopped.is_set()

    def stop(self, call_browser: bool = True) -> None:
        self._stopped.set()
        self._done.set()
        self._started.set()
        self._chunks.put(None)
        if not call_browser:
            return
        try:
            from google.colab import output  # type: ignore

            output.eval_js("window.liveCommerceMic && window.liveCommerceMic.stop()")
        except Exception:
            pass


_OPENAI_WHISPER_MODEL_CACHE: Dict[str, Any] = {}


def preload_openai_whisper_model(model_name: str = "turbo") -> Any:
    """Load and cache an OpenAI Whisper model for notebook demos."""
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise CaptioningUnavailable(
            "openai-whisper is not installed. Set install_asr_deps=True or run pip install -r requirements-asr.txt."
        ) from exc

    if model_name not in _OPENAI_WHISPER_MODEL_CACHE:
        if model_name in {"large", "large-v2", "large-v3", "turbo"}:
            print(f"Loading OpenAI Whisper {model_name}. This may take a while.")
        _OPENAI_WHISPER_MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _OPENAI_WHISPER_MODEL_CACHE[model_name]


class _OpenAIWhisperLiveASR:
    def __init__(self, model_name: str, language: str):
        self.model_name = model_name
        self.language = language
        self.model = preload_openai_whisper_model(model_name)

    def transcribe_chunk(
        self,
        chunk_path: Path,
        offset_seconds: float,
        fallback_duration_seconds: float,
    ) -> List[CaptionSegment]:
        raw = self.model.transcribe(
            str(chunk_path),
            language=self.language,
            task="transcribe",
            fp16=False,
            word_timestamps=False,
        )
        segments = [
            CaptionSegment(
                start=offset_seconds + float(item["start"]),
                end=offset_seconds + float(item["end"]),
                text=str(item.get("text", "")).strip(),
                source=f"live_mic_openai_whisper:{self.model_name}",
                confidence=None,
            )
            for item in raw.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        if not segments and str(raw.get("text", "")).strip():
            segments.append(
                CaptionSegment(
                    start=offset_seconds,
                    end=offset_seconds + fallback_duration_seconds,
                    text=str(raw["text"]).strip(),
                    source=f"live_mic_openai_whisper:{self.model_name}",
                    confidence=None,
                )
            )
        return repair_caption_timestamps(segments)


class _TyphoonWhisperLiveASR:
    def __init__(
        self,
        model_id: str,
        language: str,
        chunk_seconds: float,
        max_new_tokens: int,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # type: ignore
        except ImportError as exc:
            raise CaptioningUnavailable(
                "Typhoon Whisper requires transformers, torch, and accelerate. Set install_asr_deps=True or run pip install -r requirements-asr.txt."
            ) from exc

        print(f"Loading Typhoon Whisper model {model_id}. This may take a while.")
        self.model_id = model_id
        self.language = language
        self.chunk_seconds = chunk_seconds
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        self.max_new_tokens = safe_whisper_max_new_tokens(
            requested=max_new_tokens,
            max_target_positions=getattr(model.config, "max_target_positions", None),
        )
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=max(1, int(chunk_seconds)),
            batch_size=1,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe_chunk(
        self,
        chunk_path: Path,
        offset_seconds: float,
        fallback_duration_seconds: float,
    ) -> List[CaptionSegment]:
        raw = self.pipe(
            str(chunk_path),
            generate_kwargs={
                "language": typhoon_language(self.language),
                "max_new_tokens": self.max_new_tokens,
            },
        )
        return _segments_from_transformers_asr_result(
            raw,
            source=f"live_mic_typhoon_whisper:{self.model_id}",
            offset_seconds=offset_seconds,
            fallback_start=offset_seconds,
            fallback_end=offset_seconds + fallback_duration_seconds,
        )


def run_colab_live_mic_demo(
    config: LiveMicDemoConfig = LiveMicDemoConfig(),
    decision_provider: Optional[CommerceDecisionProvider] = None,
    chunk_source: Optional[BrowserMicChunkSource] = None,
    asr: Optional[Any] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Record browser mic chunks in Colab, transcribe them, and emit actions."""
    _install_asr_dependencies(config.install_asr_deps)
    if chunk_source is not None:
        pass
    elif config.show_debug_panel:
        install_colab_live_mic_debug_panel()
    else:
        _install_colab_mic_recorder()
    asr_engine = asr or _build_live_asr(config)
    catalog = load_product_catalog(config.catalog_path)
    promotions = load_promotions(config.promotions_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = config.output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    live_segments: List[CaptionSegment] = []
    live_actions = []
    elapsed_seconds = 0.0
    processed_chunks = 0
    skipped_chunks = 0
    display_key = f"live_mic_stream_panel_{uuid.uuid4().hex}"
    buffer_status: Dict[str, Any] = {}

    try:
        if config.continuous_recording:
            if chunk_source is None:
                _start_colab_mic_buffer(config)
            else:
                chunk_source.wait_for_start()
            while not _chunk_limit_reached(processed_chunks, config.max_chunks):
                if cancel_event and cancel_event.is_set():
                    break
                if chunk_source is None:
                    payload = _pop_colab_mic_buffered_chunk(config.poll_interval_seconds)
                else:
                    payload = chunk_source.pop_chunk(config.poll_interval_seconds)
                if payload is None:
                    if chunk_source is None:
                        buffer_status = _get_colab_mic_buffer_status()
                        if buffer_status.get("error"):
                            raise RuntimeError(
                                f"browser mic buffered recorder failed: {buffer_status['error']}"
                            )
                        if (
                            buffer_status.get("done")
                            and int(buffer_status.get("queueDepth") or 0) == 0
                        ):
                            break
                    else:
                        if chunk_source.error:
                            raise RuntimeError(
                                f"browser mic buffered recorder failed: {chunk_source.error}"
                            )
                        if chunk_source.is_done():
                            break
                    continue

                processed_chunks += 1
                chunk_number = int(payload.get("chunkIndex") or processed_chunks)
                (
                    live_segments,
                    live_actions,
                    elapsed_seconds,
                    chunk_path,
                    skipped,
                ) = _process_live_mic_payload(
                    payload=payload,
                    chunk_number=chunk_number,
                    config=config,
                    chunk_dir=chunk_dir,
                    asr=asr_engine,
                    catalog=catalog,
                    promotions=promotions,
                    decision_provider=decision_provider,
                    live_segments=live_segments,
                    elapsed_seconds=elapsed_seconds,
                    display_key=display_key,
                )
                if skipped:
                    skipped_chunks += 1
                if cancel_event and cancel_event.is_set():
                    break
                _publish_colab_mic_debug(
                    {
                        "chunkIndex": chunk_number,
                        "state": "completed",
                        "status": "silence_skipped" if skipped else "asr_done",
                        "durationSeconds": payload.get("durationSeconds"),
                        "captionCount": len(live_segments),
                        "actionCount": len(live_actions),
                        "skippedChunks": skipped_chunks,
                        "queueDepth": payload.get("queueDepth"),
                        "droppedChunks": payload.get("droppedChunks"),
                    }
                )
        else:
            if config.max_chunks is None:
                raise RuntimeError(
                    "non-continuous live mic mode requires max_chunks; "
                    "use continuous_recording=True for an open-ended mic session"
                )
            for chunk_index in range(max(1, int(config.max_chunks))):
                payload = _record_colab_mic_chunk(
                    chunk_index=chunk_index + 1,
                    max_seconds=config.chunk_seconds,
                    min_seconds=config.min_chunk_seconds,
                    pause_seconds=config.pause_seconds,
                    silence_threshold=config.silence_threshold,
                )
                processed_chunks += 1
                (
                    live_segments,
                    live_actions,
                    elapsed_seconds,
                    chunk_path,
                    skipped,
                ) = _process_live_mic_payload(
                    payload=payload,
                    chunk_number=chunk_index + 1,
                    config=config,
                    chunk_dir=chunk_dir,
                    asr=asr_engine,
                    catalog=catalog,
                    promotions=promotions,
                    decision_provider=decision_provider,
                    live_segments=live_segments,
                    elapsed_seconds=elapsed_seconds,
                    display_key=display_key,
                )
                if skipped:
                    skipped_chunks += 1
                _publish_colab_mic_debug(
                    {
                        "chunkIndex": chunk_index + 1,
                        "state": "completed",
                        "status": "silence_skipped" if skipped else "asr_done",
                        "durationSeconds": payload.get("durationSeconds"),
                        "captionCount": len(live_segments),
                        "actionCount": len(live_actions),
                        "skippedChunks": skipped_chunks,
                    }
                )
    finally:
        if chunk_source is None:
            _stop_colab_mic()
        elif cancel_event and cancel_event.is_set():
            chunk_source.stop()

    if config.continuous_recording:
        buffer_status = _get_colab_mic_buffer_status() or buffer_status

    return {
        "caption_count": len(live_segments),
        "action_count": len(live_actions),
        "processed_chunks": processed_chunks,
        "skipped_chunks": skipped_chunks,
        "dropped_chunks": buffer_status.get("droppedChunks", 0),
        "captions_json": str(config.output_dir / "live_mic_captions.json"),
        "actions_json": str(config.output_dir / "live_mic_actions.json"),
        "chunk_dir": str(chunk_dir),
    }


def _process_live_mic_payload(
    payload: Dict[str, Any],
    chunk_number: int,
    config: LiveMicDemoConfig,
    chunk_dir: Path,
    asr: Any,
    catalog: Sequence[Any],
    promotions: Sequence[Any],
    decision_provider: Optional[CommerceDecisionProvider],
    live_segments: Sequence[CaptionSegment],
    elapsed_seconds: float,
    display_key: Optional[str] = None,
) -> tuple[List[CaptionSegment], List[Any], float, Optional[Path], bool]:
    duration_seconds = float(payload.get("durationSeconds") or config.chunk_seconds)
    chunk_path = chunk_dir / f"mic_chunk_{max(0, chunk_number - 1):03d}.webm"
    offset_raw = payload.get("sessionOffsetSeconds")
    offset_seconds = (
        elapsed_seconds if offset_raw is None else max(0.0, float(offset_raw))
    )
    audio_stats = _mic_payload_audio_stats(payload)
    if _is_silent_mic_payload(payload, config):
        next_elapsed_seconds = max(elapsed_seconds, offset_seconds + duration_seconds)
        repaired_segments = list(live_segments)
        caption_result = CaptionResult(
            language=config.language,
            segments=repaired_segments,
            duration_seconds=next_elapsed_seconds,
        )
        live_actions = generate_commerce_actions(
            caption_result,
            catalog,
            promotions,
            decision_provider=decision_provider,
        )
        save_caption_json(caption_result, config.output_dir / "live_mic_captions.json")
        save_commerce_actions(live_actions, config.output_dir / "live_mic_actions.json")
        _publish_colab_mic_debug(
            {
                "chunkIndex": chunk_number,
                "state": "skipped_silence",
                "status": "below_silence_threshold",
                "durationSeconds": duration_seconds,
                "size": payload.get("size"),
                "queueDepth": payload.get("queueDepth"),
                "droppedChunks": payload.get("droppedChunks"),
                "maxRms": audio_stats.get("maxRms"),
                "meanRms": audio_stats.get("meanRms"),
                "speechDetected": audio_stats.get("speechDetected"),
            }
        )
        _display_live_state(
            chunk_index=chunk_number - 1,
            config=config,
            segments=repaired_segments,
            actions=live_actions,
            chunk_path=None,
            queue_status={
                "queueDepth": payload.get("queueDepth"),
                "droppedChunks": payload.get("droppedChunks"),
            },
            state_label="Skipped Silence",
            detail="This mic chunk did not contain detected speech, so ASR and action generation were skipped.",
            audio_stats=audio_stats,
            display_key=display_key,
        )
        return repaired_segments, live_actions, next_elapsed_seconds, None, True

    _write_data_url(payload["dataUrl"], chunk_path)
    _publish_colab_mic_debug(
        {
            "chunkIndex": chunk_number,
            "state": "transcribing",
            "status": "dequeued" if config.continuous_recording else "recorded",
            "durationSeconds": duration_seconds,
            "size": payload.get("size"),
            "queueDepth": payload.get("queueDepth"),
            "droppedChunks": payload.get("droppedChunks"),
            "maxRms": audio_stats.get("maxRms"),
            "meanRms": audio_stats.get("meanRms"),
            "speechDetected": audio_stats.get("speechDetected"),
        }
    )

    new_segments = asr.transcribe_chunk(
        chunk_path=chunk_path,
        offset_seconds=offset_seconds,
        fallback_duration_seconds=duration_seconds,
    )
    next_elapsed_seconds = max(elapsed_seconds, offset_seconds + duration_seconds)
    repaired_segments = repair_caption_timestamps([*live_segments, *new_segments])
    caption_result = CaptionResult(
        language=config.language,
        segments=repaired_segments,
        duration_seconds=next_elapsed_seconds,
    )
    live_actions = generate_commerce_actions(
        caption_result,
        catalog,
        promotions,
        decision_provider=decision_provider,
    )
    save_caption_json(caption_result, config.output_dir / "live_mic_captions.json")
    save_commerce_actions(live_actions, config.output_dir / "live_mic_actions.json")
    _display_live_state(
        chunk_index=chunk_number - 1,
        config=config,
        segments=repaired_segments,
        actions=live_actions,
        chunk_path=chunk_path,
        queue_status={
            "queueDepth": payload.get("queueDepth"),
            "droppedChunks": payload.get("droppedChunks"),
        },
        state_label="Transcribing",
        audio_stats=audio_stats,
        display_key=display_key,
    )
    return repaired_segments, live_actions, next_elapsed_seconds, chunk_path, False


def _mic_payload_audio_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "maxRms": _optional_float(payload.get("maxRms")),
        "meanRms": _optional_float(payload.get("meanRms")),
        "speechDetected": payload.get("speechDetected"),
        "silenceThreshold": _optional_float(payload.get("silenceThreshold")),
    }


def _is_silent_mic_payload(payload: Dict[str, Any], config: LiveMicDemoConfig) -> bool:
    if "speechDetected" not in payload and "maxRms" not in payload:
        return False
    stats = _mic_payload_audio_stats(payload)
    threshold = max(0.0, float(stats.get("silenceThreshold") or config.silence_threshold))
    max_rms = float(stats.get("maxRms") or 0.0)
    mean_rms = float(stats.get("meanRms") or 0.0)
    speech_detected = bool(stats.get("speechDetected"))
    if threshold <= 0:
        return False
    return (
        not speech_detected
        and max_rms < threshold
        and mean_rms < threshold * 0.5
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _install_asr_dependencies(install: bool) -> None:
    if install:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-asr.txt"]
        )


def _chunk_limit_reached(processed_chunks: int, max_chunks: Optional[int]) -> bool:
    if max_chunks is None:
        return False
    return processed_chunks >= max(1, int(max_chunks))


def _build_live_asr(config: LiveMicDemoConfig):
    provider = normalize_asr_provider(config.asr_provider)
    model_id = resolve_asr_model_id(provider, config.asr_model)
    if provider == "openai_whisper":
        return _OpenAIWhisperLiveASR(model_id, config.language)
    if provider == "typhoon_whisper":
        return _TyphoonWhisperLiveASR(
            model_id=model_id,
            language=config.language,
            chunk_seconds=config.chunk_seconds,
            max_new_tokens=config.asr_max_new_tokens,
        )
    raise CaptioningUnavailable(f"unsupported live mic ASR provider: {config.asr_provider}")


def _install_colab_mic_recorder() -> None:
    try:
        from IPython.display import Javascript, display  # type: ignore
        from google.colab import output  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Live mic recording requires Google Colab browser APIs.") from exc

    display(
        Javascript(
            """
            (() => {
              window.liveCommerceMic = window.liveCommerceMic || {};
              window.liveCommerceMic.publishDebug = window.liveCommerceMic.publishDebug || function(payload) {
                const state = window.liveCommerceMic;
                state.latestDebug = Object.assign({}, state.latestDebug || {}, payload || {});
                if (typeof state.updateDebug === 'function') {
                  state.updateDebug(state.latestDebug);
                }
              };
              window.liveCommerceMic.emitCallback = function(callbackName, payload) {
                if (!callbackName || !window.google || !google.colab || !google.colab.kernel) return;
                try {
                  google.colab.kernel.invokeFunction(callbackName, [payload || {}], {});
                } catch (error) {
                  console.warn('Live mic callback failed', error);
                }
              };
              window.liveCommerceMic.recordChunk = async function(
                maxMilliseconds,
                minMilliseconds,
                silenceMilliseconds,
                silenceThreshold,
                chunkIndex
              ) {
                const state = window.liveCommerceMic;
                if (!state.stream) {
                  state.stream = await navigator.mediaDevices.getUserMedia({audio: true});
                }
                if (!state.audioContext) {
                  state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                  state.source = state.audioContext.createMediaStreamSource(state.stream);
                  state.analyser = state.audioContext.createAnalyser();
                  state.analyser.fftSize = 1024;
                  state.source.connect(state.analyser);
                }
                if (state.audioContext.state === 'suspended') {
                  await state.audioContext.resume();
                }
                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                  ? 'audio/webm;codecs=opus'
                  : 'audio/webm';
                const startedAt = performance.now();
                return await new Promise((resolve, reject) => {
                  const chunks = [];
                  const recorder = new MediaRecorder(state.stream, {mimeType});
                  state.activeRecorder = recorder;
                  const waveform = new Uint8Array(state.analyser.fftSize);
                  let animationId = null;
                  let stopped = false;
                  let lastSpeechAt = startedAt;
                  let stopReason = null;
                  let maxRms = 0;
                  let rmsTotal = 0;
                  let rmsFrames = 0;
                  let speechDetected = false;
                  const stopRecorder = (reason) => {
                    if (!stopped && recorder.state !== 'inactive') {
                      stopped = true;
                      stopReason = reason;
                      recorder.stop();
                    }
                  };
                  const monitorPause = () => {
                    const now = performance.now();
                    state.analyser.getByteTimeDomainData(waveform);
                    let sumSquares = 0;
                    for (const value of waveform) {
                      const centered = (value - 128) / 128;
                      sumSquares += centered * centered;
                    }
                    const rms = Math.sqrt(sumSquares / waveform.length);
                    const isSilence = rms < silenceThreshold;
                    maxRms = Math.max(maxRms, rms);
                    rmsTotal += rms;
                    rmsFrames += 1;
                    if (!isSilence) {
                      lastSpeechAt = now;
                      speechDetected = true;
                    }
                    const elapsed = now - startedAt;
                    const silenceFor = now - lastSpeechAt;
                    state.publishDebug({
                      chunkIndex: chunkIndex || null,
                      state: 'recording',
                      status: isSilence ? 'silence' : 'speech',
                      rms: rms,
                      threshold: silenceThreshold,
                      elapsedSeconds: elapsed / 1000,
                      silenceSeconds: silenceFor / 1000,
                      maxSeconds: maxMilliseconds / 1000,
                      minSeconds: minMilliseconds / 1000,
                      pauseSeconds: silenceMilliseconds / 1000
                    });
                    if (
                      elapsed >= maxMilliseconds
                    ) {
                      stopRecorder('max_duration');
                      return;
                    }
                    if (elapsed >= minMilliseconds && silenceFor >= silenceMilliseconds) {
                      stopRecorder('pause_detected');
                      return;
                    }
                    animationId = requestAnimationFrame(monitorPause);
                  };
                  recorder.ondataavailable = event => {
                    if (event.data && event.data.size > 0) chunks.push(event.data);
                  };
                  recorder.onerror = event => reject(event.error || event);
                  recorder.onstop = () => {
                    if (animationId !== null) cancelAnimationFrame(animationId);
                    const stoppedAt = performance.now();
                    const blob = new Blob(chunks, {type: mimeType});
                    if (state.activeRecorder === recorder) {
                      state.activeRecorder = null;
                    }
                    state.publishDebug({
                      chunkIndex: chunkIndex || null,
                      state: 'recorded',
                      status: stopReason || 'stopped',
                      elapsedSeconds: (stoppedAt - startedAt) / 1000,
                      durationSeconds: (stoppedAt - startedAt) / 1000,
                      size: blob.size,
                      maxRms,
                      meanRms: rmsFrames ? rmsTotal / rmsFrames : 0,
                      speechDetected,
                      silenceThreshold
                    });
                    const reader = new FileReader();
                    reader.onloadend = () => resolve({
                      dataUrl: reader.result,
                      mimeType,
                      chunkIndex: chunkIndex || null,
                      stopReason: stopReason || 'stopped',
                      size: blob.size,
                      durationSeconds: (stoppedAt - startedAt) / 1000,
                      startedAtMs: startedAt,
                      stoppedAtMs: stoppedAt,
                      maxRms,
                      meanRms: rmsFrames ? rmsTotal / rmsFrames : 0,
                      speechDetected,
                      silenceThreshold,
                      sessionOffsetSeconds: state.bufferSessionStartedAt
                        ? (startedAt - state.bufferSessionStartedAt) / 1000
                        : null
                    });
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                  };
                  recorder.start();
                  animationId = requestAnimationFrame(monitorPause);
                });
              };
              window.liveCommerceMic.startBufferedRecording = async function(
                maxMilliseconds,
                minMilliseconds,
                silenceMilliseconds,
                silenceThreshold,
                maxChunks,
                maxQueueChunks,
                chunkCallbackName,
                doneCallbackName,
                errorCallbackName
              ) {
                const state = window.liveCommerceMic;
                if (state.bufferLoopActive) {
                  return {
                    started: false,
                    reason: 'already_active',
                    queueDepth: (state.bufferQueue || []).length
                  };
                }
                state.stopRequested = Boolean(state.stopRequested);
                state.startRequested = Boolean(state.startRequested);
                state.bufferQueue = [];
                state.bufferWaiters = [];
                state.bufferDone = false;
                state.bufferError = null;
                state.droppedChunks = 0;
                state.maxQueueChunks = Math.max(1, Number(maxQueueChunks || 16));
                state.waitForStart = function() {
                  if (state.startRequested) return Promise.resolve(true);
                  return new Promise(resolve => {
                    state.resolveStart = resolve;
                  });
                };
                state.publishDebug({
                  chunkIndex: 1,
                  state: 'ready',
                  status: 'press_start',
                  queueDepth: 0,
                  droppedChunks: 0,
                  maxSeconds: maxMilliseconds / 1000,
                  minSeconds: minMilliseconds / 1000,
                  pauseSeconds: silenceMilliseconds / 1000
                });
                await state.waitForStart();
                if (state.stopRequested) {
                  return {
                    started: false,
                    reason: 'stopped_before_start',
                    queueDepth: 0,
                    droppedChunks: 0
                  };
                }
                state.bufferSessionStartedAt = performance.now();
                state.bufferLoopActive = true;
                const wakeWaiter = (payload) => {
                  const waiter = (state.bufferWaiters || []).shift();
                  if (waiter) {
                    waiter(payload);
                    return true;
                  }
                  return false;
                };
                state.enqueueBufferedChunk = function(payload) {
                  const enriched = Object.assign({}, payload || {}, {
                    queueDepth: (state.bufferQueue || []).length,
                    droppedChunks: state.droppedChunks || 0
                  });
                  if (!wakeWaiter(enriched)) {
                    if (state.bufferQueue.length >= state.maxQueueChunks) {
                      state.bufferQueue.shift();
                      state.droppedChunks += 1;
                    }
                    enriched.queueDepth = state.bufferQueue.length + 1;
                    enriched.droppedChunks = state.droppedChunks;
                    state.bufferQueue.push(enriched);
                  }
                  state.emitCallback(chunkCallbackName, enriched);
                };
                state.publishDebug({
                  chunkIndex: 1,
                  state: 'buffering',
                  status: 'started',
                  queueDepth: 0,
                  droppedChunks: 0,
                  maxSeconds: maxMilliseconds / 1000,
                  minSeconds: minMilliseconds / 1000,
                  pauseSeconds: silenceMilliseconds / 1000
                });
                (async () => {
                  try {
                    const numericMaxChunks = Number(maxChunks || 0);
                    const unlimited = !numericMaxChunks || numericMaxChunks < 1;
                    for (let index = 1; unlimited || index <= numericMaxChunks; index += 1) {
                      if (state.stopRequested) break;
                      const payload = await state.recordChunk(
                        maxMilliseconds,
                        minMilliseconds,
                        silenceMilliseconds,
                        silenceThreshold,
                        index
                      );
                      if (state.stopRequested) break;
                      state.enqueueBufferedChunk(payload);
                      state.publishDebug({
                        chunkIndex: index,
                        state: 'queued',
                        status: payload.stopReason || 'recorded',
                        durationSeconds: payload.durationSeconds,
                        size: payload.size,
                        queueDepth: state.bufferQueue.length,
                        droppedChunks: state.droppedChunks
                      });
                    }
                  } catch (error) {
                    state.bufferError = error && (error.message || String(error));
                    state.publishDebug({
                      state: 'error',
                      status: state.bufferError,
                      queueDepth: state.bufferQueue.length,
                      droppedChunks: state.droppedChunks
                    });
                    state.emitCallback(errorCallbackName, {error: state.bufferError});
                  } finally {
                    state.bufferLoopActive = false;
                    state.bufferDone = true;
                    while ((state.bufferWaiters || []).length) {
                      state.bufferWaiters.shift()(null);
                    }
                    state.publishDebug({
                      state: 'recording_done',
                      status: state.bufferError || 'done',
                      queueDepth: state.bufferQueue.length,
                      droppedChunks: state.droppedChunks
                    });
                    state.emitCallback(doneCallbackName, {
                      status: state.bufferError || 'done',
                      queueDepth: state.bufferQueue.length,
                      droppedChunks: state.droppedChunks
                    });
                  }
                })();
                return {started: true, queueDepth: 0, droppedChunks: 0};
              };
              window.liveCommerceMic.popNextBufferedChunk = function(waitMilliseconds) {
                const state = window.liveCommerceMic;
                if (state.bufferQueue && state.bufferQueue.length) {
                  const payload = state.bufferQueue.shift();
                  payload.queueDepth = state.bufferQueue.length;
                  payload.droppedChunks = state.droppedChunks || 0;
                  return payload;
                }
                if (state.bufferDone || !state.bufferLoopActive) return null;
                return new Promise(resolve => {
                  const done = (payload) => {
                    clearTimeout(timer);
                    resolve(payload);
                  };
                  const timer = setTimeout(() => {
                    state.bufferWaiters = (state.bufferWaiters || []).filter(waiter => waiter !== done);
                    resolve(null);
                  }, Math.max(50, Number(waitMilliseconds || 250)));
                  state.bufferWaiters = state.bufferWaiters || [];
                  state.bufferWaiters.push(done);
                });
              };
              window.liveCommerceMic.getBufferedStatus = function() {
                const state = window.liveCommerceMic;
                return {
                  active: Boolean(state.bufferLoopActive),
                  done: Boolean(state.bufferDone),
                  queueDepth: (state.bufferQueue || []).length,
                  droppedChunks: state.droppedChunks || 0,
                  error: state.bufferError || null
                };
              };
              window.liveCommerceMic.stop = function() {
                const state = window.liveCommerceMic;
                state.stopRequested = true;
                state.bufferDone = true;
                if (typeof state.resolveStart === 'function') {
                  state.resolveStart(false);
                  state.resolveStart = null;
                }
                if (state.activeRecorder && state.activeRecorder.state !== 'inactive') {
                  state.activeRecorder.stop();
                }
                if (state.stream) {
                  state.stream.getTracks().forEach(track => track.stop());
                  state.stream = null;
                }
                if (state.audioContext) {
                  state.audioContext.close();
                  state.audioContext = null;
                  state.source = null;
                  state.analyser = null;
                }
                return true;
              };
            })();
            """
        )
    )


def _record_colab_mic_chunk(
    chunk_index: int,
    max_seconds: float,
    min_seconds: float,
    pause_seconds: float,
    silence_threshold: float,
) -> Dict[str, Any]:
    from google.colab import output  # type: ignore

    max_milliseconds = max(500, int(float(max_seconds) * 1000))
    min_milliseconds = max(200, int(float(min_seconds) * 1000))
    silence_milliseconds = max(100, int(float(pause_seconds) * 1000))
    payload = output.eval_js(
        "window.liveCommerceMic.recordChunk("
        f"{max_milliseconds}, {min_milliseconds}, "
        f"{silence_milliseconds}, {float(silence_threshold)}, "
        f"{int(chunk_index)})"
    )
    if not isinstance(payload, dict) or not payload.get("dataUrl"):
        raise RuntimeError("browser mic recorder did not return audio data")
    return payload


def _start_colab_mic_buffer(config: LiveMicDemoConfig) -> Dict[str, Any]:
    from google.colab import output  # type: ignore

    max_milliseconds = max(500, int(float(config.chunk_seconds) * 1000))
    min_milliseconds = max(200, int(float(config.min_chunk_seconds) * 1000))
    silence_milliseconds = max(100, int(float(config.pause_seconds) * 1000))
    max_chunks_js = (
        "null" if config.max_chunks is None else str(max(1, int(config.max_chunks)))
    )
    result = output.eval_js(
        "window.liveCommerceMic.startBufferedRecording("
        f"{max_milliseconds}, {min_milliseconds}, "
        f"{silence_milliseconds}, {float(config.silence_threshold)}, "
        f"{max_chunks_js}, {max(1, int(config.max_queue_chunks))})"
    )
    if not isinstance(result, dict) or not result.get("started"):
        reason = result.get("reason") if isinstance(result, dict) else result
        raise RuntimeError(f"browser mic buffered recorder did not start: {reason}")
    return result


def _pop_colab_mic_buffered_chunk(wait_seconds: float) -> Optional[Dict[str, Any]]:
    from google.colab import output  # type: ignore

    wait_milliseconds = max(50, int(float(wait_seconds) * 1000))
    payload = output.eval_js(
        f"window.liveCommerceMic.popNextBufferedChunk({wait_milliseconds})"
    )
    if payload is None:
        return None
    if not isinstance(payload, dict) or not payload.get("dataUrl"):
        raise RuntimeError("browser mic buffered recorder returned invalid audio data")
    return payload


def _get_colab_mic_buffer_status() -> Dict[str, Any]:
    try:
        from google.colab import output  # type: ignore

        status = output.eval_js(
            "window.liveCommerceMic && window.liveCommerceMic.getBufferedStatus "
            "? window.liveCommerceMic.getBufferedStatus() : null"
        )
        return status if isinstance(status, dict) else {}
    except Exception:
        return {}


def install_colab_live_mic_debug_panel(
    config: Optional[LiveMicDemoConfig] = None,
    callbacks: Optional[Dict[str, str]] = None,
) -> None:
    """Display a Colab mic debug panel that updates while live chunks record."""
    _install_colab_mic_recorder()
    try:
        from IPython.display import HTML, Javascript, display  # type: ignore
        from google.colab import output  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Live mic debug panel requires Google Colab browser APIs.") from exc
    callback_config = None
    if config is not None and callbacks:
        callback_config = {
            "maxMilliseconds": max(500, int(float(config.chunk_seconds) * 1000)),
            "minMilliseconds": max(200, int(float(config.min_chunk_seconds) * 1000)),
            "silenceMilliseconds": max(100, int(float(config.pause_seconds) * 1000)),
            "silenceThreshold": float(config.silence_threshold),
            "maxChunks": None
            if config.max_chunks is None
            else max(1, int(config.max_chunks)),
            "maxQueueChunks": max(1, int(config.max_queue_chunks)),
        }

    display(
        HTML(
            """
            <div id="live-commerce-mic-debug" style="border:1px solid #d0d5dd;border-radius:8px;padding:14px;font-family:Arial,sans-serif;background:#fff;box-shadow:0 8px 24px rgba(16,24,40,.06);">
              <style>
                #live-commerce-mic-debug .lm-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr);gap:14px}
                #live-commerce-mic-debug .lm-video{height:360px;border-radius:8px;background:linear-gradient(145deg,#171717,#7f1d1d 55%,#0f766e);position:relative;overflow:hidden}
                #live-commerce-mic-debug .lm-video:after{content:"LIVE MIC";position:absolute;top:14px;left:14px;background:#fff;color:#111827;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:800}
                #live-commerce-mic-debug .lm-host{position:absolute;left:50%;top:54%;transform:translate(-50%,-50%);width:170px;height:220px}
                #live-commerce-mic-debug .lm-head{width:76px;height:76px;border-radius:50%;background:#fff7ed;margin:0 auto 8px;border:5px solid rgba(255,255,255,.42)}
                #live-commerce-mic-debug .lm-body{width:150px;height:130px;border-radius:42px 42px 10px 10px;background:#ef4444;margin:0 auto;box-shadow:0 18px 60px rgba(0,0,0,.28)}
                #live-commerce-mic-debug .lm-caption{position:absolute;left:18px;right:18px;bottom:18px;background:rgba(17,24,39,.88);color:#fff;border-radius:8px;padding:13px;font-size:16px;line-height:1.4}
                #live-commerce-mic-debug .lm-title{font-weight:800;font-size:17px;color:#111827;margin-bottom:8px}
                #live-commerce-mic-debug .lm-muted{color:#667085;font-size:12px;line-height:1.4}
                #live-commerce-mic-debug .lm-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:12px 0}
                #live-commerce-mic-debug .lm-metric{border:1px solid #eaecf0;border-radius:8px;padding:10px;background:#f9fafb}
                #live-commerce-mic-debug .lm-metric strong{display:block;color:#111827;margin-top:3px}
                #live-commerce-mic-debug .lm-buttons{display:flex;gap:8px;margin:10px 0}
                #live-commerce-mic-debug button{border:0;border-radius:8px;width:48px;height:42px;font-size:18px;font-weight:800;cursor:pointer;display:inline-flex;align-items:center;justify-content:center}
                #live-commerce-mic-debug #mic-debug-start{background:#b42318;color:#fff}
                #live-commerce-mic-debug button:disabled{opacity:.5;cursor:not-allowed}
                #live-commerce-mic-debug .lm-meter{height:10px;background:#f2f4f7;border-radius:999px;overflow:hidden;margin-top:8px}
                #live-commerce-mic-debug .lm-meter div{height:100%;width:0%;background:#12b76a}
                @media (max-width: 900px){#live-commerce-mic-debug .lm-grid{grid-template-columns:1fr}}
              </style>
              <div class="lm-grid">
                <div>
                  <div class="lm-video">
                    <div class="lm-host"><div class="lm-head"></div><div class="lm-body"></div></div>
                    <div class="lm-caption" id="mic-debug-caption">Press Start to begin sending mic audio to ASR.</div>
                  </div>
                  <div class="lm-meter"><div id="mic-debug-meter"></div></div>
                </div>
                <div>
                  <div class="lm-title">Live Microphone Stream</div>
                  <div class="lm-muted">Stop closes the input; queued chunks still finish processing.</div>
                  <div class="lm-buttons">
                    <button id="mic-debug-start" disabled title="Enable mic input" aria-label="Enable mic input">&#127908;</button>
                  </div>
                  <div class="lm-metrics">
                    <div class="lm-metric"><div class="lm-muted">Chunk</div><strong id="mic-debug-chunk">-</strong></div>
                    <div class="lm-metric"><div class="lm-muted">Detector</div><strong id="mic-debug-status">waiting</strong></div>
                    <div class="lm-metric"><div class="lm-muted">State</div><strong id="mic-debug-state">idle</strong></div>
                    <div class="lm-metric"><div class="lm-muted">Elapsed</div><strong id="mic-debug-elapsed">0.00s</strong></div>
                    <div class="lm-metric"><div class="lm-muted">Volume RMS</div><strong id="mic-debug-rms">0.0000</strong></div>
                    <div class="lm-metric"><div class="lm-muted">Threshold</div><strong id="mic-debug-threshold">-</strong></div>
                  </div>
                  <div id="mic-debug-silence" class="lm-muted">Silence For 0.00s</div>
                  <div id="mic-debug-detail" class="lm-muted" style="margin-top:8px;">Waiting for mic input.</div>
                </div>
              </div>
            </div>
            """
        )
    )
    display(
        Javascript(
            "window.liveCommerceMicPanelConfig = "
            + json.dumps(callback_config)
            + ";\nwindow.liveCommerceMicPanelCallbacks = "
            + json.dumps(callbacks or {})
            + ";\n"
            + """
            (() => {
              const state = window.liveCommerceMic = window.liveCommerceMic || {};
              const callbackConfig = window.liveCommerceMicPanelConfig || null;
              const callbackNames = window.liveCommerceMicPanelCallbacks || {};
              const latestRoot = () => {
                const roots = document.querySelectorAll('#live-commerce-mic-debug');
                return roots.length ? roots[roots.length - 1] : null;
              };
              const bindControls = () => {
                const root = latestRoot();
                if (!root) return;
                const get = id => root.querySelector(`#${id}`);
                const startButton = get('mic-debug-start');
                const setMicButton = () => {
                  const latest = state.latestDebug || {};
                  const debugState = latest.state || 'idle';
                  const active = debugState === 'recording'
                    || debugState === 'buffering'
                    || debugState === 'queued'
                    || debugState === 'transcribing';
                  const stopped = state.stopRequested || debugState === 'stopped';
                  const ready = debugState === 'ready';
                  if (!startButton) return;
                  startButton.innerHTML = active ? '&#10073;&#10073;' : '&#127908;';
                  startButton.title = active ? 'Disable mic input' : 'Enable mic input';
                  startButton.setAttribute('aria-label', startButton.title);
                  startButton.disabled = stopped || (!ready && !active);
                };
                if (startButton && !startButton.dataset.bound) {
                  startButton.dataset.bound = '1';
                  startButton.onclick = () => {
                    const latest = state.latestDebug || {};
                    const debugState = latest.state || 'idle';
                    const active = debugState === 'recording'
                      || debugState === 'buffering'
                      || debugState === 'queued'
                      || debugState === 'transcribing';
                    if (active) {
                      if (typeof state.stop === 'function') state.stop();
                      if (typeof state.emitCallback === 'function') {
                        state.emitCallback(callbackNames.stop, {stopped: true});
                      }
                      state.stopRequested = true;
                      state.publishDebug({state: 'stopped', status: 'input_closed'});
                      setMicButton();
                      return;
                    }
                    if (debugState !== 'ready') {
                      setMicButton();
                      return;
                    }
                    state.startRequested = true;
                    state.stopRequested = false;
                    if (callbackConfig && typeof state.emitCallback === 'function') {
                      state.emitCallback(callbackNames.start, {started: true});
                    }
                    if (typeof state.resolveStart === 'function') {
                      state.resolveStart(true);
                      state.resolveStart = null;
                    }
                    state.publishDebug({state: 'buffering', status: 'started'});
                    setMicButton();
                    if (callbackConfig && typeof state.startBufferedRecording === 'function') {
                      state.startBufferedRecording(
                        callbackConfig.maxMilliseconds,
                        callbackConfig.minMilliseconds,
                        callbackConfig.silenceMilliseconds,
                        callbackConfig.silenceThreshold,
                        callbackConfig.maxChunks,
                        callbackConfig.maxQueueChunks,
                        callbackNames.chunk,
                        callbackNames.done,
                        callbackNames.error
                      ).then(result => {
                        if (!result || !result.started) {
                          const reason = result && result.reason ? result.reason : 'not_started';
                          state.publishDebug({state: 'error', status: reason});
                          state.emitCallback(callbackNames.error, {error: reason});
                        }
                      }).catch(error => {
                        const message = error && (error.message || String(error));
                        state.publishDebug({state: 'error', status: message});
                        state.emitCallback(callbackNames.error, {error: message});
                      });
                    }
                  };
                }
                setMicButton();
              };
              state.updateDebug = function(payload) {
                const root = latestRoot();
                if (!root) return;
                const get = id => root.querySelector(`#${id}`);
                bindControls();
                const rms = Number(payload.rms || 0);
                const threshold = Number(payload.threshold || 0);
                const status = payload.status || 'waiting';
                const isSpeech = status === 'speech';
                const isSilence = status === 'silence';
                const meterPct = Math.max(0, Math.min(100, (rms / Math.max(threshold || 0.001, 0.001)) * 70));
                get('mic-debug-chunk').textContent = payload.chunkIndex ? String(payload.chunkIndex) : '-';
                get('mic-debug-status').textContent = status;
                get('mic-debug-status').style.color = isSpeech ? '#027a48' : (isSilence ? '#b42318' : '#344054');
                get('mic-debug-state').textContent = payload.state || 'idle';
                get('mic-debug-rms').textContent = rms.toFixed(4);
                get('mic-debug-threshold').textContent = threshold ? threshold.toFixed(4) : '-';
                get('mic-debug-elapsed').textContent = `${Number(payload.elapsedSeconds || payload.durationSeconds || 0).toFixed(2)}s`;
                get('mic-debug-silence').textContent = `Silence For ${Number(payload.silenceSeconds || 0).toFixed(2)}s`;
                get('mic-debug-meter').style.width = `${meterPct}%`;
                get('mic-debug-meter').style.background = isSpeech ? '#12b76a' : '#f04438';
                const caption = get('mic-debug-caption');
                if (caption) {
                  caption.textContent = payload.state === 'transcribing'
                    ? 'Transcribing latest mic chunk...'
                    : (isSpeech ? 'Receiving speech from microphone...' : (isSilence ? 'Listening for speech pause...' : (payload.status || 'Waiting for mic input.')));
                }
                const detailParts = [];
                if (payload.maxSeconds) detailParts.push(`max ${Number(payload.maxSeconds).toFixed(1)}s`);
                if (payload.minSeconds) detailParts.push(`min ${Number(payload.minSeconds).toFixed(1)}s`);
                if (payload.pauseSeconds) detailParts.push(`pause ${Number(payload.pauseSeconds).toFixed(1)}s`);
                if (payload.queueDepth !== undefined) detailParts.push(`queue ${payload.queueDepth}`);
                if (payload.droppedChunks !== undefined) detailParts.push(`dropped ${payload.droppedChunks}`);
                if (payload.size) detailParts.push(`${payload.size} bytes`);
                if (payload.captionCount !== undefined) detailParts.push(`${payload.captionCount} captions`);
                if (payload.actionCount !== undefined) detailParts.push(`${payload.actionCount} actions`);
                get('mic-debug-detail').textContent = detailParts.join(' | ') || 'Waiting for mic input.';
                bindControls();
              };
              state.publishDebug = state.publishDebug || function(payload) {
                state.latestDebug = Object.assign({}, state.latestDebug || {}, payload || {});
                if (typeof state.updateDebug === 'function') state.updateDebug(state.latestDebug);
              };
              bindControls();
              if (state.latestDebug) state.updateDebug(state.latestDebug);
            })();
            """
        )
    )


def _publish_colab_mic_debug(payload: Dict[str, Any]) -> None:
    try:
        from google.colab import output  # type: ignore

        encoded = json.dumps(payload)
        output.eval_js(
            "window.liveCommerceMic && window.liveCommerceMic.publishDebug && "
            f"window.liveCommerceMic.publishDebug({encoded})"
        )
    except Exception:
        pass


def _stop_colab_mic() -> None:
    try:
        from google.colab import output  # type: ignore

        output.eval_js("window.liveCommerceMic && window.liveCommerceMic.stop()")
    except Exception:
        pass


def _write_data_url(data_url: str, path: Path) -> None:
    _, encoded = data_url.split(",", 1)
    path.write_bytes(base64.b64decode(encoded))


def _display_live_state(
    chunk_index: int,
    config: LiveMicDemoConfig,
    segments: Sequence[CaptionSegment],
    actions: Sequence[Any],
    chunk_path: Optional[Path],
    queue_status: Optional[Dict[str, Any]] = None,
    state_label: str = "Transcribing",
    detail: Optional[str] = None,
    audio_stats: Optional[Dict[str, Any]] = None,
    display_key: Optional[str] = None,
) -> None:
    max_chunks_label = "open" if config.max_chunks is None else str(config.max_chunks)
    queue_status = queue_status or {}
    audio_stats = audio_stats or {}
    caption_rows = [
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text,
            "source": segment.source,
        }
        for segment in segments[-8:]
    ]
    action_rows = [
        {
            "time": round(action.timestamp, 2),
            "action": action.action_type,
            "title": action.display_payload.get("title", action.action_type),
            "skus": " + ".join(action.skus),
            "confidence": action.confidence,
        }
        for action in actions[-8:]
    ]
    metrics = {
        "Chunk": f"{chunk_index + 1}/{max_chunks_label}",
        "Captions": len(segments),
        "Actions": len(actions),
        "Queue": queue_status.get("queueDepth"),
        "Dropped": queue_status.get("droppedChunks") or 0,
        "Max RMS": audio_stats.get("maxRms"),
        "Mean RMS": audio_stats.get("meanRms"),
        "Speech": audio_stats.get("speechDetected"),
    }
    if chunk_path is not None:
        metrics["Chunk file"] = chunk_path.name
    display_stream_state_panel(
        title="Live Mic ASR + Commerce Actions",
        state_label=state_label,
        metrics=metrics,
        captions=caption_rows,
        actions=action_rows,
        detail=detail,
        accent="#a33b2f" if "Skip" not in state_label else "#b45309",
        display_key=display_key or "live_mic_stream_panel",
    )
