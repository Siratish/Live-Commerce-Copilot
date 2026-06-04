from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import base64
import json
import subprocess
import sys

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


class _OpenAIWhisperLiveASR:
    def __init__(self, model_name: str, language: str):
        if model_name in {"large", "large-v2", "large-v3", "turbo"}:
            print(f"Loading OpenAI Whisper {model_name}. This may take a while.")
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise CaptioningUnavailable(
                "openai-whisper is not installed. Set install_asr_deps=True or run pip install -r requirements-asr.txt."
            ) from exc

        self.model_name = model_name
        self.language = language
        self.model = whisper.load_model(model_name)

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
) -> Dict[str, Any]:
    """Record browser mic chunks in Colab, transcribe them, and emit actions."""
    _install_asr_dependencies(config.install_asr_deps)
    if config.show_debug_panel:
        install_colab_live_mic_debug_panel()
    else:
        _install_colab_mic_recorder()
    asr = _build_live_asr(config)
    catalog = load_product_catalog(config.catalog_path)
    promotions = load_promotions(config.promotions_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = config.output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    live_segments: List[CaptionSegment] = []
    live_actions = []
    elapsed_seconds = 0.0
    processed_chunks = 0
    buffer_status: Dict[str, Any] = {}

    try:
        if config.continuous_recording:
            _start_colab_mic_buffer(config)
            while not _chunk_limit_reached(processed_chunks, config.max_chunks):
                payload = _pop_colab_mic_buffered_chunk(config.poll_interval_seconds)
                if payload is None:
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
                    continue

                processed_chunks += 1
                chunk_number = int(payload.get("chunkIndex") or processed_chunks)
                (
                    live_segments,
                    live_actions,
                    elapsed_seconds,
                    chunk_path,
                ) = _process_live_mic_payload(
                    payload=payload,
                    chunk_number=chunk_number,
                    config=config,
                    chunk_dir=chunk_dir,
                    asr=asr,
                    catalog=catalog,
                    promotions=promotions,
                    decision_provider=decision_provider,
                    live_segments=live_segments,
                    elapsed_seconds=elapsed_seconds,
                )
                _publish_colab_mic_debug(
                    {
                        "chunkIndex": chunk_number,
                        "state": "completed",
                        "status": "asr_done",
                        "durationSeconds": payload.get("durationSeconds"),
                        "captionCount": len(live_segments),
                        "actionCount": len(live_actions),
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
                ) = _process_live_mic_payload(
                    payload=payload,
                    chunk_number=chunk_index + 1,
                    config=config,
                    chunk_dir=chunk_dir,
                    asr=asr,
                    catalog=catalog,
                    promotions=promotions,
                    decision_provider=decision_provider,
                    live_segments=live_segments,
                    elapsed_seconds=elapsed_seconds,
                )
                _publish_colab_mic_debug(
                    {
                        "chunkIndex": chunk_index + 1,
                        "state": "completed",
                        "status": "asr_done",
                        "durationSeconds": payload.get("durationSeconds"),
                        "captionCount": len(live_segments),
                        "actionCount": len(live_actions),
                    }
                )
    finally:
        _stop_colab_mic()

    if config.continuous_recording:
        buffer_status = _get_colab_mic_buffer_status() or buffer_status

    return {
        "caption_count": len(live_segments),
        "action_count": len(live_actions),
        "processed_chunks": processed_chunks,
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
) -> tuple[List[CaptionSegment], List[Any], float, Path]:
    duration_seconds = float(payload.get("durationSeconds") or config.chunk_seconds)
    chunk_path = chunk_dir / f"mic_chunk_{max(0, chunk_number - 1):03d}.webm"
    _write_data_url(payload["dataUrl"], chunk_path)

    offset_raw = payload.get("sessionOffsetSeconds")
    offset_seconds = (
        elapsed_seconds if offset_raw is None else max(0.0, float(offset_raw))
    )
    _publish_colab_mic_debug(
        {
            "chunkIndex": chunk_number,
            "state": "transcribing",
            "status": "dequeued" if config.continuous_recording else "recorded",
            "durationSeconds": duration_seconds,
            "size": payload.get("size"),
            "queueDepth": payload.get("queueDepth"),
            "droppedChunks": payload.get("droppedChunks"),
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
    )
    return repaired_segments, live_actions, next_elapsed_seconds, chunk_path


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
                    if (!isSilence) lastSpeechAt = now;
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
                      size: blob.size
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
                maxQueueChunks
              ) {
                const state = window.liveCommerceMic;
                if (state.bufferLoopActive) {
                  return {
                    started: false,
                    reason: 'already_active',
                    queueDepth: (state.bufferQueue || []).length
                  };
                }
                state.stopRequested = false;
                state.startRequested = false;
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
                  if (wakeWaiter(enriched)) return;
                  if (state.bufferQueue.length >= state.maxQueueChunks) {
                    state.bufferQueue.shift();
                    state.droppedChunks += 1;
                  }
                  enriched.queueDepth = state.bufferQueue.length + 1;
                  enriched.droppedChunks = state.droppedChunks;
                  state.bufferQueue.push(enriched);
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


def install_colab_live_mic_debug_panel() -> None:
    """Display a Colab mic debug panel that updates while live chunks record."""
    _install_colab_mic_recorder()
    try:
        from IPython.display import HTML, Javascript, display  # type: ignore
        from google.colab import output  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Live mic debug panel requires Google Colab browser APIs.") from exc

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
                #live-commerce-mic-debug button{border:0;border-radius:8px;padding:10px 13px;font-weight:800;cursor:pointer}
                #live-commerce-mic-debug #mic-debug-start{background:#b42318;color:#fff}
                #live-commerce-mic-debug #mic-debug-stop{background:#f2f4f7;color:#344054}
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
                    <button id="mic-debug-start">Start live mic</button>
                    <button id="mic-debug-stop">Stop input</button>
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
            """
            (() => {
              const state = window.liveCommerceMic = window.liveCommerceMic || {};
              state.updateDebug = function(payload) {
                const roots = document.querySelectorAll('#live-commerce-mic-debug');
                const root = roots.length ? roots[roots.length - 1] : null;
                if (!root) return;
                const get = id => root.querySelector(`#${id}`);
                const rms = Number(payload.rms || 0);
                const threshold = Number(payload.threshold || 0);
                const status = payload.status || 'waiting';
                const isSpeech = status === 'speech';
                const isSilence = status === 'silence';
                const startButton = get('mic-debug-start');
                const stopButton = get('mic-debug-stop');
                if (startButton && !startButton.dataset.bound) {
                  startButton.dataset.bound = '1';
                  startButton.onclick = () => {
                    state.startRequested = true;
                    if (typeof state.resolveStart === 'function') {
                      state.resolveStart(true);
                      state.resolveStart = null;
                    }
                    startButton.textContent = 'Recording...';
                    startButton.disabled = true;
                    if (stopButton) stopButton.disabled = false;
                  };
                }
                if (stopButton && !stopButton.dataset.bound) {
                  stopButton.dataset.bound = '1';
                  stopButton.onclick = () => {
                    if (typeof state.stop === 'function') state.stop();
                    stopButton.textContent = 'Stopping...';
                    stopButton.disabled = true;
                    if (startButton) startButton.disabled = true;
                  };
                }
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
              };
              state.publishDebug = state.publishDebug || function(payload) {
                state.latestDebug = Object.assign({}, state.latestDebug || {}, payload || {});
                if (typeof state.updateDebug === 'function') state.updateDebug(state.latestDebug);
              };
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
    chunk_path: Path,
    queue_status: Optional[Dict[str, Any]] = None,
) -> None:
    from IPython.display import display  # type: ignore

    max_chunks_label = "open" if config.max_chunks is None else str(config.max_chunks)
    print(
        f"Live mic utterance {chunk_index + 1}/{max_chunks_label} saved to {chunk_path}"
    )
    print(f"Captions: {len(segments)} | Actions: {len(actions)}")
    if queue_status and queue_status.get("queueDepth") is not None:
        print(
            "Buffered chunks waiting: "
            f"{queue_status.get('queueDepth')} | Dropped chunks: "
            f"{queue_status.get('droppedChunks') or 0}"
        )

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
            "skus": " + ".join(action.skus),
            "confidence": action.confidence,
        }
        for action in actions[-8:]
    ]

    try:
        import pandas as pd  # type: ignore

        display(pd.DataFrame(caption_rows))
        display(pd.DataFrame(action_rows))
    except Exception:
        display({"captions": caption_rows, "actions": action_rows})
