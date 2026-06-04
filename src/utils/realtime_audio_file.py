from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import base64
import json
import mimetypes
import subprocess
import sys
import threading
import time
import uuid
import wave

from src.ai.captioning import (
    AudioWindow,
    iter_audio_windows,
    normalize_asr_provider,
    repair_caption_timestamps,
    resolve_asr_model_id,
)
from src.ai.commerce_actions import generate_commerce_actions, save_commerce_actions
from src.ai.decision import CommerceDecisionProvider
from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import CaptionResult, CaptionSegment
from src.ai.captioning import save_caption_json


@dataclass(frozen=True)
class RealtimeAudioFileDemoConfig:
    audio_path: Path = Path("data/demo/audio/1.mp3")
    chunk_seconds: float = 4.0
    dynamic_chunking: bool = True
    min_chunk_seconds: float = 1.0
    pause_seconds: float = 0.7
    silence_threshold: float = 0.012
    frame_seconds: float = 0.1
    max_chunks: Optional[int] = None
    language: str = "th"
    asr_provider: str = "openai_whisper"
    asr_model: str = "base"
    asr_max_new_tokens: int = 256
    install_asr_deps: bool = False
    playback_poll_seconds: float = 0.25
    output_dir: Path = Path("outputs/realtime_audio_file")
    catalog_path: Path = Path("data/demo/product_catalog.csv")
    promotions_path: Path = Path("data/demo/promotions.csv")


class WallClockPlaybackGate:
    """Fallback playback gate for non-Colab runtimes."""

    def __init__(self) -> None:
        self.started_at: Optional[float] = None
        self.latest_status: Dict[str, Any] = {"currentTime": 0.0}

    def install(self) -> None:
        return

    def wait_for_start(self) -> Dict[str, Any]:
        self.started_at = time.monotonic()
        self.latest_status = {"currentTime": 0.0, "mode": "wall_clock"}
        return self.latest_status

    def wait_until(self, target_seconds: float) -> Dict[str, Any]:
        if self.started_at is None:
            self.wait_for_start()
        assert self.started_at is not None
        delay = float(target_seconds) - (time.monotonic() - self.started_at)
        if delay > 0:
            time.sleep(delay)
        self.latest_status = {
            "currentTime": max(float(target_seconds), time.monotonic() - self.started_at),
            "mode": "wall_clock",
        }
        return self.latest_status

    def status(self) -> Dict[str, Any]:
        if self.started_at is None:
            return self.latest_status
        self.latest_status = {
            "currentTime": time.monotonic() - self.started_at,
            "mode": "wall_clock",
        }
        return self.latest_status

    def publish(self, payload: Dict[str, Any]) -> None:
        return


class BrowserAudioPlaybackGate:
    def __init__(
        self,
        audio_path: Path,
        windows: Sequence[AudioWindow],
        poll_seconds: float = 0.25,
        widget_id: Optional[str] = None,
    ) -> None:
        self.audio_path = audio_path
        self.windows = list(windows)
        self.poll_seconds = poll_seconds
        self.widget_id = widget_id or f"realtime-audio-file-{uuid.uuid4().hex}"
        self._installed = False
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

    def install(self) -> None:
        if self._installed:
            return
        try:
            from IPython.display import HTML, Javascript, display  # type: ignore
            from google.colab import output  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Browser audio playback gating requires Google Colab.") from exc

        display(HTML(_build_realtime_audio_file_html(self.audio_path, self.windows, self.widget_id)))
        script = _build_realtime_audio_file_js(self.widget_id, self.poll_seconds)
        try:
            output.eval_js(script)
        except Exception:
            display(Javascript(script))
        self._installed = True

    def wait_for_start(self) -> Dict[str, Any]:
        from google.colab import output  # type: ignore

        status = output.eval_js(
            _build_realtime_audio_file_api_call_js(self.widget_id, "waitForStart")
        )
        status = status if isinstance(status, dict) else {}
        current_time = float(status.get("currentTime") or 0.0)
        self._started_at = time.monotonic() - max(0.0, current_time)
        if status.get("stopped"):
            self._mark_stopped()
        return {**self._python_status(), **status}

    def wait_until(self, target_seconds: float) -> Dict[str, Any]:
        if self._started_at is None:
            self.wait_for_start()
        while not self._stop_event.is_set():
            status = self._browser_status()
            if status.get("stopped"):
                self._mark_stopped()
                return self.status()
            browser_time = float(status.get("currentTime") or 0.0)
            current_time = (
                browser_time
                if status.get("paused")
                else max(browser_time, self._python_current_time())
            )
            if current_time >= float(target_seconds):
                return {**self._python_status(), **status, "currentTime": current_time}
            time.sleep(
                min(
                    self.poll_seconds,
                    max(0.01, float(target_seconds) - current_time),
                )
            )
        return self.status()

    def status(self) -> Dict[str, Any]:
        python_status = self._python_status()
        browser_status = self._browser_status()
        browser_time = float(browser_status.get("currentTime") or 0.0)
        current_time = (
            browser_time
            if browser_status.get("paused")
            else max(float(python_status.get("currentTime") or 0.0), browser_time)
        )
        return {
            **browser_status,
            **python_status,
            "currentTime": current_time,
            "paused": bool(
                browser_status.get("paused") or python_status.get("paused")
            ),
            "stopped": bool(
                python_status.get("stopped") or browser_status.get("stopped")
            ),
        }

    def _browser_status(self) -> Dict[str, Any]:
        try:
            from google.colab import output  # type: ignore

            status = output.eval_js(
                _build_realtime_audio_file_api_call_js(self.widget_id, "getStatus")
            )
            return status if isinstance(status, dict) else {}
        except Exception:
            return {}

    def _python_current_time(self) -> float:
        if self._started_at is None:
            return 0.0
        duration = max((window.end for window in self.windows), default=0.0)
        now = self._stopped_at if self._stopped_at is not None else time.monotonic()
        current_time = max(0.0, now - self._started_at)
        return min(current_time, duration) if duration else current_time

    def _python_status(self) -> Dict[str, Any]:
        duration = max((window.end for window in self.windows), default=0.0)
        current_time = self._python_current_time()
        return {
            "currentTime": current_time,
            "duration": duration,
            "paused": self._started_at is None or self._stop_event.is_set(),
            "ended": bool(duration and current_time >= duration),
            "stopped": self._stop_event.is_set(),
            "mode": "python_clock",
        }

    def stop(self) -> None:
        self._mark_stopped()
        try:
            from google.colab import output  # type: ignore

            output.eval_js(
                "window.realtimeAudioFileDemo && window.realtimeAudioFileDemo"
                f"[{json.dumps(self.widget_id)}] && window.realtimeAudioFileDemo"
                f"[{json.dumps(self.widget_id)}].stop()"
            )
        except Exception:
            pass

    def _mark_stopped(self) -> None:
        if self._stopped_at is None:
            self._stopped_at = time.monotonic()
        self._stop_event.set()
        self._start_event.set()

    def publish(self, payload: Dict[str, Any]) -> None:
        try:
            from google.colab import output  # type: ignore

            output.eval_js(
                "window.realtimeAudioFileDemo"
                f"[{json.dumps(self.widget_id)}].publish({json.dumps(payload)})"
            )
        except Exception:
            pass


def run_realtime_audio_file_demo(
    config: RealtimeAudioFileDemoConfig = RealtimeAudioFileDemoConfig(),
    decision_provider: Optional[CommerceDecisionProvider] = None,
    playback_gate: Optional[Any] = None,
    asr: Optional[Any] = None,
    windows: Optional[Sequence[AudioWindow]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Replay an audio file as a live stream and process chunks after playback passes them."""
    if config.install_asr_deps:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-asr.txt"]
        )

    if not config.audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {config.audio_path}")

    audio_windows = list(windows) if windows is not None else _load_audio_windows(config)
    if config.max_chunks is not None:
        audio_windows = audio_windows[: max(0, int(config.max_chunks))]
    if not audio_windows:
        raise RuntimeError("no audio windows were produced for the realtime file demo")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = config.output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    captions_path = config.output_dir / "realtime_file_captions.json"
    actions_path = config.output_dir / "realtime_file_actions.json"
    save_caption_json(
        CaptionResult(language=config.language, segments=[], duration_seconds=0.0),
        captions_path,
    )
    save_commerce_actions([], actions_path)

    gate = playback_gate or _default_playback_gate(
        config.audio_path,
        audio_windows,
        config.playback_poll_seconds,
    )
    gate.install()
    if asr is None:
        gate.publish(
            {
                "state": "loading_asr",
                "status": f"loading_{config.asr_provider}_{config.asr_model}",
                "totalChunks": len(audio_windows),
                "processedChunks": 0,
                "queueDepth": 0,
            }
        )
        asr_engine = _build_file_stream_asr(config)
    else:
        asr_engine = asr
    gate.publish(
        {
            "state": "asr_ready",
            "status": "press_start",
            "totalChunks": len(audio_windows),
            "processedChunks": 0,
            "queueDepth": 0,
        }
    )
    start_status = gate.wait_for_start()
    if start_status.get("stopped") or (cancel_event and cancel_event.is_set()):
        return {
            "caption_count": 0,
            "action_count": 0,
            "processed_chunks": 0,
            "skipped_chunks": 0,
            "max_backlog_chunks": 0,
            "stopped": True,
            "audio_path": str(config.audio_path),
            "captions_json": str(captions_path),
            "actions_json": str(actions_path),
            "chunk_dir": str(chunk_dir),
        }
    catalog = load_product_catalog(config.catalog_path)
    promotions = load_promotions(config.promotions_path)

    live_segments: List[CaptionSegment] = []
    live_actions = []
    processed_chunks = 0
    skipped_chunks = 0
    max_backlog = 0
    stopped_by_user = False

    for index, window in enumerate(audio_windows, start=1):
        if cancel_event and cancel_event.is_set():
            stopped_by_user = True
            break
        gate.publish(
            {
                "state": "waiting_for_playback",
                "status": "chunk_not_released",
                "chunkIndex": index,
                "chunkEnd": window.end,
                "totalChunks": len(audio_windows),
                "processedChunks": processed_chunks,
            }
        )
        status = gate.wait_until(window.end)
        current_time = float(status.get("currentTime") or window.end)
        if (
            status.get("stopped")
            or (cancel_event and cancel_event.is_set())
        ) and current_time < window.end:
            stopped_by_user = True
            gate.publish(
                {
                    "state": "stopped",
                    "status": "input_closed",
                    "chunkIndex": index,
                    "totalChunks": len(audio_windows),
                    "processedChunks": processed_chunks,
                }
            )
            break
        released = sum(1 for item in audio_windows if item.end <= current_time)
        backlog = max(0, released - processed_chunks - 1)
        max_backlog = max(max_backlog, backlog)

        audio_stats = _audio_window_stats(window.samples)
        if _is_silent_audio_chunk(audio_stats, config.silence_threshold):
            processed_chunks += 1
            skipped_chunks += 1
            caption_result = CaptionResult(
                language=config.language,
                segments=live_segments,
                duration_seconds=max(window.end, current_time),
            )
            save_caption_json(caption_result, captions_path)
            save_commerce_actions(live_actions, actions_path)
            gate.publish(
                {
                    "state": "skipped_silence",
                    "status": "below_silence_threshold",
                    "chunkIndex": index,
                    "chunkStart": window.start,
                    "chunkEnd": window.end,
                    "playbackTime": current_time,
                    "queueDepth": backlog,
                    "processedChunks": processed_chunks,
                    "skippedChunks": skipped_chunks,
                    "totalChunks": len(audio_windows),
                    "captionCount": len(live_segments),
                    "actionCount": len(live_actions),
                    "rms": audio_stats["rms"],
                    "peak": audio_stats["peak"],
                    **_realtime_file_result_payload(live_segments, live_actions),
                }
            )
            continue

        chunk_path = chunk_dir / f"file_chunk_{index - 1:03d}.wav"
        write_audio_window_wav(window, chunk_path)
        gate.publish(
            {
                "state": "transcribing",
                "status": "released_after_playback",
                "chunkIndex": index,
                "chunkStart": window.start,
                "chunkEnd": window.end,
                "playbackTime": current_time,
                "queueDepth": backlog,
                "processedChunks": processed_chunks,
                "totalChunks": len(audio_windows),
            }
        )

        new_segments = asr_engine.transcribe_chunk(
            chunk_path=chunk_path,
            offset_seconds=window.start,
            fallback_duration_seconds=window.end - window.start,
        )
        if cancel_event and cancel_event.is_set():
            stopped_by_user = True
            break
        processed_chunks += 1
        live_segments = repair_caption_timestamps([*live_segments, *new_segments])
        caption_result = CaptionResult(
            language=config.language,
            segments=live_segments,
            duration_seconds=max(window.end, current_time),
        )
        live_actions = generate_commerce_actions(
            caption_result,
            catalog,
            promotions,
            decision_provider=decision_provider,
        )
        save_caption_json(caption_result, captions_path)
        save_commerce_actions(live_actions, actions_path)
        gate.publish(
            {
                "state": "completed",
                "status": "asr_action_done",
                "chunkIndex": index,
                "chunkStart": window.start,
                "chunkEnd": window.end,
                "playbackTime": current_time,
                "queueDepth": backlog,
                "processedChunks": processed_chunks,
                "skippedChunks": skipped_chunks,
                "totalChunks": len(audio_windows),
                "captionCount": len(live_segments),
                "actionCount": len(live_actions),
                **_realtime_file_result_payload(live_segments, live_actions),
            }
        )

    return {
        "caption_count": len(live_segments),
        "action_count": len(live_actions),
        "processed_chunks": processed_chunks,
        "skipped_chunks": skipped_chunks,
        "max_backlog_chunks": max_backlog,
        "stopped": stopped_by_user,
        "audio_path": str(config.audio_path),
        "captions_json": str(captions_path),
        "actions_json": str(actions_path),
        "chunk_dir": str(chunk_dir),
    }


def _realtime_file_result_payload(
    segments: Sequence[CaptionSegment],
    actions: Sequence[Any],
) -> Dict[str, Any]:
    return {
        "latestCaption": segments[-1].text if segments else "",
        "captionRows": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text,
            }
            for segment in segments[-12:]
        ],
        "actionRows": [
            {
                "time": round(action.timestamp, 2),
                "action": action.action_type,
                "title": action.display_payload.get("title", action.action_type),
                "skus": " + ".join(action.skus),
                "displayPayload": action.display_payload,
            }
            for action in actions[-6:]
        ],
    }


def write_audio_window_wav(window: AudioWindow, path: Path) -> None:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("writing realtime audio chunks requires numpy") from exc

    values = np.asarray(window.samples, dtype=np.float32)
    clipped = np.clip(values, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(window.sample_rate))
        handle.writeframes(pcm.tobytes())


def _audio_window_stats(samples: Any) -> Dict[str, float]:
    try:
        import numpy as np  # type: ignore

        values = np.asarray(samples, dtype=np.float32)
        if values.size == 0:
            return {"rms": 0.0, "peak": 0.0}
        return {
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "peak": float(np.max(np.abs(values))),
        }
    except Exception:
        values = [float(value) for value in samples] if samples is not None else []
        if not values:
            return {"rms": 0.0, "peak": 0.0}
        square_mean = sum(value * value for value in values) / len(values)
        return {
            "rms": square_mean**0.5,
            "peak": max(abs(value) for value in values),
        }


def _is_silent_audio_chunk(stats: Dict[str, float], silence_threshold: float) -> bool:
    threshold = max(0.0, float(silence_threshold))
    if threshold <= 0:
        return False
    return (
        float(stats.get("peak") or 0.0) < threshold
        and float(stats.get("rms") or 0.0) < threshold * 0.5
    )


def _load_audio_windows(config: RealtimeAudioFileDemoConfig) -> List[AudioWindow]:
    return list(
        iter_audio_windows(
            config.audio_path,
            int(config.chunk_seconds),
            dynamic_chunking=config.dynamic_chunking,
            min_chunk_seconds=config.min_chunk_seconds,
            pause_seconds=config.pause_seconds,
            silence_threshold=config.silence_threshold,
            frame_seconds=config.frame_seconds,
        )
    )


def _default_playback_gate(
    audio_path: Path,
    windows: Sequence[AudioWindow],
    poll_seconds: float,
) -> Any:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except ImportError:
        return WallClockPlaybackGate()
    return BrowserAudioPlaybackGate(
        audio_path=audio_path,
        windows=windows,
        poll_seconds=poll_seconds,
    )


def _build_file_stream_asr(config: RealtimeAudioFileDemoConfig) -> Any:
    from src.utils.live_mic import _OpenAIWhisperLiveASR, _TyphoonWhisperLiveASR

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
    raise ValueError(f"unsupported file-stream ASR provider: {config.asr_provider}")


def _audio_data_uri(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_realtime_audio_file_html(
    audio_path: Path,
    windows: Sequence[AudioWindow],
    widget_id: str,
) -> str:
    duration = max((window.end for window in windows), default=0.0)
    return f"""
<div id="{widget_id}" data-duration="{duration:.6f}" style="border:1px solid #d0d5dd;border-radius:8px;padding:14px;font-family:Arial,sans-serif;background:#fff;box-shadow:0 8px 24px rgba(16,24,40,.06);">
  <style>
    #{widget_id} .rt-grid{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(240px,.75fr);gap:14px}}
    #{widget_id} .rt-video{{height:360px;border-radius:8px;background:linear-gradient(145deg,#171717,#7f1d1d 55%,#0f766e);position:relative;overflow:hidden}}
    #{widget_id} .rt-video:after{{content:"LIVE";position:absolute;top:14px;left:14px;background:#fff;color:#111827;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:800}}
    #{widget_id} .rt-host{{position:absolute;left:50%;top:54%;transform:translate(-50%,-50%);width:170px;height:220px}}
    #{widget_id} .rt-head{{width:76px;height:76px;border-radius:50%;background:#fff7ed;margin:0 auto 8px;border:5px solid rgba(255,255,255,.42)}}
    #{widget_id} .rt-body{{width:150px;height:130px;border-radius:42px 42px 10px 10px;background:#ef4444;margin:0 auto;box-shadow:0 18px 60px rgba(0,0,0,.28)}}
    #{widget_id} .rt-caption{{position:absolute;left:18px;right:18px;bottom:18px;background:rgba(17,24,39,.88);color:#fff;border-radius:8px;padding:13px;font-size:16px;line-height:1.4}}
    #{widget_id} .rt-panel-title{{font-weight:800;font-size:24px;line-height:1.16;color:#111827;margin-bottom:10px}}
    #{widget_id} .rt-muted{{color:#667085;font-size:12px;line-height:1.4}}
    #{widget_id} .rt-bottom{{display:grid;grid-template-columns:48px minmax(0,1fr);gap:10px;align-items:center;margin-top:10px}}
    #{widget_id} .rt-current-action{{margin-top:10px}}
    #{widget_id} button{{border:0;border-radius:8px;padding:10px 13px;font-weight:800;cursor:pointer}}
    #{widget_id} [data-role=start]{{background:#b42318;color:#fff;width:48px;height:42px;display:inline-flex;align-items:center;justify-content:center;font-size:18px}}
    #{widget_id} button:disabled{{opacity:.5;cursor:not-allowed}}
    #{widget_id} .rt-progress{{height:10px;background:#f2f4f7;border-radius:999px;overflow:hidden}}
    #{widget_id} .rt-progress div{{height:100%;width:0%;background:#e51b23}}
    #{widget_id} .rt-stats{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}}
    #{widget_id} .rt-stats span{{background:#f2f4f7;border-radius:999px;padding:5px 10px;font-size:13px;color:#344054}}
    #{widget_id} .rt-actions{{display:flex;flex-direction:column;gap:10px}}
    #{widget_id} .rt-action-card{{border:1px solid #d0d5dd;border-radius:8px;padding:12px;background:#fff;color:#111827;box-shadow:0 2px 8px rgba(16,24,40,.04);display:flex;flex-direction:column;gap:7px}}
    #{widget_id} .rt-action-pin{{border-color:#bfdbfe;background:#eff6ff}}
    #{widget_id} .rt-action-promo{{border-color:#fed7aa;background:#fff7ed}}
    #{widget_id} .rt-action-bundle{{border-color:#bbf7d0;background:#f0fdf4}}
    #{widget_id} .rt-action-countdown{{border-color:#fecaca;background:#fef2f2}}
    #{widget_id} .rt-action-top{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
    #{widget_id} .rt-action-top strong{{font-size:14px;color:#111827}}
    #{widget_id} .rt-action-title{{font-size:16px;font-weight:800;margin-bottom:4px}}
    #{widget_id} .rt-action-meta{{font-size:14px;color:#111827;line-height:1.35}}
    #{widget_id} .rt-action-time{{font-size:12px;color:#667085;margin-top:6px}}
    #{widget_id} .rt-product-row{{display:flex;align-items:center;gap:10px;margin-top:2px}}
    #{widget_id} .rt-product-thumb{{width:52px;height:52px;border-radius:8px;background:#fee4e2;color:#9e2a23;display:flex;align-items:center;justify-content:center;font-weight:900;flex:0 0 auto;box-shadow:inset 0 0 0 1px rgba(158,42,35,.08)}}
    #{widget_id} .rt-product-info{{min-width:0;display:flex;flex-direction:column;gap:2px}}
    #{widget_id} .rt-product-name{{font-size:15px;font-weight:900;color:#111827;line-height:1.2}}
    #{widget_id} .rt-product-meta{{font-size:13px;color:#667085;line-height:1.3}}
    #{widget_id} .rt-product-price{{font-size:14px;font-weight:900;color:#b42318}}
    #{widget_id} .rt-promo-badge{{align-self:flex-start;border-radius:8px;background:#b42318;color:#fff;padding:7px 10px;font-size:18px;font-weight:900;letter-spacing:.04em}}
    #{widget_id} .rt-countdown-time{{font-size:30px;line-height:1;font-weight:900;color:#b42318}}
    #{widget_id} .rt-bundle-row{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
    #{widget_id} .rt-bundle-plus{{font-size:18px;font-weight:900;color:#166534}}
    #{widget_id} .rt-empty{{border:1px solid #d0d5dd;border-radius:8px;padding:12px;color:#667085;background:#fff}}
    @media (max-width: 900px){{#{widget_id} .rt-grid{{grid-template-columns:1fr}}}}
  </style>
  <audio preload="metadata" src="{_audio_data_uri(audio_path)}" style="display:none;"></audio>
  <div class="rt-grid">
    <div>
      <div class="rt-video">
        <div class="rt-host"><div class="rt-head"></div><div class="rt-body"></div></div>
        <div class="rt-caption" data-role="caption">Ready to start. Press play to stream audio into ASR.</div>
      </div>
      <div class="rt-current-action" data-role="current-action"><div class="rt-empty">Current action will appear here.</div></div>
      <div class="rt-bottom">
        <button data-role="start" disabled title="Start stream" aria-label="Start stream">&#9654;</button>
        <div class="rt-progress"><div data-role="progress"></div></div>
      </div>
    </div>
    <div>
      <div class="rt-panel-title">Action history</div>
      <div class="rt-stats">
        <span data-role="caption-count">0 captions</span>
        <span data-role="action-count">0 actions</span>
      </div>
      <div data-role="actions" class="rt-actions"><div class="rt-empty">No actions emitted yet.</div></div>
    </div>
  </div>
</div>
"""


def _build_realtime_audio_file_js(
    widget_id: str,
    poll_seconds: float,
    start_callback: Optional[str] = None,
    stop_callback: Optional[str] = None,
) -> str:
    poll_milliseconds = max(50, int(float(poll_seconds) * 1000))
    return f"""
(() => {{
  const widgetId = {json.dumps(widget_id)};
  const pollMilliseconds = {poll_milliseconds};
  const startCallback = {json.dumps(start_callback)};
  const stopCallback = {json.dumps(stop_callback)};
  const invokeCallback = (name, payload) => {{
    if (!name || !window.google || !google.colab || !google.colab.kernel) return;
    try {{
      google.colab.kernel.invokeFunction(name, [payload || {{}}], {{}});
    }} catch (error) {{
      console.warn("Live callback failed", error);
    }}
  }};
  const install = (attempt = 0) => {{
    const root = document.getElementById(widgetId);
    if (!root) {{
      if (attempt < 100) window.setTimeout(() => install(attempt + 1), 100);
      return;
    }}
    if (root.dataset.realtimeAudioBound === "1") return;
    root.dataset.realtimeAudioBound = "1";

    const audio = root.querySelector("audio");
    const startButton = root.querySelector('[data-role="start"]');
    const get = role => root.querySelector(`[data-role="${{role}}"]`);
    const apiRoot = window.realtimeAudioFileDemo = window.realtimeAudioFileDemo || {{}};
    let stopRequested = false;
    let playing = false;
    let baseSeconds = 0;
    let startedAtMs = 0;
    let ticker = null;
    let pendingStartResolve = null;
    let readyToStart = false;
    let latestCaption = "";
    let captionQueue = [];
    let captionDisplayActive = false;
    let captionTimer = null;
    let seenCaptionKeys = new Set();
    let lastActionRows = [];

    const duration = () => {{
      const audioDuration = audio && audio.duration && Number.isFinite(audio.duration) ? audio.duration : 0;
      const configuredDuration = Number(root.dataset.duration || 0);
      return audioDuration || configuredDuration || 0;
    }};
    const streamTime = () => {{
      const total = duration();
      const elapsed = playing ? baseSeconds + ((performance.now() - startedAtMs) / 1000) : baseSeconds;
      return total ? Math.min(total, Math.max(0, elapsed)) : Math.max(0, elapsed);
    }};
    const setDetail = text => {{
      const detail = get("detail");
      if (detail) detail.textContent = text;
    }};
    const setState = text => {{
      const state = get("state");
      if (state) state.textContent = text;
    }};
    const setCaptionStatus = text => {{
      if (!latestCaption && !captionDisplayActive) get("caption").textContent = text;
    }};
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }}[ch]));
    const humanActionTitle = value => String(value || "Action")
      .toLowerCase()
      .replace(/_/g, " ")
      .replace(/\\b\\w/g, match => match.toUpperCase());
    const initials = value => String(value || "P")
      .split(/\\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map(part => part.charAt(0).toUpperCase())
      .join("") || "P";
    const productVisual = product => {{
      const item = product || {{}};
      const label = item.product_name || item.sku || "Product";
      return `<div class="rt-product-row">
        <div class="rt-product-thumb">${{escapeHtml(initials(label))}}</div>
        <div class="rt-product-info">
          <div class="rt-product-name">${{escapeHtml(label)}}</div>
          <div class="rt-product-meta">${{escapeHtml(item.brand || item.category || "")}}</div>
          ${{item.discount_price ? `<div class="rt-product-price">THB ${{escapeHtml(item.discount_price)}}</div>` : ""}}
        </div>
      </div>`;
    }};
    const skuVisual = sku => `<div class="rt-product-thumb">${{escapeHtml(initials(sku))}}</div>`;
    const skuList = action => String(action.skus || "")
      .split("+")
      .map(value => value.trim())
      .filter(Boolean);
    const captionKey = row => `${{Number(row.start || 0).toFixed(2)}}|${{Number(row.end || 0).toFixed(2)}}|${{row.text || ""}}`;
    const showNextCaption = () => {{
      if (!captionQueue.length) {{
        captionDisplayActive = false;
        return;
      }}
      captionDisplayActive = true;
      const row = captionQueue.shift();
      latestCaption = String(row.text || "");
      if (latestCaption) get("caption").textContent = latestCaption;
      const durationMs = Math.max(
        1400,
        Math.min(3600, Math.max(0.8, Number(row.end || 0) - Number(row.start || 0)) * 1000)
      );
      if (captionTimer) window.clearTimeout(captionTimer);
      captionTimer = window.setTimeout(showNextCaption, durationMs);
    }};
    const enqueueCaptions = rows => {{
      const items = Array.isArray(rows) ? rows : [];
      const fresh = [];
      for (const row of items) {{
        if (!row || !row.text) continue;
        const key = captionKey(row);
        if (seenCaptionKeys.has(key)) continue;
        seenCaptionKeys.add(key);
        fresh.push(row);
      }}
      if (fresh.length) {{
        captionQueue.push(...fresh);
        if (!captionDisplayActive) showNextCaption();
      }}
    }};
    const formatRemaining = (action, now) => {{
      const payload = action.displayPayload || {{}};
      const total = Number(payload.duration_seconds || (Number(payload.duration_minutes || 5) * 60));
      const elapsed = Math.max(0, Number(now || 0) - Number(action.time || 0));
      const remaining = Math.max(0, Math.ceil(total - elapsed));
      const minutes = Math.floor(remaining / 60);
      const seconds = String(remaining % 60).padStart(2, "0");
      return `${{minutes}}:${{seconds}}`;
    }};
    const renderActionCard = (action, now) => {{
      const type = action.action || "ACTION";
      const payload = action.displayPayload || {{}};
      const time = `${{Number(action.time || 0).toFixed(2)}}s`;
      if (type === "PIN_PRODUCT_CARD") {{
        const product = payload.product || {{}};
        return `
          <div class="rt-action-card rt-action-pin">
            <div class="rt-action-top"><strong>${{escapeHtml(product.product_name || action.title || "Pinned product")}}</strong><div class="rt-action-time">${{time}}</div></div>
            ${{productVisual(product)}}
          </div>`;
      }}
      if (type === "SHOW_PROMO_CODE") {{
        return `
          <div class="rt-action-card rt-action-promo">
            <div class="rt-action-top"><strong>${{escapeHtml(action.title || "Promo code")}}</strong><div class="rt-action-time">${{time}}</div></div>
            <div class="rt-promo-badge">${{escapeHtml(payload.promo_code || "PROMO")}}</div>
            <div class="rt-action-meta">${{escapeHtml(payload.promo_description || "Live promo detected")}}</div>
          </div>`;
      }}
      if (type === "SHOW_BUNDLE_RECOMMENDATION") {{
        const products = Array.isArray(payload.products) ? payload.products : [];
        const visuals = products.length
          ? products.map(productVisual).join('<div class="rt-bundle-plus">+</div>')
          : skuList(action).map(skuVisual).join('<div class="rt-bundle-plus">+</div>');
        return `
          <div class="rt-action-card rt-action-bundle">
            <div class="rt-action-top"><strong>${{escapeHtml(action.title || "Recommended bundle")}}</strong><div class="rt-action-time">${{time}}</div></div>
            <div class="rt-bundle-row">${{visuals}}</div>
            <div class="rt-action-meta">${{escapeHtml(payload.reason || "Products pair well together")}}</div>
          </div>`;
      }}
      if (type === "START_FLASH_SALE_COUNTDOWN") {{
        const products = Array.isArray(payload.products) ? payload.products : [];
        const productAttach = payload.product ? productVisual(payload.product) : "";
        const bundleAttach = products.length ? `<div class="rt-bundle-row">${{products.map(productVisual).join('<div class="rt-bundle-plus">+</div>')}}</div>` : "";
        return `
          <div class="rt-action-card rt-action-countdown">
            <div class="rt-action-top"><strong>Flash sale countdown</strong><div class="rt-action-time">${{time}}</div></div>
            ${{payload.promo_code ? `<div class="rt-promo-badge">${{escapeHtml(payload.promo_code)}}</div>` : (productAttach || bundleAttach || `<div class="rt-action-title">${{escapeHtml(action.title || "Live deal")}}</div>`)}}
            <div class="rt-countdown-time">${{formatRemaining(action, now)}}</div>
          </div>`;
      }}
      return `
        <div class="rt-action-card">
          <div class="rt-action-top"><strong>${{escapeHtml(action.title || humanActionTitle(type))}}</strong><div class="rt-action-time">${{time}}</div></div>
          <div class="rt-action-meta">${{escapeHtml(action.skus || "-")}}</div>
        </div>`;
    }};
    const renderActions = (rows, now = streamTime()) => {{
      if (rows !== undefined) lastActionRows = Array.isArray(rows) ? rows : [];
      const target = get("actions");
      const currentTarget = get("current-action");
      const items = lastActionRows;
      if (!items.length) {{
        if (target) target.innerHTML = '<div class="rt-empty">No actions emitted yet.</div>';
        if (currentTarget) currentTarget.innerHTML = '<div class="rt-empty">Current action will appear here.</div>';
        return;
      }}
      const current = items[items.length - 1];
      if (currentTarget) currentTarget.innerHTML = renderActionCard(current, now);
      if (target) target.innerHTML = items.slice(-6).reverse().map(action => renderActionCard(action, now)).join("");
    }};
    const updateTime = () => {{
      const total = duration();
      const current = streamTime();
      const time = get("time");
      const progress = get("progress");
      if (time) time.textContent = `${{current.toFixed(2)}}s / ${{total ? total.toFixed(2) : "..."}}s`;
      if (progress) progress.style.width = `${{total ? Math.max(0, Math.min(100, (current / total) * 100)) : 0}}%`;
      if (lastActionRows.length) renderActions(lastActionRows, current);
      if (playing && total && current >= total) {{
        playing = false;
        baseSeconds = total;
        if (ticker) window.clearTimeout(ticker);
        ticker = null;
        setState("ended");
        setDetail("Live replay finished.");
      }}
    }};
    const scheduleTick = () => {{
      if (ticker) window.clearTimeout(ticker);
      ticker = window.setTimeout(() => {{
        updateTime();
        if (playing) scheduleTick();
      }}, pollMilliseconds);
    }};
    const getStatus = () => {{
      updateTime();
      const current = streamTime();
      const total = duration();
      return {{
        currentTime: current,
        paused: !playing,
        ended: Boolean(total && current >= total),
        duration: total,
        stopped: stopRequested
      }};
    }};
    const resolveStart = () => {{
      if (pendingStartResolve) {{
        const resolve = pendingStartResolve;
        pendingStartResolve = null;
        resolve(getStatus());
      }}
    }};
    const setToggleButton = () => {{
      startButton.innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
      startButton.title = playing ? "Pause stream" : "Start stream";
      startButton.setAttribute("aria-label", playing ? "Pause stream" : "Start stream");
      startButton.disabled = stopRequested || !readyToStart || getStatus().ended;
    }};
    const startInput = () => {{
      if (stopRequested || !readyToStart) return;
      baseSeconds = streamTime();
      startedAtMs = performance.now();
      playing = true;
      setState("playing");
      setCaptionStatus("Streaming audio into ASR...");
      setDetail("Playback clock started. Chunks are released by stream time.");
      if (audio) {{
        const playResult = audio.play();
        if (playResult && typeof playResult.catch === "function") {{
          playResult.catch(() => {{
            setDetail("Browser blocked hidden audio playback; stream clock is still running.");
          }});
        }}
      }}
      updateTime();
      scheduleTick();
      setToggleButton();
      resolveStart();
      invokeCallback(startCallback, getStatus());
    }};
    const pauseInput = () => {{
      if (!playing) return;
      baseSeconds = streamTime();
      playing = false;
      if (ticker) window.clearTimeout(ticker);
      ticker = null;
      if (audio) audio.pause();
      setState("paused");
      setCaptionStatus("Stream paused. Press play to continue captions.");
      setDetail("Input is paused. ASR waits until stream time advances again.");
      updateTime();
      setToggleButton();
    }};
    const toggleInput = () => {{
      if (playing) pauseInput();
      else startInput();
    }};
    const stopInput = () => {{
      baseSeconds = streamTime();
      playing = false;
      stopRequested = true;
      if (ticker) window.clearTimeout(ticker);
      ticker = null;
      if (audio) audio.pause();
      setState("stopped");
      setDetail("Input stopped. Already released chunks will finish processing.");
      updateTime();
      setToggleButton();
      resolveStart();
      invokeCallback(stopCallback, getStatus());
    }};

    startButton.addEventListener("click", toggleInput);
    startButton.disabled = true;

    apiRoot[widgetId] = {{
      start: startInput,
      pause: pauseInput,
      stop: stopInput,
      getStatus,
      waitForStart() {{
        updateTime();
        setState("waiting_for_start");
        setDetail("Press Start to begin the simulated live stream.");
        if (stopRequested || playing || streamTime() > 0) return Promise.resolve(getStatus());
        return new Promise(resolve => {{
          pendingStartResolve = resolve;
        }});
      }},
      waitUntil(targetSeconds) {{
        updateTime();
        setState("waiting_for_playback");
        setDetail(`Waiting until stream reaches ${{Number(targetSeconds).toFixed(2)}}s.`);
        return new Promise(resolve => {{
          const check = () => {{
            updateTime();
            if (stopRequested || streamTime() >= Number(targetSeconds) || getStatus().ended) {{
              window.clearInterval(timer);
              resolve(getStatus());
            }}
          }};
          const timer = window.setInterval(check, pollMilliseconds);
          check();
        }});
      }},
      publish(payload) {{
        updateTime();
        if (payload.state) setState(payload.state);
        if (payload.state === "loading_asr") {{
          readyToStart = false;
          setToggleButton();
        }}
        if (payload.state === "asr_ready") {{
          readyToStart = true;
          setToggleButton();
        }}
        if (payload.captionRows) {{
          enqueueCaptions(payload.captionRows);
        }} else if (payload.latestCaption) {{
          enqueueCaptions([{{text: payload.latestCaption, start: 0, end: 1.5}}]);
        }}
        const captionCount = get("caption-count");
        if (captionCount && payload.captionCount !== undefined) {{
          captionCount.textContent = `${{payload.captionCount}} captions`;
        }}
        const actionCount = get("action-count");
        if (actionCount && payload.actionCount !== undefined) {{
          actionCount.textContent = `${{payload.actionCount}} actions`;
        }}
        if (payload.actionRows) renderActions(payload.actionRows, streamTime());
        const detail = [];
        if (payload.status) detail.push(payload.status);
        if (payload.chunkStart !== undefined && payload.chunkEnd !== undefined) {{
          detail.push(`chunk ${{Number(payload.chunkStart).toFixed(2)}}-${{Number(payload.chunkEnd).toFixed(2)}}s`);
        }}
        if (payload.playbackTime !== undefined) detail.push(`stream ${{Number(payload.playbackTime).toFixed(2)}}s`);
        if (payload.captionCount !== undefined) detail.push(`${{payload.captionCount}} captions`);
        if (payload.actionCount !== undefined) detail.push(`${{payload.actionCount}} actions`);
        setDetail(detail.join(" | ") || "Waiting.");
      }}
    }};
    if (audio) audio.addEventListener("loadedmetadata", updateTime);
    setState("ready");
    setDetail("Preparing ASR. Press play when the control is enabled.");
    updateTime();
    setToggleButton();
  }};
  install();
}})();
"""


def _build_realtime_audio_file_api_call_js(
    widget_id: str,
    method_name: str,
    args: Optional[Sequence[Any]] = None,
) -> str:
    return f"""
new Promise((resolve) => {{
  const widgetId = {json.dumps(widget_id)};
  const methodName = {json.dumps(method_name)};
  const args = {json.dumps(list(args or []))};
  const startedAt = Date.now();
  const callWhenReady = () => {{
    const api = window.realtimeAudioFileDemo && window.realtimeAudioFileDemo[widgetId];
    if (api && typeof api[methodName] === "function") {{
      Promise.resolve(api[methodName](...args))
        .then(resolve)
        .catch(error => resolve({{
          stopped: true,
          error: error && (error.message || String(error))
        }}));
      return;
    }}
    if (Date.now() - startedAt > 10000) {{
      resolve({{
        stopped: true,
        error: "live audio controls were not ready"
      }});
      return;
    }}
    window.setTimeout(callWhenReady, 100);
  }};
  callWhenReady();
}})
"""
