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
        print("Colab browser playback APIs are unavailable; using wall-clock replay.")

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
        if payload.get("state"):
            print(f"[file stream] {payload['state']}: {payload.get('status', '')}")


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
        callback_suffix = self.widget_id.replace("-", "_")
        self._start_callback = f"live_commerce_file_start_{callback_suffix}"
        self._stop_callback = f"live_commerce_file_stop_{callback_suffix}"
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self._started_at: Optional[float] = None

    def install(self) -> None:
        if self._installed:
            return
        try:
            from IPython.display import HTML, Javascript, display  # type: ignore
            from google.colab import output  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Browser audio playback gating requires Google Colab.") from exc

        output.register_callback(self._start_callback, self._handle_start)
        output.register_callback(self._stop_callback, self._handle_stop)
        display(HTML(_build_realtime_audio_file_html(self.audio_path, self.windows, self.widget_id)))
        script = _build_realtime_audio_file_js(
            self.widget_id,
            self.poll_seconds,
            self._start_callback,
            self._stop_callback,
        )
        try:
            output.eval_js(script)
        except Exception:
            display(Javascript(script))
        self._installed = True

    def _handle_start(self, *_args: Any) -> Dict[str, Any]:
        if self._started_at is None:
            self._started_at = time.monotonic()
        self._start_event.set()
        return self.status()

    def _handle_stop(self, *_args: Any) -> Dict[str, Any]:
        self._mark_stopped()
        return self.status()

    def wait_for_start(self) -> Dict[str, Any]:
        self._start_event.wait()
        return self.status()

    def wait_until(self, target_seconds: float) -> Dict[str, Any]:
        if self._started_at is None:
            self.wait_for_start()
        assert self._started_at is not None
        while not self._stop_event.is_set():
            current_time = time.monotonic() - self._started_at
            if current_time >= float(target_seconds):
                return self.status()
            time.sleep(min(self.poll_seconds, max(0.01, float(target_seconds) - current_time)))
        return self.status()

    def status(self) -> Dict[str, Any]:
        current_time = 0.0
        if self._started_at is not None:
            current_time = max(0.0, time.monotonic() - self._started_at)
        duration = max((window.end for window in self.windows), default=0.0)
        return {
            "currentTime": min(current_time, duration) if duration else current_time,
            "duration": duration,
            "paused": self._started_at is None or self._stop_event.is_set(),
            "ended": bool(duration and current_time >= duration),
            "stopped": self._stop_event.is_set(),
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
    gate.publish(
        {
            "state": "waiting",
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
            "max_backlog_chunks": 0,
            "stopped": True,
            "audio_path": str(config.audio_path),
            "captions_json": str(captions_path),
            "actions_json": str(actions_path),
            "chunk_dir": str(chunk_dir),
        }

    asr_engine = asr or _build_file_stream_asr(config)
    catalog = load_product_catalog(config.catalog_path)
    promotions = load_promotions(config.promotions_path)

    live_segments: List[CaptionSegment] = []
    live_actions = []
    processed_chunks = 0
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
        _display_realtime_file_state(
            index=index,
            total=len(audio_windows),
            window=window,
            chunk_path=chunk_path,
            current_time=current_time,
            queue_depth=backlog,
            segments=live_segments,
            actions=live_actions,
        )
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
                "totalChunks": len(audio_windows),
                "captionCount": len(live_segments),
                "actionCount": len(live_actions),
            }
        )

    return {
        "caption_count": len(live_segments),
        "action_count": len(live_actions),
        "processed_chunks": processed_chunks,
        "max_backlog_chunks": max_backlog,
        "stopped": stopped_by_user,
        "audio_path": str(config.audio_path),
        "captions_json": str(captions_path),
        "actions_json": str(actions_path),
        "chunk_dir": str(chunk_dir),
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
    #{widget_id} .rt-panel-title{{font-weight:800;font-size:17px;color:#111827;margin-bottom:8px}}
    #{widget_id} .rt-muted{{color:#667085;font-size:12px;line-height:1.4}}
    #{widget_id} .rt-metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:12px 0}}
    #{widget_id} .rt-metric{{border:1px solid #eaecf0;border-radius:8px;padding:10px;background:#f9fafb}}
    #{widget_id} .rt-metric strong{{display:block;color:#111827;margin-top:3px}}
    #{widget_id} .rt-buttons{{display:flex;gap:8px;margin:10px 0}}
    #{widget_id} button{{border:0;border-radius:8px;padding:10px 13px;font-weight:800;cursor:pointer}}
    #{widget_id} [data-role=start]{{background:#b42318;color:#fff}}
    #{widget_id} [data-role=stop]{{background:#f2f4f7;color:#344054}}
    #{widget_id} button:disabled{{opacity:.5;cursor:not-allowed}}
    #{widget_id} .rt-progress{{height:10px;background:#f2f4f7;border-radius:999px;overflow:hidden;margin-top:8px}}
    #{widget_id} .rt-progress div{{height:100%;width:0%;background:#e51b23}}
    @media (max-width: 900px){{#{widget_id} .rt-grid{{grid-template-columns:1fr}}}}
  </style>
  <audio preload="metadata" src="{_audio_data_uri(audio_path)}" style="display:none;"></audio>
  <div class="rt-grid">
    <div>
      <div class="rt-video">
        <div class="rt-host"><div class="rt-head"></div><div class="rt-body"></div></div>
        <div class="rt-caption" data-role="caption">Press Start to begin the simulated live stream.</div>
      </div>
      <div class="rt-progress"><div data-role="progress"></div></div>
    </div>
    <div>
      <div class="rt-panel-title">Live Audio Stream</div>
      <div class="rt-muted">Audio is released to ASR only as stream time advances. Seeking is disabled for this demo.</div>
      <div class="rt-buttons">
        <button data-role="start" disabled>Start / Continue</button>
        <button data-role="stop">Stop input</button>
      </div>
      <div class="rt-metrics">
        <div class="rt-metric"><div class="rt-muted">Playback</div><strong data-role="time">0.00s / {duration:.2f}s</strong></div>
        <div class="rt-metric"><div class="rt-muted">State</div><strong data-role="state">ready</strong></div>
        <div class="rt-metric"><div class="rt-muted">Chunk</div><strong data-role="chunk">-</strong></div>
        <div class="rt-metric"><div class="rt-muted">Queue</div><strong data-role="queue">0</strong></div>
      </div>
      <div class="rt-muted" data-role="detail">Start the stream to release the first chunk.</div>
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
    const stopButton = root.querySelector('[data-role="stop"]');
    const get = role => root.querySelector(`[data-role="${{role}}"]`);
    const apiRoot = window.realtimeAudioFileDemo = window.realtimeAudioFileDemo || {{}};
    let stopRequested = false;
    let playing = false;
    let baseSeconds = 0;
    let startedAtMs = 0;
    let ticker = null;
    let pendingStartResolve = null;

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
    const updateTime = () => {{
      const total = duration();
      const current = streamTime();
      const time = get("time");
      const progress = get("progress");
      if (time) time.textContent = `${{current.toFixed(2)}}s / ${{total ? total.toFixed(2) : "..."}}s`;
      if (progress) progress.style.width = `${{total ? Math.max(0, Math.min(100, (current / total) * 100)) : 0}}%`;
      if (playing && total && current >= total) {{
        playing = false;
        baseSeconds = total;
        if (ticker) window.clearTimeout(ticker);
        ticker = null;
        const state = get("state");
        if (state) state.textContent = "ended";
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
    const startInput = () => {{
      if (stopRequested) return;
      baseSeconds = streamTime();
      startedAtMs = performance.now();
      playing = true;
      startButton.disabled = true;
      stopButton.disabled = false;
      get("state").textContent = "playing";
      get("caption").textContent = "Streaming audio into ASR...";
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
      resolveStart();
      invokeCallback(startCallback, getStatus());
    }};
    const stopInput = () => {{
      baseSeconds = streamTime();
      playing = false;
      stopRequested = true;
      if (ticker) window.clearTimeout(ticker);
      ticker = null;
      if (audio) audio.pause();
      startButton.disabled = true;
      stopButton.disabled = true;
      get("state").textContent = "stopped";
      setDetail("Input stopped. Already released chunks will finish processing.");
      updateTime();
      resolveStart();
      invokeCallback(stopCallback, getStatus());
    }};

    startButton.addEventListener("click", startInput);
    stopButton.addEventListener("click", stopInput);
    startButton.disabled = false;
    stopButton.disabled = false;

    apiRoot[widgetId] = {{
      start: startInput,
      stop: stopInput,
      getStatus,
      waitForStart() {{
        updateTime();
        get("state").textContent = "waiting_for_start";
        setDetail("Press Start to begin the simulated live stream.");
        if (stopRequested || playing || streamTime() > 0) return Promise.resolve(getStatus());
        return new Promise(resolve => {{
          pendingStartResolve = resolve;
        }});
      }},
      waitUntil(targetSeconds) {{
        updateTime();
        get("state").textContent = "waiting_for_playback";
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
        if (payload.state) get("state").textContent = payload.state;
        if (payload.chunkIndex) get("chunk").textContent = `${{payload.chunkIndex}} / ${{payload.totalChunks || "?"}}`;
        if (payload.queueDepth !== undefined) get("queue").textContent = String(payload.queueDepth);
        if (payload.status) get("caption").textContent = payload.status;
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
    get("state").textContent = "ready";
    setDetail("Controls ready. Press Start to release audio chunks in real time.");
    updateTime();
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


def _display_realtime_file_state(
    index: int,
    total: int,
    window: AudioWindow,
    chunk_path: Path,
    current_time: float,
    queue_depth: int,
    segments: Sequence[CaptionSegment],
    actions: Sequence[Any],
) -> None:
    print(
        f"Realtime file chunk {index}/{total} released at {window.end:.2f}s "
        f"(playback {current_time:.2f}s), saved to {chunk_path}"
    )
    print(
        f"Buffered chunks waiting: {queue_depth} | "
        f"Captions: {len(segments)} | Actions: {len(actions)}"
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
        from IPython.display import display  # type: ignore
    except ImportError:
        return

    try:
        import pandas as pd  # type: ignore

        display(pd.DataFrame(caption_rows))
        display(pd.DataFrame(action_rows))
    except Exception:
        display({"captions": caption_rows, "actions": action_rows})
