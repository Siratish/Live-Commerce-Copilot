from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import base64
import json
import mimetypes
import subprocess
import sys
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

    def install(self) -> None:
        try:
            from IPython.display import HTML, Javascript, display  # type: ignore
            from google.colab import output  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Browser audio playback gating requires Google Colab.") from exc

        display(HTML(_build_realtime_audio_file_html(self.audio_path, self.windows, self.widget_id)))
        display(Javascript(_build_realtime_audio_file_js(self.widget_id, self.poll_seconds)))

    def wait_for_start(self) -> Dict[str, Any]:
        from google.colab import output  # type: ignore

        return output.eval_js(
            "window.realtimeAudioFileDemo"
            f"[{json.dumps(self.widget_id)}].waitForStart()"
        )

    def wait_until(self, target_seconds: float) -> Dict[str, Any]:
        from google.colab import output  # type: ignore

        return output.eval_js(
            "window.realtimeAudioFileDemo"
            f"[{json.dumps(self.widget_id)}].waitUntil({float(target_seconds)})"
        )

    def status(self) -> Dict[str, Any]:
        from google.colab import output  # type: ignore

        status = output.eval_js(
            "window.realtimeAudioFileDemo"
            f"[{json.dumps(self.widget_id)}].getStatus()"
        )
        return status if isinstance(status, dict) else {}

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

    gate = playback_gate or _default_playback_gate(
        config.audio_path,
        audio_windows,
        config.playback_poll_seconds,
    )
    gate.install()
    gate.publish(
        {
            "state": "waiting",
            "status": "press_play",
            "totalChunks": len(audio_windows),
            "processedChunks": 0,
            "queueDepth": 0,
        }
    )
    gate.wait_for_start()

    asr_engine = asr or _build_file_stream_asr(config)
    catalog = load_product_catalog(config.catalog_path)
    promotions = load_promotions(config.promotions_path)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = config.output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    live_segments: List[CaptionSegment] = []
    live_actions = []
    processed_chunks = 0
    max_backlog = 0

    for index, window in enumerate(audio_windows, start=1):
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
        save_caption_json(caption_result, config.output_dir / "realtime_file_captions.json")
        save_commerce_actions(live_actions, config.output_dir / "realtime_file_actions.json")
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
        "audio_path": str(config.audio_path),
        "captions_json": str(config.output_dir / "realtime_file_captions.json"),
        "actions_json": str(config.output_dir / "realtime_file_actions.json"),
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
    rows = "".join(
        (
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{window.start:.2f}s</td>"
            f"<td>{window.end:.2f}s</td>"
            f"<td>{window.end - window.start:.2f}s</td>"
            "</tr>"
        )
        for index, window in enumerate(windows[:12], start=1)
    )
    return f"""
<div id="{widget_id}" style="border:1px solid #d0d5dd;border-radius:8px;padding:12px;font-family:Arial,sans-serif;max-width:900px;background:#fff;">
  <div style="font-weight:700;font-size:16px;margin-bottom:8px;">Real-Time Audio File Stream</div>
  <audio controls preload="metadata" src="{_audio_data_uri(audio_path)}" style="width:100%;margin-bottom:10px;"></audio>
  <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:10px;">
    <div><div style="color:#667085;font-size:12px;">Playback</div><div data-role="time" style="font-weight:700;">0.00s / {duration:.2f}s</div></div>
    <div><div style="color:#667085;font-size:12px;">State</div><div data-role="state" style="font-weight:700;">waiting</div></div>
    <div><div style="color:#667085;font-size:12px;">Chunk</div><div data-role="chunk" style="font-weight:700;">-</div></div>
    <div><div style="color:#667085;font-size:12px;">Queue</div><div data-role="queue" style="font-weight:700;">0</div></div>
  </div>
  <div style="height:10px;background:#f2f4f7;border-radius:999px;overflow:hidden;margin-bottom:8px;">
    <div data-role="progress" style="height:100%;width:0%;background:#e51b23;"></div>
  </div>
  <div data-role="detail" style="color:#667085;font-size:12px;margin-bottom:10px;">Press play. ASR will not start for a chunk until playback reaches that chunk end.</div>
  <details>
    <summary style="cursor:pointer;color:#344054;">Prepared stream chunks ({len(windows)})</summary>
    <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:12px;">
      <thead><tr><th align="left">#</th><th align="left">start</th><th align="left">release at</th><th align="left">length</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </details>
</div>
"""


def _build_realtime_audio_file_js(widget_id: str, poll_seconds: float) -> str:
    poll_milliseconds = max(50, int(float(poll_seconds) * 1000))
    return f"""
(() => {{
  const root = document.getElementById({json.dumps(widget_id)});
  if (!root) return;
  const audio = root.querySelector("audio");
  const pollMilliseconds = {poll_milliseconds};
  const get = role => root.querySelector(`[data-role="${{role}}"]`);
  const duration = () => audio.duration && Number.isFinite(audio.duration) ? audio.duration : 0;
  const updateTime = () => {{
    const total = duration();
    get("time").textContent = `${{(audio.currentTime || 0).toFixed(2)}}s / ${{total ? total.toFixed(2) : "..."}}s`;
    const pct = total ? Math.max(0, Math.min(100, (audio.currentTime / total) * 100)) : 0;
    get("progress").style.width = `${{pct}}%`;
  }};
  const apiRoot = window.realtimeAudioFileDemo = window.realtimeAudioFileDemo || {{}};
  apiRoot[{json.dumps(widget_id)}] = {{
    getStatus() {{
      updateTime();
      return {{
        currentTime: audio.currentTime || 0,
        paused: audio.paused,
        ended: audio.ended,
        duration: duration()
      }};
    }},
    waitForStart() {{
      updateTime();
      get("state").textContent = "waiting_for_play";
      get("detail").textContent = "Press play to begin the simulated live stream.";
      if (!audio.paused && !audio.ended) return Promise.resolve(this.getStatus());
      return new Promise(resolve => {{
        const onPlay = () => {{
          audio.removeEventListener("play", onPlay);
          get("state").textContent = "playing";
          get("detail").textContent = "Playback started. Chunks are released by audio time.";
          resolve(this.getStatus());
        }};
        audio.addEventListener("play", onPlay);
      }});
    }},
    waitUntil(targetSeconds) {{
      updateTime();
      get("state").textContent = "waiting_for_playback";
      get("detail").textContent = `Waiting until playback reaches ${{Number(targetSeconds).toFixed(2)}}s.`;
      return new Promise(resolve => {{
        const check = () => {{
          updateTime();
          if ((audio.currentTime || 0) >= targetSeconds || audio.ended) {{
            cleanup();
            resolve(this.getStatus());
          }}
        }};
        const cleanup = () => {{
          clearInterval(timer);
          audio.removeEventListener("timeupdate", check);
          audio.removeEventListener("seeked", check);
          audio.removeEventListener("ended", check);
          audio.removeEventListener("play", check);
        }};
        const timer = setInterval(check, pollMilliseconds);
        audio.addEventListener("timeupdate", check);
        audio.addEventListener("seeked", check);
        audio.addEventListener("ended", check);
        audio.addEventListener("play", check);
        check();
      }});
    }},
    publish(payload) {{
      updateTime();
      if (payload.state) get("state").textContent = payload.state;
      if (payload.chunkIndex) get("chunk").textContent = `${{payload.chunkIndex}} / ${{payload.totalChunks || "?"}}`;
      if (payload.queueDepth !== undefined) get("queue").textContent = String(payload.queueDepth);
      const detail = [];
      if (payload.status) detail.push(payload.status);
      if (payload.chunkStart !== undefined && payload.chunkEnd !== undefined) {{
        detail.push(`chunk ${{Number(payload.chunkStart).toFixed(2)}}-${{Number(payload.chunkEnd).toFixed(2)}}s`);
      }}
      if (payload.playbackTime !== undefined) detail.push(`playback ${{Number(payload.playbackTime).toFixed(2)}}s`);
      if (payload.captionCount !== undefined) detail.push(`${{payload.captionCount}} captions`);
      if (payload.actionCount !== undefined) detail.push(`${{payload.actionCount}} actions`);
      get("detail").textContent = detail.join(" | ") || "Waiting.";
    }}
  }};
  audio.addEventListener("timeupdate", updateTime);
  audio.addEventListener("loadedmetadata", updateTime);
  updateTime();
}})();
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
