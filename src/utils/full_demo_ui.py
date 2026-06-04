from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import csv
import html
import json
import mimetypes
import threading
import time
import uuid

from src.ai.captioning import (
    CaptionResult,
    CaptioningEngine,
    CaptioningSettings,
    caption_metrics,
    iter_audio_windows,
    load_cached_transcript,
    write_caption_outputs,
)
from src.ai.commerce_actions import generate_commerce_actions, save_commerce_actions
from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import CommerceAction, ProductCatalogItem, Promotion
from src.utils.action_timeline import save_action_timeline_html
from src.utils.live_mic import (
    BrowserMicChunkSource,
    LiveMicDemoConfig,
    _build_live_asr,
    run_colab_live_mic_demo,
)
from src.utils.realtime_audio_file import (
    RealtimeAudioFileDemoConfig,
    _build_file_stream_asr,
    run_realtime_audio_file_demo,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "data" / "demo" / "product_catalog.csv"
DEFAULT_PROMOTIONS = REPO_ROOT / "data" / "demo" / "promotions.csv"
SAMPLE_AUDIO = {
    "Audio 1 - Beauty": REPO_ROOT / "data" / "demo" / "audio" / "1.mp3",
    "Audio 2 - Tech": REPO_ROOT / "data" / "demo" / "audio" / "2.mp3",
}
SAMPLE_CACHED_TRANSCRIPTS = {
    "Audio 1 - Beauty": REPO_ROOT / "data" / "demo" / "audio_1_captions.json",
    "Audio 2 - Tech": REPO_ROOT / "data" / "demo" / "audio_2_captions.json",
}
SAMPLE_CACHED_ACTIONS = {
    "Audio 1 - Beauty": REPO_ROOT / "data" / "demo" / "audio_1_actions.json",
    "Audio 2 - Tech": REPO_ROOT / "data" / "demo" / "audio_2_actions.json",
}
FULL_DEMO_ASR_PROVIDER = "openai_whisper"
FULL_DEMO_ASR_MODEL = "turbo"
FULL_DEMO_LANGUAGE = "th"
FULL_DEMO_CHUNK_SECONDS = 15.0
FULL_DEMO_MIN_CHUNK_SECONDS = 1.0
FULL_DEMO_PAUSE_SECONDS = 0.3
FULL_DEMO_SILENCE_THRESHOLD = 0.015
FULL_DEMO_DYNAMIC_CHUNKING = True


class FullDemoSession:
    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.output_dir = repo_root / "outputs" / "full_demo"
        self.upload_dir = self.output_dir / "uploads"
        self.product_image_dir = self.output_dir / "product_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.product_image_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = load_product_catalog(DEFAULT_CATALOG)
        self.promotions = load_promotions(DEFAULT_PROMOTIONS)
        self.product_images: Dict[str, str] = {}
        self.catalog_path = self.output_dir / "session_product_catalog.csv"
        self.promotions_path = self.output_dir / "session_promotions.csv"
        self.save_catalog_files()

    def save_catalog_files(self) -> None:
        save_product_catalog_csv(self.catalog, self.catalog_path)
        save_promotions_csv(self.promotions, self.promotions_path)

    def add_product(
        self,
        item: ProductCatalogItem,
        image_path: Optional[Path] = None,
    ) -> None:
        self.catalog = [existing for existing in self.catalog if existing.sku != item.sku]
        self.catalog.append(item)
        if image_path is not None and image_path.exists():
            self.product_images[item.sku] = _image_data_uri(image_path)
        self.save_catalog_files()

    def add_promotion(self, promotion: Promotion) -> None:
        self.promotions = [
            existing
            for existing in self.promotions
            if existing.promo_code != promotion.promo_code
        ]
        self.promotions.append(promotion)
        self.save_catalog_files()


_FULL_DEMO_SESSIONS: Dict[Path, FullDemoSession] = {}


def _get_full_demo_session(repo_root: Path = REPO_ROOT) -> FullDemoSession:
    resolved = Path(repo_root).resolve()
    if resolved not in _FULL_DEMO_SESSIONS:
        _FULL_DEMO_SESSIONS[resolved] = FullDemoSession(resolved)
    return _FULL_DEMO_SESSIONS[resolved]


class FullDemoPlaybackGate:
    """ipywidgets-controlled playback gate for the full notebook UI."""

    def __init__(
        self,
        widgets: Any,
        audio_path: Path,
        windows: Sequence[Any],
        source_label: str,
        poll_seconds: float = 0.25,
    ) -> None:
        self.widgets = widgets
        self.audio_path = audio_path
        self.windows = list(windows)
        self.source_label = source_label
        self.poll_seconds = poll_seconds
        self.widget_id = f"full-demo-audio-{uuid.uuid4().hex}"
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._ready_to_start = False
        self._playing = False
        self._base_seconds = 0.0
        self._play_started_at: Optional[float] = None
        self._duration = max((window.end for window in self.windows), default=0.0)
        self._state = "ready"
        self._detail = "press_start"
        self._latest_caption = "Waiting for live transcript..."
        self._caption_count = 0
        self._action_rows: List[Dict[str, Any]] = []
        self._ticker_started = False
        self.caption_id = f"{self.widget_id}-caption"
        self.status_html = widgets.HTML()
        self.detail_html = widgets.HTML()
        self.video_html = widgets.HTML(self._video_html(), layout=widgets.Layout(width="100%"))
        self.progress_html = widgets.HTML(
            self._progress_html(),
            layout=widgets.Layout(width="100%"),
        )
        self.actions_html = widgets.HTML(
            self._action_rail_html(),
            layout=widgets.Layout(width="100%"),
        )
        self.toggle_button = widgets.Button(
            description="",
            icon="play",
            button_style="danger",
            tooltip="Start stream",
            layout=widgets.Layout(width="54px", height="42px"),
        )
        self.toggle_button.disabled = True
        self.toggle_button.on_click(lambda _button: self.toggle())
        self.ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                self.video_html,
                                widgets.HBox(
                                    [self.toggle_button, self.progress_html],
                                    layout=widgets.Layout(
                                        align_items="center",
                                        gap="10px",
                                        width="100%",
                                        margin="10px 0 0 0",
                                    ),
                                ),
                            ],
                            layout=widgets.Layout(width="62%"),
                        ),
                        widgets.VBox(
                            [self.actions_html],
                            layout=widgets.Layout(width="38%"),
                        ),
                    ],
                    layout=widgets.Layout(
                        align_items="flex-start",
                        gap="14px",
                        width="100%",
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )
        self.publish(
            {
                "state": "ready",
                "status": "press_start",
                "totalChunks": len(self.windows),
                "processedChunks": 0,
                "queueDepth": 0,
            }
        )

    def install(self) -> None:
        return

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.start()

    def start(self) -> None:
        if (
            self._stopped.is_set()
            or not self._ready_to_start
            or bool(self.status().get("ended"))
        ):
            return
        self._play_started_at = time.monotonic()
        self._playing = True
        self._started.set()
        self.publish({"state": "playing", "status": "stream_started"})
        self._start_ticker()
        self._eval_audio_js("play")

    def pause(self) -> None:
        if not self._playing:
            return
        self._base_seconds = self._current_time()
        self._play_started_at = None
        self._playing = False
        self.publish({"state": "paused", "status": "stream_paused"})
        self._eval_audio_js("pause")

    def stop(self) -> None:
        self._base_seconds = self._current_time()
        self._play_started_at = None
        self._playing = False
        self._stopped.set()
        self._started.set()
        self.publish({"state": "stopped", "status": "input_closed"})
        self._eval_audio_js("pause")

    def wait_for_start(self) -> Dict[str, Any]:
        self._started.wait()
        return self.status()

    def wait_until(self, target_seconds: float) -> Dict[str, Any]:
        if not self._started.is_set():
            self.wait_for_start()
        while not self._stopped.is_set():
            current = self._current_time()
            if current >= float(target_seconds):
                return self.status()
            time.sleep(min(self.poll_seconds, max(0.01, float(target_seconds) - current)))
        return self.status()

    def status(self) -> Dict[str, Any]:
        current = self._current_time()
        current = min(current, self._duration) if self._duration else current
        if self._duration and current >= self._duration:
            self._base_seconds = self._duration
            self._play_started_at = None
            self._playing = False
        return {
            "currentTime": current,
            "duration": self._duration,
            "paused": not self._playing,
            "ended": bool(self._duration and current >= self._duration),
            "stopped": self._stopped.is_set(),
        }

    def _current_time(self) -> float:
        if self._playing and self._play_started_at is not None:
            current = self._base_seconds + (time.monotonic() - self._play_started_at)
        else:
            current = self._base_seconds
        if self._duration:
            current = min(current, self._duration)
        return max(0.0, current)

    def publish(self, payload: Dict[str, Any]) -> None:
        self._state = str(payload.get("state") or self._state or "waiting")
        if self._state == "loading_asr":
            self._ready_to_start = False
        elif self._state == "asr_ready":
            self._ready_to_start = True
        status = self.status()
        detail_parts = [str(payload.get("status") or "waiting")]
        if payload.get("chunkIndex"):
            detail_parts.append(
                f"chunk {payload.get('chunkIndex')} / {payload.get('totalChunks', '?')}"
            )
        if payload.get("chunkStart") is not None and payload.get("chunkEnd") is not None:
            detail_parts.append(
                f"{float(payload['chunkStart']):.2f}-{float(payload['chunkEnd']):.2f}s"
            )
        if payload.get("captionCount") is not None:
            self._caption_count = int(payload.get("captionCount") or 0)
            detail_parts.append(f"{payload['captionCount']} captions")
        if payload.get("actionCount") is not None:
            detail_parts.append(f"{payload['actionCount']} actions")
        if payload.get("latestCaption"):
            self._latest_caption = str(payload["latestCaption"])
        if payload.get("actionRows") is not None:
            self._action_rows = list(payload.get("actionRows") or [])
        self._detail = " | ".join(detail_parts)
        self._render_status()

    def _render_status(self) -> None:
        status = self.status()
        self.progress_html.value = self._progress_html(status["currentTime"])
        self.actions_html.value = self._action_rail_html()
        self._eval_caption_js(self._latest_caption)
        self._render_toggle()
        return
        self.status_html.value = (
            '<div class="lc-selected-summary">'
            f"<strong>{html.escape(self._state)}</strong> · "
            f"{status['currentTime']:.2f}s / {self._duration:.2f}s"
            "</div>"
        )
        self.detail_html.value = (
            '<div class="lc-source-detail">'
            + html.escape(self._detail)
            + "</div>"
        )
        self._render_toggle()

    def _render_toggle(self) -> None:
        status = self.status()
        if self._playing:
            self.toggle_button.icon = "pause"
            self.toggle_button.tooltip = "Pause stream"
        else:
            self.toggle_button.icon = "play"
            self.toggle_button.tooltip = "Start or resume stream"
        self.toggle_button.disabled = (
            self._stopped.is_set()
            or not self._ready_to_start
            or bool(status.get("ended"))
        )

    def _start_ticker(self) -> None:
        if self._ticker_started:
            return
        self._ticker_started = True

        def tick() -> None:
            while not self._stopped.is_set():
                self._render_status()
                status = self.status()
                if status.get("ended"):
                    break
                time.sleep(max(0.1, self.poll_seconds))
            self._render_status()

        threading.Thread(target=tick, daemon=True).start()

    def _video_html(self) -> str:
        return f"""
<div class="lc-video">
    <audio id="{self.widget_id}" preload="metadata" src="{_audio_data_uri(self.audio_path)}" style="display:none;"></audio>
    <div class="lc-video-art">
      <div class="lc-video-badge">Live</div>
      <div class="lc-host-frame">
        <div class="lc-host-head"></div>
        <div class="lc-host-body"></div>
      </div>
      <div class="lc-video-caption" id="{self.caption_id}">{html.escape(self._latest_caption)}</div>
    </div>
</div>
"""

    def _progress_html(self, current_time: float = 0.0) -> str:
        percent = 0.0
        if self._duration:
            percent = max(0.0, min(100.0, (float(current_time) / self._duration) * 100.0))
        return f'<div class="lc-progress-bar lc-inline-progress"><div style="width:{percent:.2f}%"></div></div>'

    def _action_rail_html(self) -> str:
        return (
            '<div class="lc-action-rail">'
            '<div class="lc-panel-title">Live ASR + action preview</div>'
            '<div class="lc-stats">'
            f"<span>{self._caption_count} captions</span>"
            f"<span>{len(self._action_rows)} actions</span>"
            "</div>"
            f'<div class="lc-live-actions">{_live_action_cards_html(self._action_rows)}</div>'
            "</div>"
        )

    def _eval_caption_js(self, text: str) -> None:
        script = (
            f"const caption = document.getElementById({json.dumps(self.caption_id)});"
            f"if (caption) {{ caption.textContent = {json.dumps(text)}; }}"
        )
        try:
            from google.colab import output  # type: ignore

            output.eval_js(script)
        except Exception:
            try:
                from IPython.display import Javascript, display  # type: ignore

                display(Javascript(script))
            except Exception:
                pass

    def _eval_audio_js(self, action: str) -> None:
        script = (
            f"const audio = document.getElementById({json.dumps(self.widget_id)});"
            f"if (audio) {{ audio.{action}(); }}"
        )
        try:
            from google.colab import output  # type: ignore

            output.eval_js(script)
        except Exception:
            try:
                from IPython.display import Javascript, display  # type: ignore

                display(Javascript(script))
            except Exception:
                pass


class FullDemoMicGate:
    """Browser-controlled mic gate for the full notebook UI."""

    def __init__(self, widgets: Any, chunk_source: BrowserMicChunkSource) -> None:
        self.widgets = widgets
        self.chunk_source = chunk_source
        self._ready = False
        self._recording = False
        self._started_once = False
        self._stopped = False
        self._state = "ready"
        self._detail = "Press Start to open the browser mic."
        self.widget_id = f"full-demo-mic-{uuid.uuid4().hex}"
        self.caption_id = f"{self.widget_id}-caption"
        self._latest_caption = "Waiting for live transcript..."
        self._caption_count = 0
        self._action_rows: List[Dict[str, Any]] = []
        self.status_html = widgets.HTML()
        self.detail_html = widgets.HTML()
        self.video_html = widgets.HTML(self._video_html(), layout=widgets.Layout(width="100%"))
        self.progress_html = widgets.HTML(
            self._progress_html(),
            layout=widgets.Layout(width="100%"),
        )
        self.actions_html = widgets.HTML(
            self._action_rail_html(),
            layout=widgets.Layout(width="100%"),
        )
        self.controls_html = widgets.HTML(
            self._controls_html(),
            layout=widgets.Layout(width="54px"),
        )
        self.ui = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                self.video_html,
                                widgets.HBox(
                                    [self.controls_html, self.progress_html],
                                    layout=widgets.Layout(
                                        align_items="center",
                                        gap="10px",
                                        width="100%",
                                        margin="10px 0 0 0",
                                    ),
                                ),
                            ],
                            layout=widgets.Layout(width="62%"),
                        ),
                        widgets.VBox(
                            [self.actions_html],
                            layout=widgets.Layout(width="38%"),
                        ),
                    ],
                    layout=widgets.Layout(
                        align_items="flex-start",
                        gap="14px",
                        width="100%",
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )
        self.publish("ready", "Press Start to open the browser mic.")

    def install(self) -> None:
        script = self._gate_js()
        try:
            from google.colab import output  # type: ignore

            output.eval_js(script)
        except Exception:
            try:
                from IPython.display import Javascript, display  # type: ignore

                display(Javascript(script))
            except Exception:
                pass

    def toggle(self) -> None:
        if self._recording:
            self.pause()
        else:
            self.start()

    def start(self) -> None:
        if not self._ready or self._stopped or self._recording:
            return
        self._recording = True
        if self._started_once:
            self.publish("recording", "Browser mic input resumed.")
            self.chunk_source.resume()
        else:
            self._started_once = True
            self.publish("recording", "Browser mic is starting. Speak after permission is granted.")
            self.chunk_source.start()

    def pause(self) -> None:
        if not self._recording or self._stopped:
            return
        self._recording = False
        self.chunk_source.pause()
        self.publish("paused", "Mic input is paused. Queued chunks may still finish processing.")

    def stop(self) -> None:
        self._recording = False
        self._stopped = True
        self.chunk_source.stop()
        self.publish("stopped", "Input stopped. Queued chunks may still finish processing.")

    def publish(self, state: str, detail: str) -> None:
        self._state = state
        self._detail = detail
        if state == "ready":
            self._ready = True
        elif state in {"loading_asr", "stopped"}:
            self._ready = False
        self._eval_caption_js(self._latest_caption)
        self.actions_html.value = self._action_rail_html()
        self._render_toggle()
        self._publish_browser_state()

    def update_results(
        self,
        segments: Sequence[Any],
        actions: Sequence[Any],
        _metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if segments:
            self._latest_caption = str(segments[-1].text)
        self._caption_count = len(segments)
        self._action_rows = [
            {
                "time": round(action.timestamp, 2),
                "action": action.action_type,
                "title": action.display_payload.get("title", action.action_type),
                "skus": " + ".join(action.skus),
            }
            for action in actions[-6:]
        ]
        self.actions_html.value = self._action_rail_html()
        self._eval_caption_js(self._latest_caption)

    def _render_toggle(self) -> None:
        self.progress_html.value = self._progress_html()

    def _video_html(self) -> str:
        return """
<div class="lc-video">
    <div class="lc-video-art">
      <div class="lc-video-badge">Live mic</div>
      <div class="lc-host-frame">
        <div class="lc-host-head"></div>
        <div class="lc-host-body"></div>
      </div>
      <div class="lc-video-caption" id="{caption_id}">Waiting for live transcript...</div>
    </div>
</div>
""".format(caption_id=html.escape(self.caption_id))

    def _progress_html(self) -> str:
        width = "100%" if self._recording else "0%"
        return (
            f'<div class="lc-progress-bar lc-inline-progress">'
            f'<div id="{html.escape(self.widget_id)}-progress" style="width:{width}"></div>'
            "</div>"
        )

    def _controls_html(self) -> str:
        return (
            f'<div id="{html.escape(self.widget_id)}" class="lc-live-mic-controls">'
            '<button class="lc-live-toggle" data-role="toggle" disabled '
            'title="Enable mic input" aria-label="Enable mic input">&#9654;</button>'
            "</div>"
        )

    def _action_rail_html(self) -> str:
        return (
            '<div class="lc-action-rail">'
            '<div class="lc-panel-title">Live ASR + action preview</div>'
            '<div class="lc-stats">'
            f"<span>{self._caption_count} captions</span>"
            f"<span>{len(self._action_rows)} actions</span>"
            "</div>"
            f'<div class="lc-live-actions">{_live_action_cards_html(self._action_rows)}</div>'
            "</div>"
        )

    def _eval_caption_js(self, text: str) -> None:
        script = (
            f"const caption = document.getElementById({json.dumps(self.caption_id)});"
            f"if (caption) {{ caption.textContent = {json.dumps(text)}; }}"
        )
        try:
            from google.colab import output  # type: ignore

            output.eval_js(script)
        except Exception:
            try:
                from IPython.display import Javascript, display  # type: ignore

                display(Javascript(script))
            except Exception:
                pass

    def _publish_browser_state(self) -> None:
        payload = {
            "state": self._state,
            "detail": self._detail,
            "ready": self._ready,
            "recording": self._recording,
            "stopped": self._stopped,
        }
        script = (
            "window.fullDemoMicGate && "
            f"window.fullDemoMicGate[{json.dumps(self.widget_id)}] && "
            f"window.fullDemoMicGate[{json.dumps(self.widget_id)}].publish("
            f"{json.dumps(payload)}"
            ")"
        )
        try:
            from google.colab import output  # type: ignore

            output.eval_js(script)
        except Exception:
            try:
                from IPython.display import Javascript, display  # type: ignore

                display(Javascript(script))
            except Exception:
                pass

    def _gate_js(self) -> str:
        config = self.chunk_source.config
        callbacks = self.chunk_source.callbacks
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
        initial_state = {
            "state": self._state,
            "detail": self._detail,
            "ready": self._ready,
            "recording": self._recording,
            "stopped": self._stopped,
        }
        return f"""
(() => {{
  const widgetId = {json.dumps(self.widget_id)};
  const callbacks = {json.dumps(callbacks)};
  const config = {json.dumps(callback_config)};
  const initialState = {json.dumps(initial_state)};
  const install = (attempt = 0) => {{
    const root = document.getElementById(widgetId);
    if (!root) {{
      if (attempt < 100) window.setTimeout(() => install(attempt + 1), 100);
      return;
    }}
    const button = root.querySelector('[data-role="toggle"]');
    const progress = document.getElementById(`${{widgetId}}-progress`);
    const apiRoot = window.fullDemoMicGate = window.fullDemoMicGate || {{}};
    if (root.dataset.fullDemoMicBound === "1" && apiRoot[widgetId]) {{
      apiRoot[widgetId].publish(initialState);
      return;
    }}
    root.dataset.fullDemoMicBound = "1";
    const mic = () => window.liveCommerceMic || null;
    let ready = false;
    let recording = false;
    let paused = false;
    let stopped = false;
    const setVisual = () => {{
      if (!button) return;
      button.innerHTML = recording ? "&#10073;&#10073;" : "&#9654;";
      button.title = recording ? "Pause mic input" : "Enable or resume mic input";
      button.setAttribute("aria-label", button.title);
      button.disabled = stopped || !ready;
      if (progress) progress.style.width = recording ? "100%" : "0%";
    }};
    const emit = (name, payload) => {{
      const state = mic();
      if (state && typeof state.emitCallback === "function") {{
        state.emitCallback(name, payload || {{}});
      }}
    }};
    const startMic = () => {{
      if (!ready || stopped || recording) return;
      const state = mic();
      if (!state || typeof state.startBufferedRecording !== "function") return;
      recording = true;
      paused = false;
      state.startRequested = true;
      state.stopRequested = false;
      state.pauseRequested = false;
      emit(callbacks.start, {{started: true}});
      if (typeof state.resolveStart === "function") {{
        state.resolveStart(true);
        state.resolveStart = null;
      }}
      setVisual();
      if (state.bufferLoopActive) {{
        if (typeof state.resume === "function") state.resume();
        return;
      }}
      state.startBufferedRecording(
        config.maxMilliseconds,
        config.minMilliseconds,
        config.silenceMilliseconds,
        config.silenceThreshold,
        config.maxChunks,
        config.maxQueueChunks,
        callbacks.chunk,
        callbacks.done,
        callbacks.error
      ).then(result => {{
        if (!result || !result.started) {{
          recording = false;
          paused = false;
          setVisual();
          const reason = result && result.reason ? result.reason : "mic_not_started";
          emit(callbacks.error, {{error: reason}});
        }}
      }}).catch(error => {{
        recording = false;
        paused = false;
        setVisual();
        emit(callbacks.error, {{error: error && (error.message || String(error))}});
      }});
    }};
    const pauseMic = () => {{
      if (!recording) return;
      const state = mic();
      recording = false;
      paused = true;
      if (state && typeof state.pause === "function") state.pause();
      setVisual();
    }};
    if (button && !button.dataset.bound) {{
      button.dataset.bound = "1";
      button.addEventListener("click", () => {{
        if (recording) pauseMic();
        else startMic();
      }});
    }}
    apiRoot[widgetId] = {{
      publish(payload) {{
        const state = payload || {{}};
        ready = Boolean(state.ready);
        stopped = Boolean(state.stopped) || state.state === "stopped";
        if (state.state === "loading_asr") ready = false;
        if (state.state === "ready") ready = true;
        if (state.state === "stopped") {{
          recording = false;
          paused = false;
        }}
        setVisual();
      }}
    }};
    apiRoot[widgetId].publish(initialState);
  }};
  install();
}})();
"""


def _display_full_demo_ui_legacy(repo_root: Path = REPO_ROOT) -> None:
    """Display a single-notebook UI for the full live-commerce copilot demo."""
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        print("ipywidgets and IPython are required for this notebook UI.")
        print("Run: pip install ipywidgets")
        return

    session = FullDemoSession(repo_root)
    state: Dict[str, Any] = {
        "mode": "recording",
        "source": None,
        "running": False,
        "run_id": 0,
        "active_stop": None,
        "active_cancel": None,
    }

    display(HTML(_style_block()))
    title = widgets.HTML(
        """
        <div class="lc-header">
          <div>
            <div class="lc-eyebrow">TrueID Live Commerce Copilot</div>
            <h1>ASR + Commerce Actions Control Room</h1>
          </div>
          <div class="lc-status-pill">Notebook MVP</div>
        </div>
        """
    )

    upload = widgets.FileUpload(
        accept="audio/*",
        multiple=False,
        description="Upload audio",
        layout=widgets.Layout(width="100%"),
    )
    try:
        upload.icon = "upload"
    except Exception:
        pass
    upload.add_class("lc-upload-widget")
    refresh_button = widgets.Button(
        description="Refresh Catalog",
        icon="refresh",
        layout=widgets.Layout(width="170px", height="36px"),
    )

    output = widgets.Output()
    viewer = widgets.Output()
    catalog_output = widgets.Output()
    source_detail = widgets.VBox()
    selected_summary = widgets.HTML()

    mode_buttons: Dict[str, Any] = {
        "recording": widgets.Button(
            description="Recording",
            icon="file-audio-o",
            tooltip="Process the full audio first, then show captions and actions.",
            layout=widgets.Layout(width="168px", height="48px"),
        ),
        "live": widgets.Button(
            description="Live",
            icon="play-circle",
            tooltip="Run a real-time stream with start and stop controls.",
            layout=widgets.Layout(width="168px", height="48px"),
        ),
    }
    source_buttons: Dict[str, Any] = {}
    sample_cards = [
        _sample_source_card_widget(
            widgets,
            key="Audio 1 - Beauty",
            title="Audio 1",
            subtitle="Beauty live stream",
            icon="shopping-bag",
            source_buttons=source_buttons,
        ),
        _sample_source_card_widget(
            widgets,
            key="Audio 2 - Tech",
            title="Audio 2",
            subtitle="Tech live stream",
            icon="bolt",
            source_buttons=source_buttons,
        ),
    ]
    mic_tool = _tool_source_card_widget(
            widgets,
            key="mic",
            title="Mic",
            subtitle="Record or stream from browser mic",
            icon="microphone",
            source_buttons=source_buttons,
    )
    upload_tool = _upload_source_card_widget(widgets, upload)

    controls = widgets.VBox(
        [
            widgets.HTML('<div class="lc-panel-title">Studio Setup</div>'),
            widgets.HTML(
                '<div class="lc-copy">Choose a workflow, then pick the audio source. '
                "The ASR and chunking defaults are fixed for this demo.</div>"
            ),
            widgets.HBox(
                [mode_buttons["recording"], mode_buttons["live"]],
                layout=widgets.Layout(gap="10px", flex_flow="row wrap"),
            ),
            widgets.HTML('<div class="lc-subtitle">Audio source</div>'),
            widgets.GridBox(
                sample_cards,
                layout=widgets.Layout(
                    grid_template_columns="repeat(2, minmax(0, 1fr))",
                    grid_gap="12px",
                    width="100%",
                ),
            ),
            widgets.HTML('<div class="lc-subtitle">Your input</div>'),
            widgets.VBox(
                [upload_tool, mic_tool],
                layout=widgets.Layout(gap="8px", width="100%"),
            ),
            source_detail,
            selected_summary,
            widgets.HTML(
                '<div class="lc-defaults">'
                "<strong>Fixed ASR setup:</strong> OpenAI Whisper turbo, Thai, "
                "pause-aware chunks up to 15s, pause 0.3s, silence threshold 0.015, "
                "no chunk cap."
                "</div>"
            ),
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )
    controls.add_class("lc-operator-panel")

    def current_audio_path() -> Optional[Path]:
        source_key = str(state.get("source") or "")
        if source_key in SAMPLE_AUDIO:
            return SAMPLE_AUDIO[source_key]
        if source_key == "upload":
            return save_uploaded_audio(upload, session.upload_dir)
        return None

    def refresh_catalog(_: Any = None) -> None:
        session.save_catalog_files()
        with catalog_output:
            catalog_output.clear_output(wait=True)
            display(HTML(build_catalog_scene_html(session.catalog, session.promotions)))

    def selected_source_label() -> str:
        source_key = str(state.get("source") or "")
        if source_key == "upload":
            return "uploaded audio"
        if source_key == "mic":
            return "browser microphone"
        return source_key or "no source selected"

    def selected_mode_label() -> str:
        return "Live" if state["mode"] == "live" else "Recording"

    def refresh_selection_ui(render_idle: bool = True) -> None:
        for key, button in mode_buttons.items():
            button.button_style = "danger" if key == state["mode"] else ""
        for key, button in source_buttons.items():
            button.button_style = "danger" if key == state["source"] else ""
        try:
            upload.button_style = "danger" if state["source"] == "upload" else ""
        except Exception:
            pass

        if state.get("source") is None:
            selected_summary.value = (
                '<div class="lc-selected-summary">'
                "Choose an audio source to start the selected demo mode."
                "</div>"
            )
        else:
            selected_summary.value = (
                '<div class="lc-selected-summary">'
                f"<strong>{html.escape(selected_mode_label())}</strong> using "
                f"<strong>{html.escape(selected_source_label())}</strong>"
                "</div>"
            )

        if state["source"] == "upload":
            source_detail.children = [
                widgets.HTML(
                    '<div class="lc-upload-panel">'
                    '<div class="lc-source-icon">UP</div>'
                    '<div><strong>Upload selected</strong>'
                    '<div class="lc-muted">Choose a file from the upload tool above. '
                    "Recording starts after selection; live mode prepares stream controls.</div>"
                    "</div></div>"
                ),
            ]
        elif state["source"] == "mic":
            source_detail.children = [
                widgets.HTML(
                    '<div class="lc-source-detail">'
                    '<strong>Microphone source</strong><br>'
                    "Recording mode opens a confirm/cancel recorder in the showcase. "
                    "Live mode opens start/pause stream controls there."
                    "</div>"
                )
            ]
        elif state.get("source") is None:
            source_detail.children = [
                widgets.HTML(
                    '<div class="lc-source-detail">'
                    "Audio 1 and Audio 2 use cached recording outputs. Upload and mic "
                    "run through the selected ASR flow."
                    "</div>"
                )
            ]
        else:
            source_detail.children = [
                widgets.HTML(
                    f'<div class="lc-source-detail">{html.escape(selected_source_label())} '
                    "is bundled with the repo. Selecting this source starts the demo.</div>"
                )
            ]

        if render_idle:
            render_idle_viewer()

    def interrupt_active_run(clear_viewer: bool = False) -> None:
        state["run_id"] = int(state.get("run_id") or 0) + 1
        cancel_event = state.get("active_cancel")
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception:
                pass
        stop = state.get("active_stop")
        if callable(stop):
            threading.Thread(target=_call_safely, args=(stop,), daemon=True).start()
        state["active_stop"] = None
        state["active_cancel"] = None
        state["running"] = False
        if clear_viewer:
            with viewer:
                viewer.clear_output(wait=True)

    def set_mode(value: str) -> None:
        mode_changed = value != state["mode"]
        if value != state["mode"]:
            interrupt_active_run(clear_viewer=True)
            state["source"] = None
        state["mode"] = value
        refresh_selection_ui(render_idle=(not mode_changed and value != "live"))

    def set_source(value: str) -> None:
        interrupt_active_run(clear_viewer=True)
        state["source"] = value
        refresh_selection_ui(render_idle=state["mode"] != "live")
        if value != "upload":
            run_selected_source(force=True)

    for key, button in mode_buttons.items():
        button.on_click(lambda _button, key=key: set_mode(key))
    for key, button in source_buttons.items():
        button.on_click(lambda _button, key=key: set_source(key))

    def on_upload_change(change: Dict[str, Any]) -> None:
        if change.get("new"):
            interrupt_active_run(clear_viewer=True)
            state["source"] = "upload"
            refresh_selection_ui(render_idle=False)
            run_selected_source(force=True)

    upload.observe(on_upload_change, names="value")

    def render_idle_viewer() -> None:
        audio_path = current_audio_path()
        with viewer:
            viewer.clear_output(wait=True)
            display(
                HTML(
                    build_viewer_scene_html(
                        audio_path=audio_path,
                        captions=CaptionResult(
                            language=FULL_DEMO_LANGUAGE,
                            segments=[],
                            duration_seconds=None,
                        ),
                        actions=[],
                        mode_label=f"{selected_mode_label()} ready",
                        title="Ready to run ASR + commerce actions",
                    )
                )
            )

    def run_selected_source(force: bool = False) -> None:
        if state.get("running"):
            interrupt_active_run(clear_viewer=True)
        if state["source"] == "upload" and not upload.value:
            with output:
                output.clear_output(wait=True)
                print("Select an audio file to start.")
            return
        state["run_id"] = int(state.get("run_id") or 0) + 1
        run_id = int(state["run_id"])
        cancel_event = threading.Event()
        state["active_cancel"] = cancel_event
        state["running"] = True
        async_started = False
        try:
            async_started = run_demo(run_id, cancel_event)
        finally:
            if not async_started:
                if int(state.get("run_id") or 0) == run_id:
                    state["running"] = False
                    state["active_stop"] = None
                    state["active_cancel"] = None

    def run_demo(run_id: int, cancel_event: threading.Event) -> bool:
        with output:
            output.clear_output(wait=True)
            print(
                "Starting "
                f"{selected_mode_label().lower()} demo with {selected_source_label()}..."
            )
        with viewer:
            viewer.clear_output(wait=True)

        try:
            if state["mode"] == "recording":
                if state["source"] == "mic":
                    with viewer:
                        audio_path = record_colab_mic_clip(
                            session.upload_dir,
                            max_seconds=FULL_DEMO_CHUNK_SECONDS,
                        )
                else:
                    audio_path = current_audio_path()
                if audio_path is None:
                    raise RuntimeError("Choose a sample, upload an audio file, or use microphone.")
                with viewer:
                    viewer.clear_output(wait=True)
                    display(
                        HTML(
                            build_processing_scene_html(
                                title="Processing recording",
                                detail=(
                                    "Loading cached captions/actions"
                                    if state["source"] in SAMPLE_AUDIO
                                    else "Running OpenAI Whisper turbo and commerce actions"
                                ),
                            )
                        )
                    )
                summary, captions, actions = run_recording_pipeline(
                    session=session,
                    audio_path=audio_path,
                    source_key=str(state["source"]),
                    asr_provider=FULL_DEMO_ASR_PROVIDER,
                    asr_model=FULL_DEMO_ASR_MODEL,
                    use_cached=False,
                    dynamic_chunking=FULL_DEMO_DYNAMIC_CHUNKING,
                    chunk_seconds=FULL_DEMO_CHUNK_SECONDS,
                    pause_seconds=FULL_DEMO_PAUSE_SECONDS,
                    silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
                )
                render_summary(summary, output)
                with viewer:
                    viewer.clear_output(wait=True)
                    display(
                        HTML(
                            build_viewer_scene_html(
                                audio_path=audio_path,
                                captions=captions,
                                actions=actions,
                                mode_label="Recording mode",
                                title="Batch ASR + action preview",
                                show_audio_controls=True,
                            )
                        )
                    )
                state["last_summary"] = summary
                return False

            elif state["mode"] == "live":
                if state["source"] == "mic":
                    start_live_mic_worker(run_id, cancel_event)
                    return True

                audio_path = current_audio_path()
                if audio_path is None:
                    raise RuntimeError("Live replay requires sample audio or uploaded audio.")
                start_live_file_worker(audio_path, run_id, cancel_event)
                return True

        except Exception as exc:
            with output:
                print(f"Demo failed: {exc}")
            return False

        return False

    def start_live_file_worker(
        audio_path: Path,
        run_id: int,
        cancel_event: threading.Event,
    ) -> None:
        source_key = str(state["source"])
        config = RealtimeAudioFileDemoConfig(
            audio_path=audio_path,
            chunk_seconds=FULL_DEMO_CHUNK_SECONDS,
            dynamic_chunking=FULL_DEMO_DYNAMIC_CHUNKING,
            min_chunk_seconds=FULL_DEMO_MIN_CHUNK_SECONDS,
            pause_seconds=FULL_DEMO_PAUSE_SECONDS,
            silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
            max_chunks=None,
            language=FULL_DEMO_LANGUAGE,
            asr_provider=FULL_DEMO_ASR_PROVIDER,
            asr_model=FULL_DEMO_ASR_MODEL,
            install_asr_deps=False,
            output_dir=session.output_dir / f"live_replay_{_safe_output_name(source_key)}",
            catalog_path=session.catalog_path,
            promotions_path=session.promotions_path,
        )
        windows = list(
            iter_audio_windows(
                audio_path,
                int(FULL_DEMO_CHUNK_SECONDS),
                dynamic_chunking=FULL_DEMO_DYNAMIC_CHUNKING,
                min_chunk_seconds=FULL_DEMO_MIN_CHUNK_SECONDS,
                pause_seconds=FULL_DEMO_PAUSE_SECONDS,
                silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
            )
        )
        playback_gate: Optional[FullDemoPlaybackGate] = None

        with output:
            output.clear_output(wait=True)
            print(
                f"Live stream panel is starting for {selected_source_label()}. "
                "Use Start/Pause inside the showcase."
            )
        with viewer:
            viewer.clear_output(wait=True)
            try:
                playback_gate = FullDemoPlaybackGate(
                    widgets=widgets,
                    audio_path=audio_path,
                    windows=windows,
                    source_label=selected_source_label(),
                    poll_seconds=config.playback_poll_seconds,
                )
                display(playback_gate.ui)
                state["active_stop"] = playback_gate.stop
            except Exception as exc:
                print(f"Full UI live controls unavailable, using wall-clock replay: {exc}")
                playback_gate = None

        def worker() -> None:
            try:
                asr_engine = None
                if not cancel_event.is_set():
                    if playback_gate is not None:
                        playback_gate.publish(
                            {
                                "state": "loading_asr",
                                "status": (
                                    "Preparing OpenAI Whisper turbo. "
                                    "First load can take a moment."
                                ),
                                "totalChunks": len(windows),
                                "processedChunks": 0,
                                "queueDepth": 0,
                            }
                        )
                    asr_engine = _build_file_stream_asr(config)
                summary = run_realtime_audio_file_demo(
                    config,
                    playback_gate=playback_gate,
                    asr=asr_engine,
                    windows=windows,
                    cancel_event=cancel_event,
                )
                if int(state.get("run_id") or 0) == run_id:
                    state["last_summary"] = summary
                    render_summary(summary, output)
            except Exception as exc:
                if int(state.get("run_id") or 0) == run_id:
                    with output:
                        print(f"Live file demo failed: {exc}")
            finally:
                if int(state.get("run_id") or 0) == run_id:
                    state["running"] = False
                    state["active_stop"] = None
                    state["active_cancel"] = None

        threading.Thread(target=worker, daemon=True).start()

    def start_live_mic_worker(run_id: int, cancel_event: threading.Event) -> None:
        config = LiveMicDemoConfig(
            chunk_seconds=FULL_DEMO_CHUNK_SECONDS,
            min_chunk_seconds=FULL_DEMO_MIN_CHUNK_SECONDS,
            pause_seconds=FULL_DEMO_PAUSE_SECONDS,
            silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
            max_chunks=None,
            language=FULL_DEMO_LANGUAGE,
            asr_provider=FULL_DEMO_ASR_PROVIDER,
            asr_model=FULL_DEMO_ASR_MODEL,
            install_asr_deps=False,
            show_debug_panel=False,
            continuous_recording=True,
            output_dir=session.output_dir / "live_mic",
            catalog_path=session.catalog_path,
            promotions_path=session.promotions_path,
        )

        with output:
            output.clear_output(wait=True)
            print("Live mic panel is starting. Use Start/Pause inside the showcase.")
        with viewer:
            viewer.clear_output(wait=True)
            try:
                mic_source = BrowserMicChunkSource(config)
                mic_source.install_recorder()
                mic_gate = FullDemoMicGate(widgets, mic_source)
                display(mic_gate.ui)
                mic_gate.install()
                mic_gate.publish(
                    "loading_asr",
                    "Preparing OpenAI Whisper turbo. Mic input will unlock when ASR is ready.",
                )
                state["active_stop"] = mic_gate.stop
            except Exception as exc:
                print(f"Browser mic controls unavailable: {exc}")
                mic_source = None

        def worker() -> None:
            try:
                asr_engine = None
                if not cancel_event.is_set():
                    asr_engine = _build_live_asr(config)
                    if mic_source is not None:
                        mic_gate.publish(
                            "ready",
                            "ASR is ready. Enable the mic when you want to stream.",
                        )
                summary = run_colab_live_mic_demo(
                    config,
                    chunk_source=mic_source,
                    asr=asr_engine,
                    result_sink=mic_gate.update_results if mic_source is not None else None,
                    cancel_event=cancel_event,
                )
                if int(state.get("run_id") or 0) == run_id:
                    state["last_summary"] = summary
                    render_summary(summary, output)
            except Exception as exc:
                if int(state.get("run_id") or 0) == run_id:
                    with output:
                        print(f"Live mic demo failed: {exc}")
            finally:
                if int(state.get("run_id") or 0) == run_id:
                    state["running"] = False
                    state["active_stop"] = None
                    state["active_cancel"] = None

        threading.Thread(target=worker, daemon=True).start()

    refresh_button.on_click(refresh_catalog)

    form_ui = build_catalog_forms(session, refresh_catalog)
    form_ui.add_class("lc-form-panel")
    demo_page = widgets.HBox(
        [
            widgets.VBox(
                [controls, output],
                layout=widgets.Layout(width="36%", min_width="320px", gap="10px"),
            ),
            widgets.VBox([viewer], layout=widgets.Layout(width="64%")),
        ],
        layout=widgets.Layout(align_items="stretch", gap="14px"),
    )
    catalog_page = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.HTML(
                        '<div><div class="lc-panel-title">Catalog and Promotions</div>'
                        '<div class="lc-copy">Review demo commerce data or add temporary '
                        "products and promotions for this notebook session.</div></div>"
                    ),
                    refresh_button,
                ],
                layout=widgets.Layout(
                    justify_content="space-between",
                    align_items="center",
                    margin="0 0 10px 0",
                ),
            ),
            widgets.HBox(
                [
                    widgets.VBox([catalog_output], layout=widgets.Layout(width="62%")),
                    widgets.VBox([form_ui], layout=widgets.Layout(width="38%")),
                ],
                layout=widgets.Layout(align_items="flex-start", gap="14px"),
            ),
        ],
        layout=widgets.Layout(width="100%"),
    )
    tabs = widgets.Tab(children=[demo_page, catalog_page])
    tabs.set_title(0, "ASR + Actions")
    tabs.set_title(1, "Catalog")
    app = widgets.VBox(
        [
            title,
            tabs,
        ]
    )
    display(app)
    refresh_selection_ui(render_idle=False)
    with viewer:
        viewer.clear_output(wait=True)
    refresh_catalog()


def display_full_demo_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display the simplified recording-only notebook demo UI."""
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        return

    session = _get_full_demo_session(repo_root)
    state: Dict[str, Any] = {"running": False, "source": None}
    display(HTML(_style_block()))

    source_picker = widgets.HBox(layout=widgets.Layout(gap="10px", flex_flow="row wrap"))
    status_html = widgets.HTML()
    viewer = widgets.Output()
    upload = widgets.FileUpload(
        accept="audio/*",
        multiple=False,
        description="Upload audio",
        layout=widgets.Layout(width="180px", height="44px"),
    )
    try:
        upload.icon = "upload"
    except Exception:
        pass

    audio1_button = widgets.Button(
        description="Audio 1",
        icon="play",
        button_style="danger",
        tooltip="Beauty sample",
        layout=widgets.Layout(width="132px", height="44px"),
    )
    audio2_button = widgets.Button(
        description="Audio 2",
        icon="play",
        button_style="danger",
        tooltip="Tech sample",
        layout=widgets.Layout(width="132px", height="44px"),
    )
    mic_button = widgets.Button(
        description="Mic",
        icon="microphone",
        layout=widgets.Layout(width="120px", height="44px"),
    )
    source_picker.children = [audio1_button, audio2_button, upload, mic_button]

    controls = widgets.VBox(
        [
            widgets.HTML('<div class="lc-panel-title">Recording demo</div>'),
            source_picker,
            status_html,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )
    controls.add_class("lc-operator-panel")
    app = widgets.HBox(
        [
            widgets.VBox([controls], layout=widgets.Layout(width="32%", min_width="280px")),
            widgets.VBox([viewer], layout=widgets.Layout(width="68%")),
        ],
        layout=widgets.Layout(align_items="flex-start", gap="14px"),
    )
    display(app)

    def set_busy(is_busy: bool) -> None:
        state["running"] = is_busy
        source_picker.layout.display = "none" if is_busy else ""
        upload.disabled = is_busy
        audio1_button.disabled = is_busy
        audio2_button.disabled = is_busy
        mic_button.disabled = is_busy

    def set_status(kind: str, title: str, detail: str = "") -> None:
        status_html.value = _status_card_html(kind, title, detail)

    def clear_status() -> None:
        status_html.value = ""

    def render_idle() -> None:
        with viewer:
            viewer.clear_output(wait=True)
            display(HTML(_recording_empty_state_html()))

    def process_audio(audio_path: Path, source_key: str, source_label: str) -> None:
        if state.get("running"):
            return
        set_busy(True)
        clear_status()
        with viewer:
            viewer.clear_output(wait=True)
            display(HTML(build_processing_scene_html("Processing recording", source_label)))
        try:
            summary, captions, actions = run_recording_pipeline(
                session=session,
                audio_path=audio_path,
                source_key=source_key,
                asr_provider=FULL_DEMO_ASR_PROVIDER,
                asr_model=FULL_DEMO_ASR_MODEL,
                use_cached=False,
                dynamic_chunking=FULL_DEMO_DYNAMIC_CHUNKING,
                chunk_seconds=FULL_DEMO_CHUNK_SECONDS,
                pause_seconds=FULL_DEMO_PAUSE_SECONDS,
                silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
            )
            state["last_summary"] = summary
            with viewer:
                viewer.clear_output(wait=True)
                display(
                    HTML(
                        build_viewer_scene_html(
                            audio_path=audio_path,
                            captions=captions,
                            actions=actions,
                            mode_label="Recording",
                            title="ASR + action preview",
                            show_audio_controls=True,
                            catalog=session.catalog,
                            product_images=session.product_images,
                        )
                    )
                )
        except Exception as exc:
            set_status("error", "Demo failed", str(exc))
        finally:
            set_busy(False)

    def process_mic() -> None:
        if state.get("running"):
            return
        set_busy(True)
        clear_status()
        try:
            with viewer:
                viewer.clear_output(wait=True)
                audio_path = record_colab_mic_clip(
                    session.upload_dir,
                    max_seconds=FULL_DEMO_CHUNK_SECONDS,
                )
            set_busy(False)
            process_audio(audio_path, "mic", "Browser microphone recording")
        except Exception as exc:
            set_status("error", "Mic recording failed", str(exc))
            render_idle()
        finally:
            if state.get("running"):
                set_busy(False)

    def on_upload_change(change: Dict[str, Any]) -> None:
        if not change.get("new") or state.get("running"):
            return
        audio_path = save_uploaded_audio(upload, session.upload_dir)
        if audio_path is None:
            set_status("error", "Upload failed", "No audio file was available.")
            return
        process_audio(audio_path, "upload", audio_path.name)

    audio1_button.on_click(
        lambda _button: process_audio(
            SAMPLE_AUDIO["Audio 1 - Beauty"],
            "Audio 1 - Beauty",
            "Audio 1 - Beauty",
        )
    )
    audio2_button.on_click(
        lambda _button: process_audio(
            SAMPLE_AUDIO["Audio 2 - Tech"],
            "Audio 2 - Tech",
            "Audio 2 - Tech",
        )
    )
    mic_button.on_click(lambda _button: process_mic())
    upload.observe(on_upload_change, names="value")
    render_idle()


def display_catalog_manager_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display the separate catalog/promotion manager notebook UI."""
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        return

    session = _get_full_demo_session(repo_root)
    state: Dict[str, str] = {"view": "products"}
    display(HTML(_style_block()))

    product_button = widgets.Button(
        description="Products",
        icon="shopping-bag",
        button_style="danger",
        layout=widgets.Layout(width="150px", height="42px"),
    )
    promo_button = widgets.Button(
        description="Promotions",
        icon="tags",
        layout=widgets.Layout(width="150px", height="42px"),
    )
    add_button = widgets.Button(
        description="",
        icon="plus",
        button_style="success",
        tooltip="Add item",
        layout=widgets.Layout(width="44px", height="40px"),
    )
    list_output = widgets.Output()
    form_box = widgets.VBox(layout=widgets.Layout(width="100%"))

    def render_list() -> None:
        product_button.button_style = "danger" if state["view"] == "products" else ""
        promo_button.button_style = "danger" if state["view"] == "promotions" else ""
        with list_output:
            list_output.clear_output(wait=True)
            display(
                HTML(
                    build_catalog_scene_html(
                        session.catalog,
                        session.promotions,
                        view=state["view"],
                        product_images=session.product_images,
                    )
                )
            )

    def show_product_form() -> None:
        product_fields = {
            "sku": widgets.Text(value=f"SKU{len(session.catalog) + 1:03d}", description="SKU"),
            "product_name": widgets.Text(value="Live Demo Product", description="Name"),
            "brand": widgets.Text(value="DemoBrand", description="Brand"),
            "category": widgets.Text(value="Skincare", description="Category"),
            "price": widgets.IntText(value=399, description="Price"),
            "discount_price": widgets.IntText(value=299, description="Live price"),
            "stock": widgets.IntText(value=50, description="Stock"),
            "description": widgets.Textarea(value="Demo product added from notebook UI", description="Desc"),
            "tags": widgets.Text(value="demo;skincare", description="Tags"),
            "compatible_with": widgets.Text(value="SKU001", description="Compatible"),
            "deeplink": widgets.Text(value="https://trueid.example/live/product", description="Link"),
        }
        image_upload = widgets.FileUpload(
            accept="image/*",
            multiple=False,
            description="Upload image",
            layout=widgets.Layout(width="220px"),
        )
        save = widgets.Button(description="Add product", icon="check", button_style="success")
        cancel = widgets.Button(description="Cancel", icon="times")

        def on_save(_: Any) -> None:
            item = ProductCatalogItem.from_dict(
                {key: field.value for key, field in product_fields.items()}
            )
            image_path = save_uploaded_image(
                image_upload,
                session.product_image_dir,
                item.sku,
            )
            session.add_product(item, image_path=image_path)
            form_box.children = []
            render_list()

        save.on_click(on_save)
        cancel.on_click(lambda _button: setattr(form_box, "children", []))
        form_box.children = [
            widgets.VBox(
                [
                    widgets.HTML('<div class="lc-form-title">Add product</div>'),
                    *product_fields.values(),
                    image_upload,
                    widgets.HBox([save, cancel], layout=widgets.Layout(gap="8px")),
                ]
            )
        ]

    def show_promo_form() -> None:
        promo_fields = {
            "promo_code": widgets.Text(value="NEWLIVE", description="Code"),
            "promo_description": widgets.Text(value="Notebook-added live promo", description="Desc"),
            "discount_type": widgets.Dropdown(
                options=["percent", "fixed_amount"],
                value="percent",
                description="Type",
            ),
            "discount_value": widgets.IntText(value=10, description="Value"),
            "live_only": widgets.Checkbox(value=True, description="Live only", indent=False),
            "eligible_categories": widgets.Text(value="Skincare", description="Categories"),
            "eligible_tags": widgets.Text(value="skincare", description="Tags"),
            "eligible_skus": widgets.Text(value="", description="SKUs"),
            "required_min_stock": widgets.IntText(value=1, description="Min stock"),
        }
        save = widgets.Button(description="Add promotion", icon="check", button_style="success")
        cancel = widgets.Button(description="Cancel", icon="times")

        def on_save(_: Any) -> None:
            promotion = Promotion.from_dict(
                {key: field.value for key, field in promo_fields.items()}
            )
            session.add_promotion(promotion)
            form_box.children = []
            render_list()

        save.on_click(on_save)
        cancel.on_click(lambda _button: setattr(form_box, "children", []))
        form_box.children = [
            widgets.VBox(
                [
                    widgets.HTML('<div class="lc-form-title">Add promotion</div>'),
                    *promo_fields.values(),
                    widgets.HBox([save, cancel], layout=widgets.Layout(gap="8px")),
                ]
            )
        ]

    def switch_view(view: str) -> None:
        state["view"] = view
        form_box.children = []
        render_list()

    product_button.on_click(lambda _button: switch_view("products"))
    promo_button.on_click(lambda _button: switch_view("promotions"))
    add_button.on_click(
        lambda _button: show_product_form()
        if state["view"] == "products"
        else show_promo_form()
    )

    display(
        widgets.VBox(
            [
                widgets.HBox(
                    [product_button, promo_button],
                    layout=widgets.Layout(gap="10px"),
                ),
                list_output,
                widgets.HBox([add_button], layout=widgets.Layout(justify_content="center")),
                form_box,
            ],
            layout=widgets.Layout(width="100%", gap="12px"),
        )
    )
    render_list()


def _sample_source_card_widget(
    widgets: Any,
    key: str,
    title: str,
    subtitle: str,
    icon: str,
    source_buttons: Dict[str, Any],
) -> Any:
    thumb_html = widgets.HTML(
        f"""
        <div class="lc-sample-thumb">
          <div class="lc-sample-badge"><span class="fa fa-{html.escape(icon)}"></span></div>
          <div class="lc-sample-host">
            <div class="lc-sample-head"></div>
            <div class="lc-sample-body"></div>
          </div>
          <div class="lc-sample-strip">{html.escape(subtitle)}</div>
        </div>
        """
    )
    button = widgets.Button(
        description=title,
        icon=icon,
        tooltip=subtitle,
        layout=widgets.Layout(width="100%", height="42px"),
    )
    button.add_class("lc-source-button")
    source_buttons[key] = button
    card = widgets.VBox(
        [
            thumb_html,
            button,
            widgets.HTML(f'<div class="lc-source-caption">{html.escape(subtitle)}</div>'),
        ],
        layout=widgets.Layout(width="100%"),
    )
    card.add_class("lc-sample-card")
    return card


def _tool_source_card_widget(
    widgets: Any,
    key: str,
    title: str,
    subtitle: str,
    icon: str,
    source_buttons: Dict[str, Any],
) -> Any:
    button = widgets.Button(
        description=title,
        icon=icon,
        tooltip=subtitle,
        layout=widgets.Layout(width="132px", height="40px"),
    )
    button.add_class("lc-tool-button")
    source_buttons[key] = button
    card = widgets.HBox(
        [
            widgets.HTML(
                f'<div class="lc-tool-icon"><span class="fa fa-{html.escape(icon)}"></span></div>'
            ),
            widgets.VBox(
                [
                    widgets.HTML(f'<strong class="lc-tool-title">{html.escape(title)}</strong>'),
                    widgets.HTML(f'<div class="lc-muted">{html.escape(subtitle)}</div>'),
                ],
                layout=widgets.Layout(flex="1 1 auto"),
            ),
            button,
        ],
        layout=widgets.Layout(width="100%", align_items="center"),
    )
    card.add_class("lc-tool-card")
    return card


def _upload_source_card_widget(widgets: Any, upload_widget: Any) -> Any:
    card = widgets.HBox(
        [
            widgets.HTML('<div class="lc-tool-icon"><span class="fa fa-upload"></span></div>'),
            widgets.VBox(
                [
                    widgets.HTML('<strong class="lc-tool-title">Upload</strong>'),
                    widgets.HTML('<div class="lc-muted">Choose a local audio file</div>'),
                ],
                layout=widgets.Layout(flex="1 1 auto"),
            ),
            widgets.Box([upload_widget], layout=widgets.Layout(width="150px")),
        ],
        layout=widgets.Layout(width="100%", align_items="center"),
    )
    card.add_class("lc-tool-card")
    return card


def run_recording_pipeline(
    session: FullDemoSession,
    audio_path: Path,
    source_key: str,
    asr_provider: str,
    asr_model: str,
    use_cached: bool,
    dynamic_chunking: bool,
    chunk_seconds: float,
    pause_seconds: float,
    silence_threshold: float,
) -> Tuple[Dict[str, Any], CaptionResult, List[CommerceAction]]:
    cached_path = SAMPLE_CACHED_TRANSCRIPTS.get(source_key)
    cached_actions_path = SAMPLE_CACHED_ACTIONS.get(source_key)
    output_dir = session.output_dir / "recording"
    if cached_path and cached_actions_path and cached_path.exists() and cached_actions_path.exists():
        captions = load_cached_transcript(cached_path)
        actions = _load_actions(cached_actions_path)
        paths = write_caption_outputs(captions, output_dir)
        actions_path = output_dir / "commerce_actions.json"
        html_path = output_dir / "commerce_actions_timeline.html"
        save_commerce_actions(actions, actions_path)
        save_action_timeline_html(actions, html_path, audio_path=audio_path)
        source_mode = "recording_cache"
    else:
        settings = CaptioningSettings(
            mode=asr_provider,
            language="th",
            audio_path=audio_path,
            asr_provider=asr_provider,
            asr_model=asr_model,
            whisper_model=asr_model,
            asr_chunk_length_seconds=int(chunk_seconds),
            asr_dynamic_chunking=dynamic_chunking,
            asr_min_chunk_seconds=FULL_DEMO_MIN_CHUNK_SECONDS,
            asr_pause_seconds=pause_seconds,
            asr_silence_threshold=silence_threshold,
            asr_max_new_tokens=256,
            allow_cached_fallback=False,
        )
        captions = CaptioningEngine(settings).transcribe()
        paths = write_caption_outputs(captions, output_dir)
        actions = generate_commerce_actions(captions, session.catalog, session.promotions)
        actions_path = output_dir / "commerce_actions.json"
        html_path = output_dir / "commerce_actions_timeline.html"
        save_commerce_actions(actions, actions_path)
        save_action_timeline_html(actions, html_path, audio_path=audio_path)
        source_mode = asr_provider

    metrics = caption_metrics(captions)
    return (
        {
            "mode": "recording",
            "source_mode": source_mode,
            "audio_path": str(audio_path),
            "caption_count": len(captions.segments),
            "action_count": len(actions),
            "metrics": metrics,
            "outputs": {
                "captions_json": str(paths["json"]),
                "actions_json": str(actions_path),
                "timeline_html": str(html_path),
            },
        },
        captions,
        actions,
    )


def run_live_showcase_pipeline(
    session: FullDemoSession,
    audio_path: Path,
    source_key: str,
    asr_provider: str,
    asr_model: str,
) -> Tuple[Dict[str, Any], CaptionResult, List[CommerceAction]]:
    cached_path = SAMPLE_CACHED_TRANSCRIPTS.get(source_key)
    cached_actions_path = SAMPLE_CACHED_ACTIONS.get(source_key)
    if cached_path and cached_actions_path and cached_path.exists() and cached_actions_path.exists():
        captions = load_cached_transcript(cached_path)
        actions = _load_actions(cached_actions_path)
        source_mode = "live_cache"
    else:
        _, captions, actions = run_recording_pipeline(
            session=session,
            audio_path=audio_path,
            source_key=source_key,
            asr_provider=asr_provider,
            asr_model=asr_model,
            use_cached=False,
            dynamic_chunking=FULL_DEMO_DYNAMIC_CHUNKING,
            chunk_seconds=FULL_DEMO_CHUNK_SECONDS,
            pause_seconds=FULL_DEMO_PAUSE_SECONDS,
            silence_threshold=FULL_DEMO_SILENCE_THRESHOLD,
        )
        source_mode = asr_provider

    return (
        {
            "mode": "live_showcase",
            "source_mode": source_mode,
            "audio_path": str(audio_path),
            "caption_count": len(captions.segments),
            "action_count": len(actions),
            "metrics": caption_metrics(captions),
        },
        captions,
        actions,
    )


def build_catalog_forms(session: FullDemoSession, refresh_callback: Any) -> Any:
    import ipywidgets as widgets  # type: ignore

    product_fields = {
        "sku": widgets.Text(value="SKU009", description="SKU"),
        "product_name": widgets.Text(value="Live Demo Product", description="Name"),
        "brand": widgets.Text(value="DemoBrand", description="Brand"),
        "category": widgets.Text(value="Skincare", description="Category"),
        "price": widgets.IntText(value=399, description="Price"),
        "discount_price": widgets.IntText(value=299, description="Live price"),
        "stock": widgets.IntText(value=50, description="Stock"),
        "description": widgets.Textarea(value="Demo product added from notebook UI", description="Desc"),
        "tags": widgets.Text(value="demo;skincare", description="Tags"),
        "compatible_with": widgets.Text(value="SKU001", description="Compatible"),
        "deeplink": widgets.Text(value="https://trueid.example/live/product/SKU009", description="Link"),
    }
    promo_fields = {
        "promo_code": widgets.Text(value="NEWLIVE", description="Code"),
        "promo_description": widgets.Text(value="Notebook-added live promo", description="Desc"),
        "discount_type": widgets.Dropdown(
            options=["percent", "fixed_amount"],
            value="percent",
            description="Type",
        ),
        "discount_value": widgets.IntText(value=10, description="Value"),
        "live_only": widgets.Checkbox(value=True, description="Live only", indent=False),
        "eligible_categories": widgets.Text(value="Skincare", description="Categories"),
        "eligible_tags": widgets.Text(value="skincare", description="Tags"),
        "eligible_skus": widgets.Text(value="SKU009", description="SKUs"),
        "required_min_stock": widgets.IntText(value=1, description="Min stock"),
    }
    add_product = widgets.Button(description="Add Product", icon="plus", button_style="success")
    add_promo = widgets.Button(description="Add Promotion", icon="plus", button_style="success")

    def on_add_product(_: Any) -> None:
        item = ProductCatalogItem.from_dict(
            {
                key: field.value
                for key, field in product_fields.items()
            }
        )
        session.add_product(item)
        refresh_callback()

    def on_add_promo(_: Any) -> None:
        promotion = Promotion.from_dict(
            {
                key: field.value
                for key, field in promo_fields.items()
            }
        )
        session.add_promotion(promotion)
        refresh_callback()

    add_product.on_click(on_add_product)
    add_promo.on_click(on_add_promo)

    accordion = widgets.Accordion(
        children=[
            widgets.VBox([*product_fields.values(), add_product]),
            widgets.VBox([*promo_fields.values(), add_promo]),
        ]
    )
    accordion.set_title(0, "Add Product")
    accordion.set_title(1, "Add Promotion")
    return accordion


def render_summary(summary: Dict[str, Any], output: Any) -> None:
    with output:
        output.clear_output(wait=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_catalog_scene_html(
    catalog: Sequence[ProductCatalogItem],
    promotions: Sequence[Promotion],
    view: str = "products",
    product_images: Optional[Dict[str, str]] = None,
) -> str:
    product_images = product_images or {}
    normalized_view = (view or "products").lower()
    if normalized_view == "promotions":
        promo_cards = "\n".join(_promo_card_html(promotion) for promotion in promotions)
        return f"""
<div class="lc-catalog-shell">
  <div class="lc-catalog-heading">
    <div class="lc-panel-title">Promotions</div>
    <span>{len(promotions)} active rules</span>
  </div>
  <div class="lc-card-grid">{promo_cards}</div>
</div>
"""

    product_cards = "\n".join(
        _product_card_html(item, product_images.get(item.sku)) for item in catalog
    )
    return f"""
<div class="lc-catalog-shell">
  <div class="lc-catalog-heading">
    <div class="lc-panel-title">Products</div>
    <span>{len(catalog)} catalog items</span>
  </div>
  <div class="lc-card-grid">{product_cards}</div>
</div>
"""


def build_live_ready_scene_html(
    audio_path: Optional[Path],
    source_label: str,
    is_mic: bool = False,
) -> str:
    source_note = (
        "Microphone input will stream into a queue. Stop closes the mic while queued chunks finish."
        if is_mic
        else "Audio will be released to ASR by stream time. Pause holds new chunks until the stream resumes."
    )
    audio_note = (
        "Browser mic source"
        if is_mic
        else f"Prepared source: {html.escape(audio_path.name if audio_path else source_label)}"
    )
    return f"""
<div class="lc-viewer">
  <div class="lc-video">
    <div class="lc-video-art">
      <div class="lc-video-badge">Live ready</div>
      <div class="lc-host-frame">
        <div class="lc-host-head"></div>
        <div class="lc-host-body"></div>
      </div>
      <div class="lc-video-caption">Press Start below to begin live ASR and commerce actions.</div>
    </div>
    <div class="lc-no-audio">{audio_note}</div>
  </div>
  <div class="lc-action-rail">
    <div class="lc-panel-title">Live stream controls</div>
    <div class="lc-stats">
      <span>{html.escape(source_label)}</span>
      <span>OpenAI Whisper turbo</span>
    </div>
    <div class="lc-source-detail">{source_note}</div>
    <div class="lc-live-actions">
      <div class="lc-action-card">
        <div class="lc-action-type">WAITING</div>
        <strong>Stream is ready</strong>
        <div class="lc-muted">Start keeps this showcase active while ASR/action chunks are processed.</div>
      </div>
    </div>
  </div>
</div>
"""


def _live_action_cards_html(action_rows: Sequence[Dict[str, Any]]) -> str:
    if not action_rows:
        return '<div class="lc-action-card"><div class="lc-muted">No actions emitted yet.</div></div>'
    cards = []
    for action in action_rows[-6:]:
        action_type = html.escape(str(action.get("action") or "ACTION"))
        title = html.escape(str(action.get("title") or _human_action_title(action_type)))
        skus = html.escape(str(action.get("skus") or "-"))
        timestamp = html.escape(str(action.get("time") or "0.00"))
        cards.append(
            '<div class="lc-action-card">'
            f'<div class="lc-action-type">{action_type}</div>'
            f"<strong>{title}</strong>"
            f"<div>{skus}</div>"
            f"<small>{timestamp}s</small>"
            "</div>"
        )
    return "".join(cards)


def _human_action_title(action_type: str) -> str:
    return " ".join(
        word.upper() if word in {"asr", "sku"} else word.capitalize()
        for word in str(action_type).lower().replace("_", " ").split()
    )


def build_processing_scene_html(title: str, detail: str) -> str:
    return f"""
<div class="lc-processing-panel">
  <div>
    <div class="lc-panel-title">{html.escape(title)}</div>
    <div class="lc-muted">{html.escape(detail)}</div>
  </div>
  <div class="lc-progress-bar"><div></div></div>
  <div class="lc-processing-steps">
    <span>ASR</span>
    <span>Caption validation</span>
    <span>Commerce action extraction</span>
  </div>
</div>
"""


def _status_card_html(kind: str, title: str, detail: str = "") -> str:
    return (
        f'<div class="lc-status-card lc-status-{html.escape(kind)}">'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span>"
        "</div>"
    )


def _recording_empty_state_html() -> str:
    return """
<div class="lc-recording-empty">
  <div class="lc-empty-mark">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 7h16v10H4z"/>
      <path d="M8 11h8"/>
      <path d="M10 15h4"/>
    </svg>
  </div>
  <div>
    <div class="lc-panel-title">Choose an audio source</div>
    <div class="lc-muted">Captions and commerce actions appear here after the recording is processed.</div>
  </div>
</div>
"""


def build_viewer_scene_html(
    audio_path: Optional[Path],
    captions: CaptionResult,
    actions: Sequence[CommerceAction],
    mode_label: str,
    title: str,
    show_audio_controls: bool = True,
    catalog: Optional[Sequence[ProductCatalogItem]] = None,
    product_images: Optional[Dict[str, str]] = None,
) -> str:
    segments = [segment.to_dict() for segment in captions.segments]
    action_payload = [action.to_dict() for action in actions]
    product_payload = _product_map_payload(catalog or [], product_images)
    audio_html = ""
    if audio_path and audio_path.exists() and show_audio_controls:
        audio_html = (
            f'<audio class="lc-audio" controls preload="metadata" '
            f'src="{_audio_data_uri(audio_path)}"></audio>'
        )
    elif audio_path and audio_path.exists():
        audio_html = '<div class="lc-no-audio">Playback controls are hidden for live mode</div>'
    else:
        audio_html = '<div class="lc-no-audio">No playback audio for this run</div>'

    return f"""
<div class="lc-viewer" id="viewer-{abs(hash(title))}">
  <div class="lc-video">
    <div class="lc-video-art">
      <div class="lc-video-badge">{html.escape(mode_label)}</div>
      <div class="lc-host-frame">
        <div class="lc-host-head"></div>
        <div class="lc-host-body"></div>
      </div>
      <div class="lc-video-caption" data-role="caption">Waiting for playback...</div>
    </div>
    {audio_html}
  </div>
  <div class="lc-action-rail">
    <div class="lc-panel-title">{html.escape(title)}</div>
    <div class="lc-stats">
      <span>{len(segments)} captions</span>
      <span>{len(actions)} actions</span>
    </div>
    <div data-role="actions" class="lc-live-actions"></div>
  </div>
</div>
<script>
(() => {{
  const root = document.currentScript.previousElementSibling;
  const captions = {json.dumps(segments, ensure_ascii=False)};
  const actions = {json.dumps(action_payload, ensure_ascii=False)};
  const products = {json.dumps(product_payload, ensure_ascii=False)};
  const audio = root.querySelector("audio");
  const captionEl = root.querySelector('[data-role="caption"]');
  const actionsEl = root.querySelector('[data-role="actions"]');
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }}[ch]));
  const productVisual = sku => products[sku]?.visual || `<div class="lc-action-product-thumb"><span>${{escapeHtml(String(sku || "?").slice(-2))}}</span></div>`;
  const productLine = sku => {{
    const product = products[sku] || {{}};
    const name = product.product_name || sku;
    const price = product.discount_price ? `THB ${{product.discount_price}}` : "";
    const original = product.price && product.price !== product.discount_price ? `<s>${{product.price}}</s>` : "";
    return `
      <div class="lc-action-product-row">
        ${{productVisual(sku)}}
        <div>
          <strong>${{escapeHtml(name)}}</strong>
          <div class="lc-muted">${{escapeHtml(product.brand || product.category || sku)}}</div>
          <div class="lc-action-price">${{price}} ${{original}}</div>
        </div>
      </div>`;
  }};
  const renderAction = (action) => {{
    const payload = action.display_payload || {{}};
    const actionType = action.action_type || "ACTION";
    const skus = action.skus || [];
    const time = Number(action.timestamp || 0).toFixed(2);
    if (actionType === "PIN_PRODUCT_CARD") {{
      const sku = skus[0];
      return `<div class="lc-action-card lc-action-pin">
        <div class="lc-action-top"><span>${{escapeHtml(actionType)}}</span><small>${{time}}s</small></div>
        ${{productLine(sku)}}
      </div>`;
    }}
    if (actionType === "SHOW_PROMO_CODE") {{
      return `<div class="lc-action-card lc-action-promo">
        <div class="lc-action-top"><span>${{escapeHtml(actionType)}}</span><small>${{time}}s</small></div>
        <div class="lc-promo-badge">${{escapeHtml(payload.promo_code || "PROMO")}}</div>
        <strong>${{escapeHtml(payload.promo_description || payload.title || "Promo code detected")}}</strong>
        <div class="lc-mini-product-row">${{skus.map(productVisual).join("")}}</div>
        <div class="lc-muted">${{skus.map(escapeHtml).join(" + ")}}</div>
      </div>`;
    }}
    if (actionType === "SHOW_BUNDLE_RECOMMENDATION") {{
      return `<div class="lc-action-card lc-action-bundle">
        <div class="lc-action-top"><span>${{escapeHtml(actionType)}}</span><small>${{time}}s</small></div>
        <strong>${{escapeHtml(payload.title || "Recommended bundle")}}</strong>
        <div class="lc-bundle-row">${{skus.map(productVisual).join('<div class="lc-bundle-plus">+</div>')}}</div>
        <div class="lc-muted">${{escapeHtml(payload.reason || skus.join(" + "))}}</div>
      </div>`;
    }}
    if (actionType === "START_FLASH_SALE_COUNTDOWN") {{
      return `<div class="lc-action-card lc-action-countdown">
        <div class="lc-action-top"><span>${{escapeHtml(actionType)}}</span><small>${{time}}s</small></div>
        <div class="lc-countdown-time">${{escapeHtml(payload.duration_minutes || 5)}} min</div>
        <strong>${{escapeHtml(payload.promo_code || "Live deal")}}</strong>
        <div class="lc-mini-product-row">${{skus.map(productVisual).join("")}}</div>
        <div class="lc-muted">Flash sale window started.</div>
      </div>`;
    }}
    return `<div class="lc-action-card">
      <div class="lc-action-top"><span>${{escapeHtml(actionType)}}</span><small>${{time}}s</small></div>
      <strong>${{escapeHtml(payload.title || actionType)}}</strong>
      <div class="lc-muted">${{skus.map(escapeHtml).join(" + ")}}</div>
    </div>`;
  }};
  const render = () => {{
    const t = audio ? (audio.currentTime || 0) : Number.POSITIVE_INFINITY;
    const current = captions.find(item => t >= item.start && t <= item.end) || captions.filter(item => item.end <= t).slice(-1)[0];
    captionEl.textContent = current ? current.text : "Waiting for caption...";
    const visible = actions.filter(item => item.timestamp <= t);
    actionsEl.innerHTML = (visible.length ? visible : actions.slice(0, 3)).slice(-6).map(renderAction).join("");
  }};
  if (audio) {{
    audio.addEventListener("timeupdate", render);
    audio.addEventListener("seeked", render);
    audio.addEventListener("play", render);
  }}
  render();
}})();
</script>
"""


def save_uploaded_audio(upload_widget: Any, output_dir: Path) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    value = upload_widget.value
    if not value:
        return None

    if isinstance(value, dict):
        name, payload = next(iter(value.items()))
        content = payload.get("content")
    else:
        payload = value[0]
        name = payload.get("name", "uploaded_audio")
        content = payload.get("content")

    if content is None:
        return None
    safe_name = "".join(ch for ch in str(name) if ch.isalnum() or ch in "._-") or "uploaded_audio"
    path = output_dir / safe_name
    path.write_bytes(bytes(content))
    return path


def save_uploaded_image(
    upload_widget: Any,
    output_dir: Path,
    sku: str,
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    value = upload_widget.value
    if not value:
        return None

    if isinstance(value, dict):
        name, payload = next(iter(value.items()))
        content = payload.get("content")
    else:
        payload = value[0]
        name = payload.get("name", "product_image")
        content = payload.get("content")

    if content is None:
        return None
    suffix = Path(str(name)).suffix or ".png"
    safe_sku = "".join(ch for ch in str(sku) if ch.isalnum() or ch in "._-") or "product"
    path = output_dir / f"{safe_sku}{suffix}"
    path.write_bytes(bytes(content))
    return path


def record_colab_mic_clip(output_dir: Path, max_seconds: float = 12.0) -> Path:
    try:
        from google.colab import output  # type: ignore
        from IPython.display import HTML, Javascript, display  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Recording from mic requires Google Colab browser APIs.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    display(
        HTML(
            """
            <div id="full-demo-mic-recorder" class="lc-mic-recorder">
              <div class="lc-mic-recorder-head">
                <div>
                  <div class="lc-panel-title">Record a voice sample</div>
                  <div class="lc-muted">Start recording, stop when done, then confirm to run ASR and commerce actions.</div>
                </div>
                <div class="lc-recording-time" data-role="time">0.00s</div>
              </div>
              <div class="lc-mic-meter"><div data-role="meter"></div></div>
              <div class="lc-mic-controls">
                <button data-role="start">Start recording</button>
                <button data-role="stop" disabled>Stop</button>
                <button data-role="confirm" disabled>Confirm</button>
                <button data-role="cancel">Cancel</button>
              </div>
              <audio data-role="preview" controls style="display:none;width:100%;margin-top:10px;"></audio>
              <div class="lc-muted" data-role="status">Waiting for microphone permission.</div>
            </div>
            """
        )
    )
    display(
        Javascript(
            """
            (() => {
              const roots = document.querySelectorAll('#full-demo-mic-recorder');
              const root = roots[roots.length - 1];
              const get = role => root.querySelector(`[data-role="${role}"]`);
              window.fullDemoMicRecorder = window.fullDemoMicRecorder || {};
              window.fullDemoMicRecorder.recordInteractive = async function(maxMilliseconds) {
                const maxMs = Math.max(500, Number(maxMilliseconds || 15000));
                return await new Promise((resolve, reject) => {
                  let stream = null;
                  let recorder = null;
                  let audioContext = null;
                  let analyser = null;
                  let timer = null;
                  let startedAt = null;
                  let blob = null;
                  let mimeType = 'audio/webm';
                  const chunks = [];
                  const setStatus = text => { get('status').textContent = text; };
                  const cleanupStream = () => {
                    if (timer) clearInterval(timer);
                    timer = null;
                    if (stream) stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    if (audioContext) audioContext.close();
                    audioContext = null;
                  };
                  const updateMeter = () => {
                    if (!analyser || !startedAt) return;
                    const waveform = new Uint8Array(analyser.fftSize);
                    analyser.getByteTimeDomainData(waveform);
                    let sumSquares = 0;
                    for (const value of waveform) {
                      const centered = (value - 128) / 128;
                      sumSquares += centered * centered;
                    }
                    const rms = Math.sqrt(sumSquares / waveform.length);
                    const elapsed = (performance.now() - startedAt) / 1000;
                    get('time').textContent = `${elapsed.toFixed(2)}s`;
                    get('meter').style.width = `${Math.max(2, Math.min(100, rms * 700))}%`;
                    if (elapsed * 1000 >= maxMs && recorder && recorder.state !== 'inactive') {
                      recorder.stop();
                    }
                  };
                  const stopRecorder = () => {
                    if (recorder && recorder.state !== 'inactive') recorder.stop();
                  };
                  get('start').onclick = async () => {
                    try {
                      stream = await navigator.mediaDevices.getUserMedia({audio: true});
                      mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                        ? 'audio/webm;codecs=opus'
                        : 'audio/webm';
                      audioContext = new (window.AudioContext || window.webkitAudioContext)();
                      const source = audioContext.createMediaStreamSource(stream);
                      analyser = audioContext.createAnalyser();
                      analyser.fftSize = 1024;
                      source.connect(analyser);
                      recorder = new MediaRecorder(stream, {mimeType});
                      recorder.ondataavailable = event => {
                        if (event.data && event.data.size > 0) chunks.push(event.data);
                      };
                      recorder.onerror = event => reject(event.error || event);
                      recorder.onstop = () => {
                        blob = new Blob(chunks, {type: mimeType});
                        cleanupStream();
                        const preview = get('preview');
                        preview.src = URL.createObjectURL(blob);
                        preview.style.display = 'block';
                        get('start').disabled = true;
                        get('stop').disabled = true;
                        get('confirm').disabled = false;
                        setStatus('Recording stopped. Preview it, then confirm or cancel.');
                      };
                      startedAt = performance.now();
                      recorder.start();
                      get('start').disabled = true;
                      get('stop').disabled = false;
                      get('confirm').disabled = true;
                      setStatus('Recording...');
                      timer = setInterval(updateMeter, 100);
                    } catch (error) {
                      cleanupStream();
                      reject(error);
                    }
                  };
                  get('stop').onclick = stopRecorder;
                  get('cancel').onclick = () => {
                    cleanupStream();
                    resolve({cancelled: true});
                  };
                  get('confirm').onclick = () => {
                    if (!blob) return;
                    const reader = new FileReader();
                    reader.onloadend = () => resolve({
                      dataUrl: reader.result,
                      mimeType,
                      size: blob.size,
                      durationSeconds: startedAt ? (performance.now() - startedAt) / 1000 : null
                    });
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                  };
                  setStatus('Click Start recording to begin.');
                });
              };
            })();
            """
        )
    )
    payload = output.eval_js(
        "window.fullDemoMicRecorder.recordInteractive("
        f"{max(500, int(float(max_seconds) * 1000))})"
    )
    if isinstance(payload, dict) and payload.get("cancelled"):
        raise RuntimeError("microphone recording cancelled")
    if not isinstance(payload, dict) or not payload.get("dataUrl"):
        raise RuntimeError("microphone recording did not return audio")
    _, encoded = str(payload["dataUrl"]).split(",", 1)
    path = output_dir / "recording_mode_mic.webm"
    path.write_bytes(base64.b64decode(encoded))
    return path


def _load_actions(path: Path) -> List[CommerceAction]:
    raw_actions = json.loads(path.read_text(encoding="utf-8"))
    return [
        CommerceAction(
            timestamp=float(raw["timestamp"]),
            action_type=str(raw["action_type"]),
            skus=[str(sku) for sku in raw.get("skus", [])],
            confidence=float(raw.get("confidence", 0.0)),
            evidence_text=str(raw.get("evidence_text", "")),
            display_payload=dict(raw.get("display_payload", {})),
        )
        for raw in raw_actions
    ]


def save_product_catalog_csv(items: Sequence[ProductCatalogItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sku",
            "product_name",
            "brand",
            "category",
            "price",
            "discount_price",
            "stock",
            "description",
            "tags",
            "compatible_with",
            "deeplink",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    **item.to_dict(),
                    "tags": ";".join(item.tags),
                    "compatible_with": ";".join(item.compatible_with),
                }
            )


def save_promotions_csv(promotions: Sequence[Promotion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "promo_code",
            "promo_description",
            "discount_type",
            "discount_value",
            "live_only",
            "eligible_categories",
            "eligible_tags",
            "eligible_skus",
            "required_min_stock",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for promotion in promotions:
            writer.writerow(
                {
                    **promotion.to_dict(),
                    "live_only": str(promotion.live_only).lower(),
                    "eligible_categories": ";".join(promotion.eligible_categories),
                    "eligible_tags": ";".join(promotion.eligible_tags),
                    "eligible_skus": ";".join(promotion.eligible_skus),
                }
            )


def default_asr_model(provider: str) -> str:
    return "turbo" if provider in {"openai_whisper", "typhoon_whisper"} else "turbo"


def _audio_data_uri(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _product_card_html(item: ProductCatalogItem, image_uri: Optional[str] = None) -> str:
    initials = _initials(item.product_name)
    tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in item.tags[:4])
    compatible = ", ".join(item.compatible_with) or "None"
    return f"""
<div class="lc-product-card">
  {_product_visual_html(item, image_uri)}
  <div>
    <strong>{html.escape(item.product_name)}</strong>
    <div class="lc-muted">{html.escape(item.brand)} &middot; {html.escape(item.category)}</div>
    <div class="lc-price">THB {item.discount_price} <s>{item.price}</s></div>
    <div class="lc-muted">Stock {item.stock} &middot; Compatible: {html.escape(compatible)}</div>
    <div class="lc-tags">{tags}</div>
  </div>
</div>
"""


def _promo_card_html(promotion: Promotion) -> str:
    return f"""
<div class="lc-promo-card">
  <div class="lc-promo-code">{html.escape(promotion.promo_code)}</div>
  <strong>{html.escape(promotion.promo_description)}</strong>
  <div class="lc-muted">{html.escape(promotion.discount_type)} &middot; {promotion.discount_value}</div>
  <div class="lc-muted">Categories: {html.escape(', '.join(promotion.eligible_categories) or 'Any')}</div>
  <div class="lc-muted">SKUs: {html.escape(', '.join(promotion.eligible_skus) or 'Any')}</div>
</div>
"""


def _initials(value: str) -> str:
    parts = [part for part in value.split() if part]
    if not parts:
        return "PR"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _safe_output_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part) or "source"


def _product_visual_html(
    item: ProductCatalogItem,
    image_uri: Optional[str] = None,
    class_name: str = "lc-product-thumb",
) -> str:
    if image_uri:
        return (
            f'<div class="{class_name} lc-product-photo">'
            f'<img src="{html.escape(image_uri)}" alt="{html.escape(item.product_name)}">'
            "</div>"
        )
    return (
        f'<div class="{class_name}">'
        f"{_product_icon_svg(item)}"
        f'<span>{html.escape(_initials(item.product_name))}</span>'
        "</div>"
    )


def _product_icon_svg(item: ProductCatalogItem) -> str:
    text = " ".join(
        [
            item.product_name,
            item.category,
            item.description,
            " ".join(item.tags),
        ]
    ).lower()
    if "earbud" in text or "bluetooth" in text or "music" in text:
        path = (
            '<path d="M8 12a4 4 0 0 1 8 0v4a2 2 0 0 1-2 2h-1v-6h3"/>'
            '<path d="M8 12v6H7a2 2 0 0 1-2-2v-4"/>'
            '<path d="M13 18v2"/>'
            '<path d="M7 18v2"/>'
        )
    elif "power" in text or "bank" in text or "charge" in text:
        path = (
            '<rect x="4" y="7" width="14" height="10" rx="2"/>'
            '<path d="M18 10h2v4h-2"/>'
            '<path d="M8 11h3l-2 4h3"/>'
        )
    elif "stand" in text:
        path = (
            '<rect x="8" y="4" width="8" height="12" rx="2"/>'
            '<path d="M9 20h6"/>'
            '<path d="M12 16v4"/>'
        )
    elif "drink" in text or "collagen" in text or "jelly" in text:
        path = (
            '<path d="M8 4h8l-1 16H9L8 4z"/>'
            '<path d="M9 8h6"/>'
            '<path d="M10 12h4"/>'
        )
    elif "sunscreen" in text or "spf" in text:
        path = (
            '<circle cx="12" cy="12" r="4"/>'
            '<path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.5 4.5l2.1 2.1M17.4 17.4l2.1 2.1M19.5 4.5l-2.1 2.1M6.6 17.4l-2.1 2.1"/>'
        )
    else:
        path = (
            '<path d="M9 3h6l1 4v13H8V7l1-4z"/>'
            '<path d="M9 7h6"/>'
            '<path d="M10 12h4"/>'
        )
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{path}</svg>'
    )


def _product_map_payload(
    catalog: Sequence[ProductCatalogItem],
    product_images: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    product_images = product_images or {}
    return {
        item.sku: {
            **item.to_dict(),
            "visual": _product_visual_html(
                item,
                product_images.get(item.sku),
                class_name="lc-action-product-thumb",
            ),
        }
        for item in catalog
    }


def _call_safely(callback: Any) -> None:
    try:
        callback()
    except Exception:
        pass


def _style_block() -> str:
    return """
<style>
.lc-header{display:flex;justify-content:space-between;align-items:flex-start;padding:22px 24px;background:#171717;color:#fff;border-radius:8px;margin-bottom:14px;border:1px solid #2f2f2f}
.lc-header h1{font-size:30px;line-height:1.1;margin:6px 0 0 0;letter-spacing:0;font-weight:800;color:#f9fafb}
.lc-eyebrow{font-size:12px;text-transform:uppercase;color:#fca5a5;font-weight:800}
.lc-status-pill{border:1px solid #525252;border-radius:999px;padding:7px 12px;font-size:12px;color:#e5e7eb;background:#262626}
.lc-panel-title{font-weight:800;font-size:17px;margin:0 0 8px 0;color:#111827}
.lc-subtitle{font-size:12px;font-weight:800;text-transform:uppercase;color:#6b7280;margin-top:6px}
.lc-copy{font-size:13px;color:#475467;line-height:1.45;margin-bottom:4px}
.lc-operator-panel{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:14px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-sample-card{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:10px;box-shadow:0 2px 8px rgba(16,24,40,.04)}
.lc-tool-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;box-shadow:0 2px 8px rgba(16,24,40,.04);gap:10px}
.lc-form-panel{background:#fff;border:1px solid #d0d5dd;border-radius:8px;padding:10px}
.lc-defaults,.lc-source-detail,.lc-selected-summary{border:1px solid #eaecf0;border-radius:8px;background:#f9fafb;color:#344054;font-size:12px;line-height:1.45;padding:10px}
.lc-selected-summary{background:#fff7ed;border-color:#fed7aa;color:#7c2d12}
.lc-sample-thumb{height:132px;border-radius:8px;background:linear-gradient(145deg,#171717,#7f1d1d 58%,#0f766e);position:relative;overflow:hidden;margin-bottom:8px;border:1px solid #e5e7eb}
.lc-sample-badge{position:absolute;top:10px;left:10px;width:34px;height:34px;border-radius:50%;background:#fff;color:#b42318;display:flex;align-items:center;justify-content:center;font-size:15px}
.lc-sample-host{position:absolute;left:50%;top:52%;transform:translate(-50%,-50%);width:82px;height:104px}
.lc-sample-head{width:38px;height:38px;border-radius:50%;background:#fff7ed;margin:0 auto 5px;border:3px solid rgba(255,255,255,.38)}
.lc-sample-body{width:72px;height:58px;border-radius:24px 24px 6px 6px;background:#ef4444;margin:0 auto}
.lc-sample-strip{position:absolute;left:10px;right:10px;bottom:10px;background:rgba(17,24,39,.84);color:#fff;border-radius:6px;padding:6px 8px;font-size:12px;font-weight:800;text-align:center}
.lc-tool-icon{width:42px;height:42px;border-radius:50%;background:#fff7ed;color:#b42318;border:1px solid #fed7aa;display:flex;align-items:center;justify-content:center;font-size:16px;flex:0 0 auto}
.lc-tool-title{color:#111827}
.lc-source-caption{font-size:12px;color:#667085;line-height:1.3;min-height:30px;text-align:center}
.lc-source-button{border-radius:8px!important;font-weight:800!important}
.lc-tool-button,.lc-upload-widget button{border-radius:8px!important;font-weight:800!important}
.lc-upload-panel{display:grid;grid-template-columns:42px minmax(0,1fr);gap:10px;align-items:center;border:1px dashed #d0d5dd;border-radius:8px;background:#f9fafb;color:#344054;padding:12px}
.lc-source-icon{width:42px;height:42px;border-radius:50%;background:#fff7ed;color:#b42318;display:flex;align-items:center;justify-content:center;font-weight:900}
.lc-mic-recorder{border:1px solid #d0d5dd;border-radius:8px;padding:14px;background:#fff;color:#111827;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-mic-recorder-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}
.lc-recording-time{border:1px solid #fed7aa;border-radius:999px;background:#fff7ed;color:#7c2d12;padding:6px 10px;font-weight:800}
.lc-mic-meter{height:10px;background:#f2f4f7;border-radius:999px;overflow:hidden;margin-bottom:12px}
.lc-mic-meter div{height:100%;width:0%;background:#b42318}
.lc-mic-controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.lc-mic-controls button{border:0;border-radius:8px;padding:9px 12px;font-weight:800;cursor:pointer}
.lc-mic-controls button[data-role=start],.lc-mic-controls button[data-role=confirm]{background:#b42318;color:#fff}
.lc-mic-controls button[data-role=stop],.lc-mic-controls button[data-role=cancel]{background:#f2f4f7;color:#344054}
.lc-mic-controls button:disabled{opacity:.5;cursor:not-allowed}
.lc-processing-panel{border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#111827;padding:18px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-processing-steps{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.lc-processing-steps span{border-radius:999px;background:#f2f4f7;color:#344054;padding:5px 9px;font-size:12px;font-weight:700}
.lc-status-card{border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#111827;padding:10px;display:flex;flex-direction:column;gap:3px}
.lc-status-card span{color:#667085;font-size:12px;line-height:1.35}
.lc-status-error{border-color:#fecaca;background:#fef2f2;color:#7f1d1d}
.lc-recording-empty{border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#111827;min-height:480px;display:flex;align-items:center;justify-content:center;gap:14px;padding:24px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-empty-mark{width:64px;height:64px;border-radius:8px;background:#fff7ed;color:#b42318;border:1px solid #fed7aa;display:flex;align-items:center;justify-content:center}
.lc-empty-mark svg{width:32px;height:32px}
.lc-viewer{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr);gap:14px;border:1px solid #d0d5dd;border-radius:8px;padding:14px;background:#fff;margin-bottom:12px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-video-art{height:390px;background:#171717;border-radius:8px;position:relative;overflow:hidden}
.lc-video-art:before{content:"";position:absolute;inset:0;background:linear-gradient(145deg,#171717 0%,#7f1d1d 54%,#0f766e 100%)}
.lc-video-art:after{content:"LIVE COMMERCE";position:absolute;right:18px;top:18px;color:rgba(255,255,255,.18);font-size:30px;font-weight:900;letter-spacing:0}
.lc-video-badge{position:absolute;top:14px;left:14px;background:#fff;color:#111827;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:800;box-shadow:0 8px 20px rgba(0,0,0,.18)}
.lc-host-frame{position:absolute;left:50%;top:54%;transform:translate(-50%,-50%);width:190px;height:240px}
.lc-host-head{width:82px;height:82px;border-radius:50%;background:#fff7ed;margin:0 auto 8px auto;border:5px solid rgba(255,255,255,.42)}
.lc-host-body{width:168px;height:142px;border-radius:48px 48px 10px 10px;background:#ef4444;margin:0 auto;box-shadow:0 18px 60px rgba(0,0,0,.28)}
.lc-video-caption{position:absolute;left:18px;right:18px;bottom:18px;background:rgba(17,24,39,.88);color:#fff;border-radius:8px;padding:13px 14px;font-size:16px;line-height:1.45;min-height:52px}
.lc-audio{width:100%;margin-top:10px}
.lc-no-audio{border:1px solid #d0d5dd;border-radius:8px;padding:10px;margin-top:10px;color:#667085;background:#f9fafb}
.lc-progress-bar{height:10px;border-radius:999px;overflow:hidden;background:#f2f4f7;margin-top:10px}
.lc-progress-bar div{height:100%;width:0%;background:#b42318}
.lc-inline-progress{width:100%;margin-top:0}
.lc-live-mic-controls{width:54px;height:42px}
.lc-live-toggle{width:54px;height:42px;border:0;border-radius:8px;background:#b42318;color:#fff;font-size:17px;font-weight:900;cursor:pointer;display:flex;align-items:center;justify-content:center}
.lc-live-toggle:disabled{opacity:.5;cursor:not-allowed}
.lc-processing-panel .lc-progress-bar div{width:42%;animation:lc-progress-sweep 1.15s ease-in-out infinite}
@keyframes lc-progress-sweep{0%{transform:translateX(-120%)}100%{transform:translateX(260%)}}
.lc-action-rail{min-height:390px;color:#111827}
.lc-stats{display:flex;gap:8px;margin-bottom:10px}
.lc-stats span{background:#f2f4f7;border-radius:999px;padding:4px 9px;font-size:12px;color:#344054}
.lc-live-actions{display:flex;flex-direction:column;gap:8px}
.lc-live-controls{display:flex;gap:8px;margin:10px 0}
.lc-live-controls button{border:0;border-radius:8px;padding:10px 13px;font-weight:800;cursor:pointer}
.lc-live-controls button[data-role=start]{background:#b42318;color:#fff}
.lc-live-controls button[data-role=stop]{background:#f2f4f7;color:#344054}
.lc-live-controls button:disabled{opacity:.5;cursor:not-allowed}
.lc-action-card,.lc-product-card,.lc-promo-card{border:1px solid #d0d5dd;border-radius:8px;padding:11px;background:#fff;color:#111827;box-shadow:0 2px 8px rgba(16,24,40,.04)}
.lc-action-card{display:flex;flex-direction:column;gap:8px}
.lc-action-top{display:flex;justify-content:space-between;gap:8px;align-items:center}
.lc-action-top span,.lc-action-type{font-size:11px;color:#b42318;font-weight:900;letter-spacing:.04em}
.lc-action-top small{font-size:11px;color:#667085}
.lc-action-pin{border-color:#bfdbfe;background:#eff6ff}
.lc-action-promo{border-color:#fed7aa;background:#fff7ed}
.lc-action-bundle{border-color:#bbf7d0;background:#f0fdf4}
.lc-action-countdown{border-color:#fecaca;background:#fef2f2}
.lc-action-product-row{display:grid;grid-template-columns:54px minmax(0,1fr);gap:10px;align-items:center}
.lc-action-product-thumb{width:52px;height:52px;border-radius:8px;background:#fee2e2;color:#991b1b;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;flex:0 0 auto}
.lc-action-product-thumb svg{width:27px;height:27px}
.lc-action-product-thumb span{position:absolute;right:5px;bottom:3px;font-size:10px;font-weight:900}
.lc-action-product-thumb img{width:100%;height:100%;object-fit:cover}
.lc-action-price{font-weight:900;color:#111827}.lc-action-price s{color:#667085;font-weight:400;margin-left:5px}
.lc-promo-badge{align-self:flex-start;border-radius:8px;background:#b42318;color:#fff;padding:7px 10px;font-size:18px;font-weight:900;letter-spacing:.04em}
.lc-bundle-row{display:flex;align-items:center;gap:8px}
.lc-mini-product-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.lc-mini-product-row .lc-action-product-thumb{width:36px;height:36px}
.lc-mini-product-row .lc-action-product-thumb svg{width:20px;height:20px}
.lc-mini-product-row .lc-action-product-thumb span{font-size:8px}
.lc-bundle-plus{font-weight:900;color:#15803d}
.lc-countdown-time{font-size:34px;line-height:1;font-weight:900;color:#b42318}
.lc-catalog-grid{display:grid;grid-template-columns:1.35fr .9fr;gap:14px}
.lc-catalog-shell{border:1px solid #374151;border-radius:8px;background:#111827;color:#f9fafb;padding:14px;box-shadow:0 10px 28px rgba(0,0,0,.22)}
.lc-catalog-shell .lc-panel-title{color:#f9fafb}
.lc-catalog-shell .lc-muted{color:#cbd5e1}
.lc-catalog-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}
.lc-catalog-heading span{color:#cbd5e1;font-size:12px;background:#1f2937;border-radius:999px;padding:5px 10px}
.lc-catalog-shell .lc-product-card,.lc-catalog-shell .lc-promo-card{background:#1f2937;border-color:#374151;color:#f9fafb;box-shadow:none}
.lc-catalog-shell .lc-product-card strong,.lc-catalog-shell .lc-promo-card strong,.lc-catalog-shell .lc-price{color:#f9fafb}
.lc-catalog-shell .lc-tags span{background:#374151;color:#e5e7eb}
.lc-card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.lc-product-card{display:grid;grid-template-columns:64px minmax(0,1fr);gap:12px}
.lc-product-card strong,.lc-promo-card strong{color:#111827}
.lc-product-thumb{width:62px;height:62px;border-radius:8px;background:#fee2e2;color:#991b1b;display:flex;align-items:center;justify-content:center;font-weight:900;position:relative;overflow:hidden}
.lc-product-thumb svg{width:32px;height:32px}
.lc-product-thumb span{position:absolute;right:6px;bottom:4px;font-size:11px}
.lc-product-thumb img{width:100%;height:100%;object-fit:cover}
.lc-price{font-weight:800;margin:5px 0;color:#111827}.lc-price s{color:#667085;font-weight:400;margin-left:6px}
.lc-muted{color:#667085;font-size:12px;line-height:1.35}
.lc-tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
.lc-tags span{background:#f2f4f7;border-radius:999px;padding:2px 7px;font-size:11px;color:#344054}
.lc-promo-code{font-weight:900;color:#b42318;font-size:15px;margin-bottom:5px}
.lc-form-title{font-weight:900;font-size:16px;color:#111827;margin:8px 0}
.lc-catalog-shell .lc-product-card strong,.lc-catalog-shell .lc-promo-card strong,.lc-catalog-shell .lc-price{color:#f9fafb}
@media (max-width: 960px){.lc-viewer,.lc-catalog-grid{grid-template-columns:1fr}.lc-header{flex-direction:column;gap:10px}}
</style>
"""
