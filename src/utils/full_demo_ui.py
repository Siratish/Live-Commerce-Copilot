from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import csv
import html
import json
import mimetypes
import subprocess
import sys

from src.ai.captioning import (
    CaptionResult,
    CaptioningEngine,
    CaptioningSettings,
    caption_metrics,
    load_cached_transcript,
    save_caption_json,
    write_caption_outputs,
)
from src.ai.commerce_actions import generate_commerce_actions, save_commerce_actions
from src.ai.decision import DEFAULT_TYPHOON_MODEL_ID
from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import CommerceAction, ProductCatalogItem, Promotion
from src.utils.action_timeline import save_action_timeline_html
from src.utils.live_mic import LiveMicDemoConfig, run_colab_live_mic_demo
from src.utils.realtime_audio_file import (
    RealtimeAudioFileDemoConfig,
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
}


class FullDemoSession:
    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.output_dir = repo_root / "outputs" / "full_demo"
        self.upload_dir = self.output_dir / "uploads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = load_product_catalog(DEFAULT_CATALOG)
        self.promotions = load_promotions(DEFAULT_PROMOTIONS)
        self.catalog_path = self.output_dir / "session_product_catalog.csv"
        self.promotions_path = self.output_dir / "session_promotions.csv"
        self.save_catalog_files()

    def save_catalog_files(self) -> None:
        save_product_catalog_csv(self.catalog, self.catalog_path)
        save_promotions_csv(self.promotions, self.promotions_path)

    def add_product(self, item: ProductCatalogItem) -> None:
        self.catalog = [existing for existing in self.catalog if existing.sku != item.sku]
        self.catalog.append(item)
        self.save_catalog_files()

    def add_promotion(self, promotion: Promotion) -> None:
        self.promotions = [
            existing
            for existing in self.promotions
            if existing.promo_code != promotion.promo_code
        ]
        self.promotions.append(promotion)
        self.save_catalog_files()


def display_full_demo_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display a single-notebook UI for the full live-commerce copilot demo."""
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        print("ipywidgets and IPython are required for this notebook UI.")
        print("Run: pip install ipywidgets")
        return

    session = FullDemoSession(repo_root)
    state: Dict[str, Any] = {}

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

    mode = widgets.ToggleButtons(
        options=[
            ("Recording", "recording"),
            ("Live replay", "live_file"),
            ("Live mic", "live_mic"),
        ],
        value="recording",
        description="Mode",
        style={"description_width": "72px"},
    )
    source = widgets.Dropdown(
        options=[
            ("Audio 1 sample", "Audio 1 - Beauty"),
            ("Audio 2 sample", "Audio 2 - Tech"),
            ("Upload audio", "upload"),
            ("Microphone", "mic"),
        ],
        value="Audio 1 - Beauty",
        description="Audio",
        style={"description_width": "72px"},
    )
    upload = widgets.FileUpload(
        accept="audio/*",
        multiple=False,
        description="Upload",
        layout=widgets.Layout(width="160px"),
    )
    provider = widgets.Dropdown(
        options=[
            ("OpenAI Whisper", "openai_whisper"),
            ("Typhoon Whisper", "typhoon_whisper"),
        ],
        value="openai_whisper",
        description="ASR",
        style={"description_width": "72px"},
    )
    model = widgets.Text(
        value="base",
        description="Model",
        placeholder="base, large, turbo, large-v3",
        style={"description_width": "72px"},
        layout=widgets.Layout(width="260px"),
    )
    use_cached = widgets.Checkbox(
        value=True,
        description="Use cached transcript for Audio 1 when available",
        indent=False,
    )
    install_deps = widgets.Checkbox(
        value=False,
        description="Install optional ASR dependencies before run",
        indent=False,
    )
    dynamic_chunking = widgets.Checkbox(
        value=True,
        description="Pause-aware chunking",
        indent=False,
    )
    chunk_seconds = widgets.FloatSlider(
        value=8.0,
        min=2.0,
        max=20.0,
        step=1.0,
        description="Max chunk",
        readout_format=".0f",
        style={"description_width": "84px"},
    )
    pause_seconds = widgets.FloatSlider(
        value=0.7,
        min=0.2,
        max=2.0,
        step=0.1,
        description="Pause",
        readout_format=".1f",
        style={"description_width": "84px"},
    )
    mic_chunks = widgets.IntSlider(
        value=6,
        min=1,
        max=20,
        step=1,
        description="Mic chunks",
        style={"description_width": "84px"},
    )
    run_button = widgets.Button(
        description="Run Demo",
        button_style="danger",
        icon="play",
        layout=widgets.Layout(width="140px"),
    )
    refresh_button = widgets.Button(
        description="Refresh Catalog",
        icon="refresh",
        layout=widgets.Layout(width="150px"),
    )

    output = widgets.Output()
    viewer = widgets.Output()
    catalog_output = widgets.Output()
    form_output = widgets.Output()

    controls = widgets.VBox(
        [
            widgets.HTML('<div class="lc-panel-title">Run Controls</div>'),
            widgets.HBox([mode, source, upload]),
            widgets.HBox([provider, model, run_button]),
            widgets.HBox([chunk_seconds, pause_seconds, mic_chunks]),
            widgets.HBox([use_cached, dynamic_chunking]),
            widgets.HBox([install_deps, refresh_button]),
            widgets.HTML(
                '<div class="lc-note">Recording mode processes the full speech first. '
                "Live replay waits for playback to pass each chunk. Live mic records "
                "continuously into a queue while ASR/actions consume chunks.</div>"
            ),
        ],
        layout=widgets.Layout(width="100%"),
    )

    def current_audio_path() -> Optional[Path]:
        if source.value in SAMPLE_AUDIO:
            return SAMPLE_AUDIO[str(source.value)]
        if source.value == "upload":
            return save_uploaded_audio(upload, session.upload_dir)
        return None

    def refresh_catalog(_: Any = None) -> None:
        session.save_catalog_files()
        with catalog_output:
            catalog_output.clear_output(wait=True)
            display(HTML(build_catalog_scene_html(session.catalog, session.promotions)))

    def run_demo(_: Any) -> None:
        with output:
            output.clear_output(wait=True)
            print("Starting demo run...")
        with viewer:
            viewer.clear_output(wait=True)

        try:
            if install_deps.value:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "-r",
                        str(repo_root / "requirements-asr.txt"),
                    ]
                )

            if mode.value == "recording":
                audio_path = (
                    record_colab_mic_clip(session.upload_dir, max_seconds=float(chunk_seconds.value))
                    if source.value == "mic"
                    else current_audio_path()
                )
                if audio_path is None:
                    raise RuntimeError("Choose a sample, upload an audio file, or use microphone.")
                summary, captions, actions = run_recording_pipeline(
                    session=session,
                    audio_path=audio_path,
                    source_key=str(source.value),
                    asr_provider=str(provider.value),
                    asr_model=model.value.strip() or default_asr_model(str(provider.value)),
                    use_cached=bool(use_cached.value),
                    dynamic_chunking=bool(dynamic_chunking.value),
                    chunk_seconds=float(chunk_seconds.value),
                    pause_seconds=float(pause_seconds.value),
                )
                render_summary(summary, output)
                with viewer:
                    display(
                        HTML(
                            build_viewer_scene_html(
                                audio_path=audio_path,
                                captions=captions,
                                actions=actions,
                                mode_label="Recording mode",
                                title="Batch ASR + action preview",
                            )
                        )
                    )
                state["last_summary"] = summary

            elif mode.value == "live_file":
                audio_path = current_audio_path()
                if audio_path is None:
                    raise RuntimeError("Live replay requires sample audio or uploaded audio.")
                config = RealtimeAudioFileDemoConfig(
                    audio_path=audio_path,
                    chunk_seconds=float(chunk_seconds.value),
                    dynamic_chunking=bool(dynamic_chunking.value),
                    pause_seconds=float(pause_seconds.value),
                    language="th",
                    asr_provider=str(provider.value),
                    asr_model=model.value.strip() or default_asr_model(str(provider.value)),
                    install_asr_deps=False,
                    output_dir=session.output_dir / "live_replay",
                    catalog_path=session.catalog_path,
                    promotions_path=session.promotions_path,
                )
                with viewer:
                    summary = run_realtime_audio_file_demo(config)
                captions = load_cached_transcript(Path(summary["captions_json"]))
                actions = _load_actions(Path(summary["actions_json"]))
                render_summary(summary, output)
                with viewer:
                    display(
                        HTML(
                            build_viewer_scene_html(
                                audio_path=audio_path,
                                captions=captions,
                                actions=actions,
                                mode_label="Live replay mode",
                                title="Real-time gated file stream result",
                            )
                        )
                    )
                state["last_summary"] = summary

            else:
                config = LiveMicDemoConfig(
                    chunk_seconds=float(chunk_seconds.value),
                    pause_seconds=float(pause_seconds.value),
                    max_chunks=int(mic_chunks.value),
                    language="th",
                    asr_provider=str(provider.value),
                    asr_model=model.value.strip() or default_asr_model(str(provider.value)),
                    install_asr_deps=False,
                    show_debug_panel=True,
                    continuous_recording=True,
                    output_dir=session.output_dir / "live_mic",
                    catalog_path=session.catalog_path,
                    promotions_path=session.promotions_path,
                )
                with viewer:
                    summary = run_colab_live_mic_demo(config)
                captions = load_cached_transcript(Path(summary["captions_json"]))
                actions = _load_actions(Path(summary["actions_json"]))
                render_summary(summary, output)
                with viewer:
                    display(
                        HTML(
                            build_viewer_scene_html(
                                audio_path=None,
                                captions=captions,
                                actions=actions,
                                mode_label="Live mic mode",
                                title="Microphone live-stream result",
                            )
                        )
                    )
                state["last_summary"] = summary

        except Exception as exc:
            with output:
                print(f"Demo failed: {exc}")

    run_button.on_click(run_demo)
    refresh_button.on_click(refresh_catalog)

    form_ui = build_catalog_forms(session, refresh_catalog)
    app = widgets.VBox(
        [
            title,
            widgets.HBox(
                [
                    widgets.VBox([controls, output], layout=widgets.Layout(width="38%")),
                    widgets.VBox([viewer], layout=widgets.Layout(width="62%")),
                ],
                layout=widgets.Layout(align_items="stretch"),
            ),
            widgets.HTML('<div class="lc-section-title">Catalog & Promotion Scene</div>'),
            widgets.HBox(
                [
                    widgets.VBox([catalog_output], layout=widgets.Layout(width="58%")),
                    widgets.VBox([form_ui, form_output], layout=widgets.Layout(width="42%")),
                ],
                layout=widgets.Layout(align_items="flex-start"),
            ),
        ]
    )
    display(app)
    refresh_catalog()


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
) -> Tuple[Dict[str, Any], CaptionResult, List[CommerceAction]]:
    cached_path = SAMPLE_CACHED_TRANSCRIPTS.get(source_key)
    output_dir = session.output_dir / "recording"
    if use_cached and cached_path and cached_path.exists():
        captions = load_cached_transcript(cached_path)
        paths = write_caption_outputs(captions, output_dir)
        source_mode = "cached"
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
            asr_pause_seconds=pause_seconds,
            asr_max_new_tokens=256,
            allow_cached_fallback=False,
        )
        captions = CaptioningEngine(settings).transcribe()
        paths = write_caption_outputs(captions, output_dir)
        source_mode = asr_provider

    actions = generate_commerce_actions(captions, session.catalog, session.promotions)
    actions_path = output_dir / "commerce_actions.json"
    html_path = output_dir / "commerce_actions_timeline.html"
    save_commerce_actions(actions, actions_path)
    save_action_timeline_html(actions, html_path, audio_path=audio_path)
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
) -> str:
    product_cards = "\n".join(_product_card_html(item) for item in catalog)
    promo_cards = "\n".join(_promo_card_html(promotion) for promotion in promotions)
    return f"""
<div class="lc-catalog-grid">
  <div>
    <div class="lc-panel-title">Products ({len(catalog)})</div>
    <div class="lc-card-grid">{product_cards}</div>
  </div>
  <div>
    <div class="lc-panel-title">Promotions ({len(promotions)})</div>
    <div class="lc-card-grid">{promo_cards}</div>
  </div>
</div>
"""


def build_viewer_scene_html(
    audio_path: Optional[Path],
    captions: CaptionResult,
    actions: Sequence[CommerceAction],
    mode_label: str,
    title: str,
) -> str:
    segments = [segment.to_dict() for segment in captions.segments]
    action_payload = [action.to_dict() for action in actions]
    audio_html = ""
    if audio_path and audio_path.exists():
        audio_html = (
            f'<audio class="lc-audio" controls preload="metadata" '
            f'src="{_audio_data_uri(audio_path)}"></audio>'
        )
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
  const audio = root.querySelector("audio");
  const captionEl = root.querySelector('[data-role="caption"]');
  const actionsEl = root.querySelector('[data-role="actions"]');
  const renderAction = (action) => {{
    const title = action.display_payload?.title || action.action_type;
    const skus = (action.skus || []).join(" + ");
    return `<div class="lc-action-card"><div class="lc-action-type">${{action.action_type}}</div><strong>${{title}}</strong><div>${{skus}}</div><small>${{Number(action.timestamp || 0).toFixed(2)}}s</small></div>`;
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


def record_colab_mic_clip(output_dir: Path, max_seconds: float = 12.0) -> Path:
    try:
        from google.colab import output  # type: ignore
        from IPython.display import Javascript, display  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Recording from mic requires Google Colab browser APIs.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    display(
        Javascript(
            """
            window.recordFullDemoMicClip = async function(maxMilliseconds) {
              const stream = await navigator.mediaDevices.getUserMedia({audio: true});
              const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';
              return await new Promise((resolve, reject) => {
                const chunks = [];
                const recorder = new MediaRecorder(stream, {mimeType});
                recorder.ondataavailable = event => {
                  if (event.data && event.data.size > 0) chunks.push(event.data);
                };
                recorder.onerror = event => reject(event.error || event);
                recorder.onstop = () => {
                  stream.getTracks().forEach(track => track.stop());
                  const blob = new Blob(chunks, {type: mimeType});
                  const reader = new FileReader();
                  reader.onloadend = () => resolve({dataUrl: reader.result, size: blob.size});
                  reader.onerror = reject;
                  reader.readAsDataURL(blob);
                };
                recorder.start();
                setTimeout(() => recorder.stop(), Math.max(500, maxMilliseconds));
              });
            };
            """
        )
    )
    payload = output.eval_js(
        f"window.recordFullDemoMicClip({max(500, int(float(max_seconds) * 1000))})"
    )
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
    return "turbo" if provider == "typhoon_whisper" else "base"


def _audio_data_uri(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _product_card_html(item: ProductCatalogItem) -> str:
    initials = _initials(item.product_name)
    tags = " ".join(f"<span>{html.escape(tag)}</span>" for tag in item.tags[:4])
    compatible = ", ".join(item.compatible_with) or "None"
    return f"""
<div class="lc-product-card">
  <div class="lc-product-thumb">{html.escape(initials)}</div>
  <div>
    <strong>{html.escape(item.product_name)}</strong>
    <div class="lc-muted">{html.escape(item.brand)} · {html.escape(item.category)}</div>
    <div class="lc-price">THB {item.discount_price} <s>{item.price}</s></div>
    <div class="lc-muted">Stock {item.stock} · Compatible: {html.escape(compatible)}</div>
    <div class="lc-tags">{tags}</div>
  </div>
</div>
"""


def _promo_card_html(promotion: Promotion) -> str:
    return f"""
<div class="lc-promo-card">
  <div class="lc-promo-code">{html.escape(promotion.promo_code)}</div>
  <strong>{html.escape(promotion.promo_description)}</strong>
  <div class="lc-muted">{html.escape(promotion.discount_type)} · {promotion.discount_value}</div>
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


def _style_block() -> str:
    return """
<style>
.lc-header{display:flex;justify-content:space-between;align-items:flex-start;padding:16px 18px;background:#111827;color:#fff;border-radius:8px;margin-bottom:12px}
.lc-header h1{font-size:24px;line-height:1.15;margin:4px 0 0 0;letter-spacing:0}
.lc-eyebrow{font-size:12px;text-transform:uppercase;color:#fca5a5;font-weight:700}
.lc-status-pill{border:1px solid #4b5563;border-radius:999px;padding:6px 10px;font-size:12px;color:#e5e7eb}
.lc-panel-title{font-weight:700;font-size:15px;margin:0 0 8px 0;color:#111827}
.lc-section-title{font-weight:700;font-size:18px;margin:16px 0 10px 0;color:#111827}
.lc-note{font-size:12px;color:#667085;line-height:1.45;margin-top:6px}
.lc-viewer{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr);gap:12px;border:1px solid #d0d5dd;border-radius:8px;padding:12px;background:#fff;margin-bottom:12px}
.lc-video-art{height:360px;background:#111827;border-radius:8px;position:relative;overflow:hidden}
.lc-video-art:before{content:"";position:absolute;inset:0;background:linear-gradient(135deg,#1f2937,#991b1b)}
.lc-video-badge{position:absolute;top:12px;left:12px;background:#fff;color:#111827;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700}
.lc-host-frame{position:absolute;left:50%;top:54%;transform:translate(-50%,-50%);width:170px;height:220px}
.lc-host-head{width:76px;height:76px;border-radius:50%;background:#f3f4f6;margin:0 auto 8px auto}
.lc-host-body{width:150px;height:130px;border-radius:42px 42px 10px 10px;background:#ef4444;margin:0 auto}
.lc-video-caption{position:absolute;left:18px;right:18px;bottom:18px;background:rgba(17,24,39,.84);color:#fff;border-radius:8px;padding:12px;font-size:16px;line-height:1.4;min-height:48px}
.lc-audio{width:100%;margin-top:8px}
.lc-no-audio{border:1px solid #d0d5dd;border-radius:8px;padding:10px;margin-top:8px;color:#667085}
.lc-action-rail{min-height:360px}
.lc-stats{display:flex;gap:8px;margin-bottom:8px}
.lc-stats span{background:#f2f4f7;border-radius:999px;padding:4px 8px;font-size:12px}
.lc-live-actions{display:flex;flex-direction:column;gap:8px}
.lc-action-card,.lc-product-card,.lc-promo-card{border:1px solid #d0d5dd;border-radius:8px;padding:10px;background:#fff}
.lc-action-type{font-size:11px;color:#b42318;font-weight:700;margin-bottom:3px}
.lc-catalog-grid{display:grid;grid-template-columns:1.4fr .9fr;gap:12px}
.lc-card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}
.lc-product-card{display:grid;grid-template-columns:56px minmax(0,1fr);gap:10px}
.lc-product-thumb{width:54px;height:54px;border-radius:8px;background:#fee2e2;color:#991b1b;display:flex;align-items:center;justify-content:center;font-weight:800}
.lc-price{font-weight:700;margin:4px 0}.lc-price s{color:#667085;font-weight:400;margin-left:6px}
.lc-muted{color:#667085;font-size:12px;line-height:1.35}
.lc-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.lc-tags span{background:#f2f4f7;border-radius:999px;padding:2px 6px;font-size:11px;color:#344054}
.lc-promo-code{font-weight:800;color:#b42318;font-size:13px;margin-bottom:4px}
</style>
"""
