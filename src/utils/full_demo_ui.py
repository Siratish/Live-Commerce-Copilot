from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import base64
import csv
import html
import json
import mimetypes

from src.ai.captioning import (
    CaptionResult,
    CaptioningEngine,
    CaptioningSettings,
    caption_metrics,
    load_cached_transcript,
    write_caption_outputs,
)
from src.ai.commerce_actions import generate_commerce_actions, save_commerce_actions
from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import CommerceAction, ProductCatalogItem, Promotion
from src.utils.action_timeline import save_action_timeline_html
from src.utils.live_mic import (
    LiveMicDemoConfig,
    run_colab_live_mic_demo,
)
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


def display_full_demo_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display the simplified full notebook demo UI."""
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
            widgets.HTML('<div class="lc-panel-title">Full demo</div>'),
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
            display(HTML(_full_demo_empty_state_html()))

    def process_audio(audio_path: Path, source_key: str, source_label: str) -> None:
        if state.get("running"):
            return
        set_busy(True)
        clear_status()
        with viewer:
            viewer.clear_output(wait=True)
            display(HTML(build_processing_scene_html("Processing full demo", source_label)))
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
                            mode_label="Full demo",
                            title="Action history",
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
            process_audio(audio_path, "mic", "Browser microphone input")
        except Exception as exc:
            set_status("error", "Mic input failed", str(exc))
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
        description="+",
        icon="plus",
        button_style="success",
        tooltip="Add item",
        layout=widgets.Layout(width="54px", height="40px"),
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


def display_live_audio_file_demo_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display the live-mode audio-file stream demo UI."""
    try:
        import ipywidgets as widgets  # type: ignore
        from IPython.display import HTML, clear_output, display  # type: ignore
    except ImportError:
        return

    repo_root = Path(repo_root)
    upload_dir = repo_root / "outputs" / "live_mode" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"running": False}

    display(HTML(_style_block()))
    status = widgets.HTML()
    runner = widgets.Output()
    audio1_button = widgets.Button(
        description="Audio 1",
        icon="play",
        button_style="danger",
        layout=widgets.Layout(width="132px", height="44px"),
    )
    audio2_button = widgets.Button(
        description="Audio 2",
        icon="play",
        button_style="danger",
        layout=widgets.Layout(width="132px", height="44px"),
    )
    upload = widgets.FileUpload(
        accept="audio/*",
        multiple=False,
        description="Upload audio",
        layout=widgets.Layout(width="190px", height="44px"),
    )
    source_picker = widgets.HBox(
        [audio1_button, audio2_button, upload],
        layout=widgets.Layout(gap="10px", flex_flow="row wrap"),
    )

    def set_running(is_running: bool) -> None:
        state["running"] = is_running
        source_picker.layout.display = "none" if is_running else ""
        audio1_button.disabled = is_running
        audio2_button.disabled = is_running
        upload.disabled = is_running

    def run_audio(audio_path: Path, label: str) -> None:
        if state.get("running"):
            return
        if not audio_path.exists():
            status.value = _status_card_html("error", "Audio file not found", str(audio_path))
            return

        safe_label = _safe_output_name(label)
        output_dir = repo_root / "outputs" / "live_mode" / "realtime_audio_file" / safe_label
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
            output_dir=output_dir,
            catalog_path=repo_root / "data" / "demo" / "product_catalog.csv",
            promotions_path=repo_root / "data" / "demo" / "promotions.csv",
        )
        set_running(True)
        status.value = _status_card_html(
            "info",
            "Realtime stream ready",
            "Use the play/pause button inside the generated stream panel. Source choices return after the run finishes.",
        )
        with runner:
            clear_output(wait=True)
            try:
                summary = run_realtime_audio_file_demo(config)
                status.value = _status_card_html(
                    "done",
                    "Realtime audio-file demo finished",
                    (
                        f"{summary.get('caption_count', 0)} captions / "
                        f"{summary.get('action_count', 0)} actions<br>"
                        f"Captions: <code>{html.escape(str(summary.get('captions_json', '')))}</code><br>"
                        f"Actions: <code>{html.escape(str(summary.get('actions_json', '')))}</code>"
                    ),
                )
            except Exception as exc:
                status.value = _status_card_html(
                    "error",
                    "Realtime audio-file demo failed",
                    str(exc),
                )
            finally:
                set_running(False)

    def on_upload_change(change: Dict[str, Any]) -> None:
        if not change.get("new") or state.get("running"):
            return
        audio_path = save_uploaded_audio(upload, upload_dir)
        if audio_path is None:
            status.value = _status_card_html(
                "error",
                "Upload failed",
                "No audio file was available.",
            )
            return
        run_audio(audio_path, f"Uploaded - {audio_path.name}")

    audio1_button.on_click(
        lambda _button: run_audio(SAMPLE_AUDIO["Audio 1 - Beauty"], "Audio 1 - Beauty")
    )
    audio2_button.on_click(
        lambda _button: run_audio(SAMPLE_AUDIO["Audio 2 - Tech"], "Audio 2 - Tech")
    )
    upload.observe(on_upload_change, names="value")
    display(
        widgets.VBox(
            [
                widgets.HTML('<div class="lc-panel-title">Live mode: audio-file stream</div>'),
                source_picker,
                status,
                runner,
            ],
            layout=widgets.Layout(gap="10px"),
        )
    )


def display_live_mic_demo_ui(repo_root: Path = REPO_ROOT) -> None:
    """Display the live-mode browser microphone demo UI."""
    try:
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        return

    repo_root = Path(repo_root)
    display(HTML(_style_block()))
    display(
        HTML(
            _status_card_html(
                "info",
                "Live microphone demo",
                "Use the mic button in the panel below to start/pause/resume microphone input. Interrupt this cell when you are done.",
            )
        )
    )
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
        show_debug_panel=True,
        continuous_recording=True,
        max_queue_chunks=16,
        output_dir=repo_root / "outputs" / "live_mode" / "live_mic",
        catalog_path=repo_root / "data" / "demo" / "product_catalog.csv",
        promotions_path=repo_root / "data" / "demo" / "promotions.csv",
    )
    summary = run_colab_live_mic_demo(config)
    display(
        HTML(
            _status_card_html(
                "done",
                "Live mic demo finished",
                (
                    f"{summary.get('caption_count', 0)} captions / "
                    f"{summary.get('action_count', 0)} actions<br>"
                    f"Captions: <code>{html.escape(str(summary.get('captions_json', '')))}</code><br>"
                    f"Actions: <code>{html.escape(str(summary.get('actions_json', '')))}</code>"
                ),
            )
        )
    )


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


def _full_demo_empty_state_html() -> str:
    return """
<div class="lc-full-demo-empty">
  <div class="lc-empty-mark">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 7h16v10H4z"/>
      <path d="M8 11h8"/>
      <path d="M10 15h4"/>
    </svg>
  </div>
  <div>
    <div class="lc-panel-title">Choose an audio source</div>
    <div class="lc-muted">Captions and commerce actions appear here after processing completes.</div>
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
    <div class="lc-current-action" data-role="current-action">
      <div class="lc-muted">Current action will appear here as playback reaches it.</div>
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
  const currentActionEl = root.querySelector('[data-role="current-action"]');
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
  const formatRemaining = (action, now) => {{
    const payload = action.display_payload || {{}};
    const total = Number(payload.duration_seconds || (Number(payload.duration_minutes || 5) * 60));
    const elapsed = Math.max(0, Number(now || 0) - Number(action.timestamp || 0));
    const remaining = Math.max(0, Math.ceil(total - elapsed));
    const minutes = Math.floor(remaining / 60);
    const seconds = String(remaining % 60).padStart(2, "0");
    return `${{minutes}}:${{seconds}}`;
  }};
  const renderAction = (action, now = 0, isCurrent = false) => {{
    const payload = action.display_payload || {{}};
    const actionType = action.action_type || "ACTION";
    const skus = action.skus || [];
    const time = Number(action.timestamp || 0).toFixed(2);
    if (actionType === "PIN_PRODUCT_CARD") {{
      const sku = skus[0];
      return `<div class="lc-action-card lc-action-pin">
        <div class="lc-action-top"><strong>${{escapeHtml(payload.title || "Pinned product")}}</strong><small>${{time}}s</small></div>
        ${{productLine(sku)}}
      </div>`;
    }}
    if (actionType === "SHOW_PROMO_CODE") {{
      return `<div class="lc-action-card lc-action-promo">
        <div class="lc-action-top"><strong>${{escapeHtml(payload.title || "Promo code")}}</strong><small>${{time}}s</small></div>
        <div class="lc-promo-badge">${{escapeHtml(payload.promo_code || "PROMO")}}</div>
        <strong>${{escapeHtml(payload.promo_description || payload.title || "Promo code detected")}}</strong>
        <div class="lc-mini-product-row">${{skus.map(productVisual).join("")}}</div>
      </div>`;
    }}
    if (actionType === "SHOW_BUNDLE_RECOMMENDATION") {{
      return `<div class="lc-action-card lc-action-bundle">
        <div class="lc-action-top"><strong>${{escapeHtml(payload.title || "Recommended bundle")}}</strong><small>${{time}}s</small></div>
        <div class="lc-bundle-row">${{skus.map(productVisual).join('<div class="lc-bundle-plus">+</div>')}}</div>
        <div class="lc-muted">${{escapeHtml(payload.reason || skus.join(" + "))}}</div>
      </div>`;
    }}
    if (actionType === "START_FLASH_SALE_COUNTDOWN") {{
      const countdown = formatRemaining(action, now);
      const attachedProduct = skus.length ? productLine(skus[0]) : "";
      const promo = payload.promo_code ? `<div class="lc-promo-badge">${{escapeHtml(payload.promo_code)}}</div>` : "";
      return `<div class="lc-action-card lc-action-countdown">
        <div class="lc-action-top"><strong>Flash sale countdown</strong><small>${{time}}s</small></div>
        ${{promo || attachedProduct}}
        <div class="lc-countdown-time" data-countdown-at="${{escapeHtml(time)}}">${{countdown}}</div>
        <div class="lc-mini-product-row">${{skus.map(productVisual).join("")}}</div>
      </div>`;
    }}
    return `<div class="lc-action-card">
      <div class="lc-action-top"><strong>${{escapeHtml(payload.title || "Action")}}</strong><small>${{time}}s</small></div>
      <strong>${{escapeHtml(payload.title || actionType)}}</strong>
      <div class="lc-muted">${{skus.map(escapeHtml).join(" + ")}}</div>
    </div>`;
  }};
  const render = () => {{
    const t = audio ? (audio.currentTime || 0) : Number.POSITIVE_INFINITY;
    const current = captions.find(item => t >= item.start && t <= item.end) || captions.filter(item => item.end <= t).slice(-1)[0];
    captionEl.textContent = current ? current.text : "Waiting for caption...";
    const visible = actions.filter(item => item.timestamp <= t);
    const currentAction = visible.slice(-1)[0];
    if (currentActionEl) {{
      currentActionEl.innerHTML = currentAction
        ? renderAction(currentAction, t, true)
        : '<div class="lc-muted">Current action will appear here as playback reaches it.</div>';
    }}
    actionsEl.innerHTML = visible.length
      ? visible.slice(-6).reverse().map(item => renderAction(item, t)).join("")
      : '<div class="lc-action-card"><div class="lc-muted">No actions emitted yet.</div></div>';
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
.lc-full-demo-empty{border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#111827;min-height:480px;display:flex;align-items:center;justify-content:center;gap:14px;padding:24px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
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
.lc-catalog-shell{border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#111827;padding:14px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.lc-catalog-shell .lc-panel-title{color:#111827}
.lc-catalog-shell .lc-muted{color:#667085}
.lc-catalog-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}
.lc-catalog-heading span{color:#344054;font-size:12px;background:#f2f4f7;border-radius:999px;padding:5px 10px}
.lc-catalog-shell .lc-product-card,.lc-catalog-shell .lc-promo-card{background:#fff;border-color:#d0d5dd;color:#111827;box-shadow:0 2px 8px rgba(16,24,40,.04)}
.lc-catalog-shell .lc-product-card strong,.lc-catalog-shell .lc-promo-card strong,.lc-catalog-shell .lc-price{color:#111827}
.lc-catalog-shell .lc-tags span{background:#f2f4f7;color:#344054}
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
.lc-catalog-shell .lc-product-card strong,.lc-catalog-shell .lc-promo-card strong,.lc-catalog-shell .lc-price{color:#111827}
@media (max-width: 960px){.lc-viewer,.lc-catalog-grid{grid-template-columns:1fr}.lc-header{flex-direction:column;gap:10px}}
</style>
"""
