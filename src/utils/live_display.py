from __future__ import annotations

from html import escape
from typing import Any, Mapping, Optional, Sequence


_DISPLAY_HANDLES: dict[str, Any] = {}


def display_stream_state_panel(
    *,
    title: str,
    state_label: str,
    metrics: Mapping[str, Any],
    captions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    detail: Optional[str] = None,
    accent: str = "#a33b2f",
    display_key: Optional[str] = None,
) -> None:
    """Render a recording-style live preview panel in notebooks."""
    try:
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        _print_stream_state(title, state_label, metrics, captions, actions, detail)
        return

    metrics_html = "".join(
        f"<span>{escape(str(label))}: {escape(_format_value(value))}</span>"
        for label, value in metrics.items()
        if value is not None
    )
    latest_caption = (
        str(captions[-1].get("text") or "Waiting for captions...")
        if captions
        else "Waiting for captions..."
    )
    detail_html = (
        f'<div class="lc-stream-detail">{escape(detail)}</div>' if detail else ""
    )
    html = f"""
    <style>
      .lc-stream-panel {{
        font-family: Arial, Helvetica, sans-serif;
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(280px, .75fr);
        gap: 18px;
        color: #172033;
        background: #ffffff;
      }}
      .lc-stream-video {{
        min-height: 390px;
        border-radius: 12px;
        background: linear-gradient(145deg, #171717 0%, #7f1d1d 54%, #0f766e 100%);
        position: relative;
        overflow: hidden;
      }}
      .lc-stream-video:after {{
        content: "LIVE COMMERCE";
        position: absolute;
        right: 22px;
        top: 22px;
        color: rgba(255, 255, 255, .18);
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 0;
      }}
      .lc-stream-state {{
        position: absolute;
        top: 16px;
        left: 16px;
        border-radius: 999px;
        background: #ffffff;
        color: #172033;
        font-size: 12px;
        font-weight: 800;
        padding: 8px 12px;
        white-space: nowrap;
        box-shadow: 0 8px 20px rgba(0, 0, 0, .18);
      }}
      .lc-stream-host {{
        position: absolute;
        left: 50%;
        top: 54%;
        transform: translate(-50%, -50%);
        width: 190px;
        height: 240px;
      }}
      .lc-stream-head {{
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: #fff7ed;
        margin: 0 auto 8px auto;
        border: 5px solid rgba(255, 255, 255, .42);
      }}
      .lc-stream-body {{
        width: 168px;
        height: 142px;
        border-radius: 48px 48px 10px 10px;
        background: #ef4444;
        margin: 0 auto;
        box-shadow: 0 18px 60px rgba(0, 0, 0, .28);
      }}
      .lc-stream-caption {{
        position: absolute;
        left: 20px;
        right: 20px;
        bottom: 20px;
        background: rgba(17, 24, 39, .88);
        color: #ffffff;
        border-radius: 10px;
        padding: 16px 18px;
        font-size: 20px;
        line-height: 1.45;
        min-height: 62px;
      }}
      .lc-stream-title {{
        font-size: 26px;
        line-height: 1.18;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 12px;
      }}
      .lc-stream-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 14px;
      }}
      .lc-stream-metrics span {{
        border-radius: 999px;
        background: #f2f4f7;
        color: #344054;
        padding: 5px 10px;
        font-size: 13px;
      }}
      .lc-stream-detail {{
        border: 1px solid #eaecf0;
        background: #f9fafb;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
        color: #475467;
        font-size: 13px;
        line-height: 1.4;
      }}
      .lc-action-list {{
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      .lc-action-card {{
        border: 1px solid #d7dde8;
        border-radius: 10px;
        padding: 16px 18px;
        background: #ffffff;
        color: #172033;
        box-shadow: 0 2px 8px rgba(20, 31, 48, .04);
      }}
      .lc-action-type {{
        color: {accent};
        font-size: 13px;
        font-weight: 900;
        letter-spacing: .04em;
        margin-bottom: 8px;
      }}
      .lc-action-title {{
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 4px;
      }}
      .lc-action-meta {{
        color: #172033;
        font-size: 15px;
        line-height: 1.35;
      }}
      .lc-action-time {{
        color: #475467;
        font-size: 13px;
        margin-top: 8px;
      }}
      .lc-empty {{
        border: 1px solid #d7dde8;
        border-radius: 10px;
        padding: 16px;
        color: #667085;
        font-size: 13px;
      }}
      @media (max-width: 760px) {{
        .lc-stream-panel {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <div class="lc-stream-panel">
      <div class="lc-stream-video">
        <div class="lc-stream-state">{escape(state_label)}</div>
        <div class="lc-stream-host">
          <div class="lc-stream-head"></div>
          <div class="lc-stream-body"></div>
        </div>
        <div class="lc-stream-caption">{escape(latest_caption)}</div>
      </div>
      <div class="lc-stream-rail">
        <div class="lc-stream-title">{escape(title)}</div>
        <div class="lc-stream-metrics">{metrics_html}</div>
        {detail_html}
        <div class="lc-action-list">{_action_cards_html(actions)}</div>
      </div>
    </div>
    """
    key = display_key or title
    handle = _DISPLAY_HANDLES.get(key)
    if handle is not None:
        try:
            handle.update(HTML(html))
            return
        except Exception:
            _DISPLAY_HANDLES.pop(key, None)
    _DISPLAY_HANDLES[key] = display(HTML(html), display_id=key)


def _action_cards_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return '<div class="lc-empty">No actions emitted yet.</div>'
    return "".join(
        f"""
        <div class="lc-action-card">
          <div class="lc-action-type">{escape(str(row.get("action") or row.get("action_type") or "ACTION"))}</div>
          <div class="lc-action-title">{escape(str(row.get("title") or _human_action_title(row.get("action") or row.get("action_type"))))}</div>
          <div class="lc-action-meta">{escape(str(row.get("skus") or "-"))}</div>
          <div class="lc-action-time">{escape(_format_value(row.get("time")))}s</div>
        </div>
        """
        for row in rows[-6:]
    )


def _human_action_title(action_type: Any) -> str:
    value = str(action_type or "Action").lower().replace("_", " ")
    return " ".join(word.upper() if word in {"asr", "sku"} else word.capitalize() for word in value.split())


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _print_stream_state(
    title: str,
    state_label: str,
    metrics: Mapping[str, Any],
    captions: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    detail: Optional[str],
) -> None:
    print(f"{title} | {state_label}")
    if detail:
        print(detail)
    print(" | ".join(f"{label}: {_format_value(value)}" for label, value in metrics.items()))
    print({"captions": list(captions), "actions": list(actions)})
