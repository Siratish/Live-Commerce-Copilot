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
    """Render a compact live status panel in notebooks, with a text fallback."""
    try:
        from IPython.display import HTML, display  # type: ignore
    except ImportError:
        _print_stream_state(title, state_label, metrics, captions, actions, detail)
        return

    metric_html = "".join(
        f"""
        <div class="lc-metric">
          <div class="lc-metric-label">{escape(str(label))}</div>
          <div class="lc-metric-value">{escape(_format_value(value))}</div>
        </div>
        """
        for label, value in metrics.items()
    )
    detail_html = (
        f'<div class="lc-stream-detail">{escape(detail)}</div>' if detail else ""
    )
    html = f"""
    <style>
      .lc-stream-panel {{
        font-family: Arial, Helvetica, sans-serif;
        border: 1px solid #d7dde8;
        border-radius: 12px;
        background: #ffffff;
        color: #172033;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(20, 31, 48, 0.08);
      }}
      .lc-stream-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
      }}
      .lc-stream-title {{
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0;
      }}
      .lc-stream-state {{
        border-radius: 999px;
        background: {accent};
        color: #ffffff;
        font-size: 12px;
        font-weight: 800;
        padding: 7px 10px;
        white-space: nowrap;
      }}
      .lc-stream-metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 8px;
        margin-bottom: 12px;
      }}
      .lc-metric {{
        border: 1px solid #e5e9f1;
        border-radius: 8px;
        background: #f8fafc;
        padding: 9px 10px;
        min-height: 54px;
      }}
      .lc-metric-label {{
        color: #667085;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
      }}
      .lc-metric-value {{
        color: #172033;
        font-size: 16px;
        font-weight: 800;
        margin-top: 4px;
      }}
      .lc-stream-detail {{
        border-left: 4px solid {accent};
        background: #fff7ed;
        border-radius: 8px;
        padding: 9px 10px;
        margin-bottom: 12px;
        color: #7c2d12;
        font-weight: 700;
      }}
      .lc-stream-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr);
        gap: 12px;
      }}
      .lc-stream-section {{
        border: 1px solid #e5e9f1;
        border-radius: 10px;
        overflow: hidden;
      }}
      .lc-section-title {{
        background: #f1f5f9;
        color: #334155;
        font-size: 12px;
        font-weight: 800;
        padding: 9px 10px;
        text-transform: uppercase;
      }}
      .lc-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
      }}
      .lc-table th,
      .lc-table td {{
        border-top: 1px solid #edf1f7;
        padding: 8px 10px;
        font-size: 13px;
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
      }}
      .lc-table th {{
        color: #667085;
        font-size: 11px;
        text-transform: uppercase;
        background: #fbfdff;
      }}
      .lc-empty {{
        padding: 16px 10px;
        color: #667085;
        font-size: 13px;
      }}
      @media (max-width: 760px) {{
        .lc-stream-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <div class="lc-stream-panel">
      <div class="lc-stream-head">
        <div class="lc-stream-title">{escape(title)}</div>
        <div class="lc-stream-state">{escape(state_label)}</div>
      </div>
      <div class="lc-stream-metrics">{metric_html}</div>
      {detail_html}
      <div class="lc-stream-grid">
        <div class="lc-stream-section">
          <div class="lc-section-title">Recent Captions</div>
          {_table_html(captions, "No captions emitted yet.")}
        </div>
        <div class="lc-stream-section">
          <div class="lc-section-title">Commerce Actions</div>
          {_table_html(actions, "No actions emitted yet.")}
        </div>
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


def _table_html(rows: Sequence[Mapping[str, Any]], empty_label: str) -> str:
    if not rows:
        return f'<div class="lc-empty">{escape(empty_label)}</div>'
    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(_format_value(row.get(column, '')))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="lc-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'


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
