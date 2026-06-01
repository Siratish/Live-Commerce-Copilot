from __future__ import annotations

from pathlib import Path
from typing import Sequence
import base64
import html
import json
import mimetypes
import uuid

from src.schemas import CommerceAction


def _audio_data_uri(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_action_timeline_html(
    actions: Sequence[CommerceAction],
    audio_path: Path | None = None,
    title: str = "Live Commerce Action Timeline",
) -> str:
    widget_id = f"action-widget-{uuid.uuid4().hex}"
    actions_json = json.dumps(
        [action.to_dict() for action in actions],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    audio_html = ""
    if audio_path and audio_path.exists():
        audio_html = f'<audio controls preload="metadata" src="{_audio_data_uri(audio_path)}"></audio>'

    return f"""
<div id="{widget_id}" class="action-widget">
  <style>
    #{widget_id} {{
      font-family: Arial, sans-serif;
      color: #111827;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 16px;
      max-width: 920px;
      background: #ffffff;
    }}
    #{widget_id} h3 {{ margin: 0 0 12px; font-size: 18px; }}
    #{widget_id} audio {{ width: 100%; margin-bottom: 12px; }}
    #{widget_id} .now {{
      padding: 12px 14px;
      border-radius: 6px;
      background: #111827;
      color: #ffffff;
      min-height: 48px;
      line-height: 1.45;
    }}
    #{widget_id} .cards {{ margin-top: 12px; display: grid; gap: 8px; }}
    #{widget_id} .card {{
      display: none;
      border: 1px solid #e5e7eb;
      border-left: 4px solid #d1d5db;
      border-radius: 6px;
      padding: 10px 12px;
      background: #f9fafb;
    }}
    #{widget_id} .card.visible {{ display: block; }}
    #{widget_id} .card.active {{ border-left-color: #e51b23; background: #fff5f5; }}
    #{widget_id} .meta {{ color: #6b7280; font-size: 12px; margin-bottom: 4px; }}
    #{widget_id} .type {{ font-weight: 700; }}
    #{widget_id} .payload {{ white-space: pre-wrap; font: 12px Consolas, monospace; color: #374151; }}
  </style>
  <h3>{html.escape(title)}</h3>
  {audio_html}
  <div class="now">Press play to reveal actions in stream time.</div>
  <div class="cards"></div>
</div>
<script>
(function() {{
  const root = document.getElementById("{widget_id}");
  const audio = root.querySelector("audio");
  const now = root.querySelector(".now");
  const cards = root.querySelector(".cards");
  const actions = {actions_json};

  function fmt(seconds) {{
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${{mins}}:${{String(secs).padStart(2, "0")}}`;
  }}

  actions.forEach((action, index) => {{
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.index = String(index);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${{fmt(action.timestamp)}} · ${{action.skus.join(" + ")}} · confidence ${{action.confidence.toFixed(2)}}`;
    const type = document.createElement("div");
    type.className = "type";
    type.textContent = action.action_type;
    const payload = document.createElement("div");
    payload.className = "payload";
    payload.textContent = JSON.stringify(action.display_payload, null, 2);
    card.appendChild(meta);
    card.appendChild(type);
    card.appendChild(payload);
    cards.appendChild(card);
  }});

  function renderAt(t) {{
    let active = null;
    cards.querySelectorAll(".card").forEach((card, index) => {{
      const visible = t >= actions[index].timestamp;
      const isActive = visible && (!actions[index + 1] || t < actions[index + 1].timestamp);
      card.classList.toggle("visible", visible);
      card.classList.toggle("active", isActive);
      if (isActive) active = actions[index];
    }});
    if (active) {{
      now.textContent = `${{fmt(active.timestamp)}} · ${{active.action_type}} · ${{active.skus.join(" + ")}}`;
    }} else {{
      now.textContent = "Listening for the first commerce action...";
    }}
  }}

  if (audio) {{
    audio.addEventListener("timeupdate", () => renderAt(audio.currentTime || 0));
    audio.addEventListener("seeked", () => renderAt(audio.currentTime || 0));
    audio.addEventListener("play", () => renderAt(audio.currentTime || 0));
    renderAt(0);
  }} else {{
    actions.forEach((_, index) => {{
      const card = cards.querySelector(`[data-index="${{index}}"]`);
      if (card) card.classList.add("visible");
    }});
    now.textContent = `${{actions.length}} commerce actions generated.`;
  }}
}})();
</script>
"""


def save_action_timeline_html(
    actions: Sequence[CommerceAction],
    path: Path,
    audio_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_action_timeline_html(actions, audio_path=audio_path),
        encoding="utf-8",
    )
