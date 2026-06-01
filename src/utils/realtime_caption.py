from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import base64
import html
import json
import mimetypes
import uuid

from src.schemas import CaptionResult, CaptionSegment


def _segments_from_any(captions: CaptionResult | Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(captions, CaptionResult):
        return [segment.to_dict() for segment in captions.segments]
    return list(captions.get("segments", []))


def _duration_from_any(
    captions: CaptionResult | Dict[str, Any],
    segments: Iterable[Dict[str, Any]],
) -> float:
    if isinstance(captions, CaptionResult):
        if captions.duration_seconds is not None:
            return captions.duration_seconds
        return captions.segments[-1].end if captions.segments else 0.0

    duration = captions.get("duration_seconds")
    if duration is not None:
        return float(duration)
    segment_list = list(segments)
    return float(segment_list[-1]["end"]) if segment_list else 0.0


def _audio_data_uri(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    mime_type = mime_type or "audio/mpeg"
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_realtime_caption_html(
    audio_path: Path,
    captions: CaptionResult | Dict[str, Any],
    title: str = "Real-Time Caption Preview",
) -> str:
    """Return a self-contained audio player with synchronized live captions."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    segments = _segments_from_any(captions)
    duration = _duration_from_any(captions, segments)
    widget_id = f"caption-widget-{uuid.uuid4().hex}"
    data_uri = _audio_data_uri(audio_path)
    segments_json = json.dumps(segments, ensure_ascii=False).replace("</", "<\\/")
    escaped_title = html.escape(title)

    return f"""
<div id="{widget_id}" class="caption-widget">
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
    #{widget_id} h3 {{
      margin: 0 0 12px;
      font-size: 18px;
      font-weight: 700;
    }}
    #{widget_id} audio {{
      width: 100%;
      margin: 6px 0 14px;
    }}
    #{widget_id} .caption-now {{
      min-height: 58px;
      padding: 14px 16px;
      border-radius: 6px;
      background: #111827;
      color: #ffffff;
      font-size: 21px;
      line-height: 1.45;
    }}
    #{widget_id} .caption-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 10px 0;
      font-size: 13px;
      color: #4b5563;
    }}
    #{widget_id} .meter {{
      height: 8px;
      background: #eef2f7;
      border-radius: 999px;
      overflow: hidden;
      margin: 10px 0 14px;
    }}
    #{widget_id} .meter-fill {{
      width: 0%;
      height: 100%;
      background: #e51b23;
      transition: width 120ms linear;
    }}
    #{widget_id} .generated {{
      max-height: 230px;
      overflow: auto;
      border-top: 1px solid #e5e7eb;
      padding-top: 10px;
    }}
    #{widget_id} .line {{
      display: none;
      padding: 8px 10px;
      margin: 4px 0;
      border-left: 3px solid #d1d5db;
      background: #f9fafb;
      border-radius: 4px;
      font-size: 14px;
      line-height: 1.45;
    }}
    #{widget_id} .line.visible {{
      display: block;
    }}
    #{widget_id} .line.active {{
      border-left-color: #e51b23;
      background: #fff5f5;
    }}
    #{widget_id} .time {{
      color: #6b7280;
      font-size: 12px;
      margin-right: 8px;
    }}
  </style>
  <h3>{escaped_title}</h3>
  <audio controls preload="metadata" src="{data_uri}"></audio>
  <div class="caption-now">Press play to start live captions.</div>
  <div class="caption-meta">
    <span class="status">Waiting for audio playback</span>
    <span class="clock">0.0s / {duration:.1f}s</span>
  </div>
  <div class="meter"><div class="meter-fill"></div></div>
  <div class="generated"></div>
</div>
<script>
(function() {{
  const root = document.getElementById("{widget_id}");
  const audio = root.querySelector("audio");
  const now = root.querySelector(".caption-now");
  const status = root.querySelector(".status");
  const clock = root.querySelector(".clock");
  const meter = root.querySelector(".meter-fill");
  const generated = root.querySelector(".generated");
  const segments = {segments_json};
  const duration = {duration:.6f};

  function fmt(seconds) {{
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${{mins}}:${{String(secs).padStart(2, "0")}}`;
  }}

  segments.forEach((segment, index) => {{
    const line = document.createElement("div");
    line.className = "line";
    line.dataset.index = String(index);
    const time = document.createElement("span");
    time.className = "time";
    time.textContent = `${{fmt(segment.start)}}-${{fmt(segment.end)}}`;
    const text = document.createElement("span");
    text.textContent = segment.text;
    line.appendChild(time);
    line.appendChild(text);
    generated.appendChild(line);
  }});

  function render() {{
    const t = audio.currentTime || 0;
    const activeIndex = segments.findIndex((segment) => t >= segment.start && t < segment.end);
    const lines = generated.querySelectorAll(".line");
    let revealed = 0;

    lines.forEach((line, index) => {{
      const segment = segments[index];
      const visible = t >= segment.start;
      line.classList.toggle("visible", visible);
      line.classList.toggle("active", index === activeIndex);
      if (visible) revealed += 1;
    }});

    if (activeIndex >= 0) {{
      now.textContent = segments[activeIndex].text;
      status.textContent = `Generating caption ${{activeIndex + 1}} of ${{segments.length}}`;
    }} else if (t >= duration && segments.length > 0) {{
      now.textContent = "Caption playback complete.";
      status.textContent = `Generated ${{segments.length}} caption segments`;
    }} else {{
      now.textContent = "Listening for the next caption...";
      status.textContent = `Generated ${{revealed}} of ${{segments.length}} caption segments`;
    }}

    const total = audio.duration && Number.isFinite(audio.duration) ? audio.duration : duration;
    clock.textContent = `${{t.toFixed(1)}}s / ${{total.toFixed(1)}}s`;
    meter.style.width = `${{Math.min(100, (t / Math.max(total, 0.01)) * 100)}}%`;
  }}

  audio.addEventListener("timeupdate", render);
  audio.addEventListener("seeked", render);
  audio.addEventListener("play", render);
  audio.addEventListener("loadedmetadata", render);
  render();
}})();
</script>
"""
