from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional


@dataclass(frozen=True)
class AudioSourceStatus:
    path: Optional[Path]
    exists: bool
    ffmpeg_available: bool
    ffprobe_available: bool
    duration_seconds: Optional[float]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe_audio_duration(audio_path: Path) -> Optional[float]:
    if not ffprobe_available() or not audio_path.exists():
        return None

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def inspect_audio_source(audio_path: Optional[Path]) -> AudioSourceStatus:
    exists = bool(audio_path and audio_path.exists())
    return AudioSourceStatus(
        path=audio_path,
        exists=exists,
        ffmpeg_available=ffmpeg_available(),
        ffprobe_available=ffprobe_available(),
        duration_seconds=probe_audio_duration(audio_path) if audio_path else None,
    )
