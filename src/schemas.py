from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import re


_SPACE_RE = re.compile(r"\s+")


def normalize_caption_text(text: str) -> str:
    """Collapse repeated whitespace while preserving the spoken content."""
    return _SPACE_RE.sub(" ", text or "").strip()


@dataclass(frozen=True)
class CaptionSegment:
    start: float
    end: float
    text: str
    source: str
    confidence: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CaptionSegment":
        return cls(
            start=float(raw["start"]),
            end=float(raw["end"]),
            text=normalize_caption_text(str(raw["text"])),
            source=str(raw.get("source", "unknown")),
            confidence=(
                None
                if raw.get("confidence") is None
                else float(raw.get("confidence"))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CaptionResult:
    language: str
    segments: List[CaptionSegment]
    duration_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CaptionResult":
        segments = [CaptionSegment.from_dict(item) for item in raw.get("segments", [])]
        result = cls(
            language=str(raw.get("language", "unknown")),
            segments=segments,
            duration_seconds=(
                None
                if raw.get("duration_seconds") is None
                else float(raw.get("duration_seconds"))
            ),
        )
        validate_caption_segments(result.segments)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def validate_caption_segments(segments: Iterable[CaptionSegment]) -> None:
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if segment.start < 0:
            raise ValueError(f"caption segment {index} starts before zero")
        if segment.end < segment.start:
            raise ValueError(f"caption segment {index} ends before it starts")
        if index > 0 and segment.start < previous_end:
            raise ValueError(
                f"caption segment {index} overlaps the previous segment"
            )
        if not segment.text:
            raise ValueError(f"caption segment {index} has empty text")
        previous_end = segment.end
