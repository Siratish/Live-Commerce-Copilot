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


@dataclass(frozen=True)
class ProductCatalogItem:
    sku: str
    product_name: str
    brand: str
    category: str
    price: int
    discount_price: int
    promo_code: str
    promo_description: str
    stock: int
    description: str
    tags: List[str]
    compatible_with: List[str]
    deeplink: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProductCatalogItem":
        return cls(
            sku=str(raw["sku"]).strip(),
            product_name=str(raw["product_name"]).strip(),
            brand=str(raw["brand"]).strip(),
            category=str(raw["category"]).strip(),
            price=int(raw["price"]),
            discount_price=int(raw["discount_price"]),
            promo_code=str(raw["promo_code"]).strip().upper(),
            promo_description=str(raw["promo_description"]).strip(),
            stock=int(raw["stock"]),
            description=str(raw["description"]).strip(),
            tags=_split_semicolon(raw.get("tags", "")),
            compatible_with=_split_semicolon(raw.get("compatible_with", "")),
            deeplink=str(raw["deeplink"]).strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "product_name": self.product_name,
            "brand": self.brand,
            "category": self.category,
            "price": self.price,
            "discount_price": self.discount_price,
            "promo_code": self.promo_code,
            "promo_description": self.promo_description,
            "stock": self.stock,
            "description": self.description,
            "tags": self.tags,
            "compatible_with": self.compatible_with,
            "deeplink": self.deeplink,
        }


@dataclass(frozen=True)
class CommerceAction:
    timestamp: float
    action_type: str
    skus: List[str]
    confidence: float
    evidence_text: str
    display_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "skus": self.skus,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "display_payload": self.display_payload,
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


def repair_caption_timestamps(
    segments: Iterable[CaptionSegment],
    min_duration_seconds: float = 0.05,
) -> List[CaptionSegment]:
    """Clamp ASR segment timestamps into a valid, non-overlapping timeline."""
    repaired: List[CaptionSegment] = []
    previous_end = 0.0

    for segment in segments:
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))

        if start < previous_end:
            start = previous_end
        if end <= start:
            end = start + min_duration_seconds

        start = round(start, 3)
        end = round(end, 3)
        repaired_segment = CaptionSegment(
            start=start,
            end=end,
            text=segment.text,
            source=segment.source,
            confidence=segment.confidence,
        )
        repaired.append(repaired_segment)
        previous_end = end

    validate_caption_segments(repaired)
    return repaired


def _split_semicolon(value: Any) -> List[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]
