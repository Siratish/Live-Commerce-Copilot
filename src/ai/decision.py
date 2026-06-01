from __future__ import annotations

from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence

from src.ai.retrieval import ProductCandidate, PromotionCandidate, extract_numbers, normalize_search_text
from src.schemas import ProductCatalogItem, Promotion


BUNDLE_CUES = [
    "\u0e04\u0e39\u0e48\u0e01\u0e31\u0e19",
    "\u0e43\u0e0a\u0e49\u0e04\u0e39\u0e48",
    "\u0e15\u0e32\u0e21\u0e14\u0e49\u0e27\u0e22",
    "\u0e01\u0e48\u0e2d\u0e19\u0e41\u0e25\u0e49\u0e27",
    "routine",
    "\u0e23\u0e39\u0e17\u0e34\u0e19",
]
FLASH_CUES = [
    "\u0e40\u0e2b\u0e25\u0e37\u0e2d\u0e40\u0e27\u0e25\u0e32",
    "\u0e40\u0e2b\u0e25\u0e37\u0e2d",
    "flash",
    "\u0e44\u0e25\u0e1f\u0e4c",
]
MINUTE_WORDS = ["\u0e19\u0e32\u0e17\u0e35", "minute"]


@dataclass(frozen=True)
class TranscriptWindow:
    index: int
    start: float
    text: str
    next_text: str = ""

    @property
    def lookahead_text(self) -> str:
        return f"{self.text} {self.next_text}".strip()


@dataclass(frozen=True)
class CommerceCandidates:
    products: List[ProductCandidate]
    bundle_products: List[ProductCandidate]
    promotions: List[PromotionCandidate]


@dataclass
class CommerceSessionState:
    active_sku: Optional[str] = None
    active_bundle: Optional[List[str]] = None
    active_promo_code: Optional[str] = None
    mentioned_skus: List[str] = field(default_factory=list)
    pinned_skus: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CommerceDecision:
    product_sku: Optional[str] = None
    product_confidence: float = 0.0
    promo_code: Optional[str] = None
    promo_confidence: float = 0.0
    bundle_skus: Optional[List[str]] = None
    bundle_confidence: float = 0.0
    flash_minutes: Optional[int] = None
    flash_confidence: float = 0.0


class CommerceDecisionProvider(Protocol):
    def decide(
        self,
        window: TranscriptWindow,
        candidates: CommerceCandidates,
        state: CommerceSessionState,
        catalog: Sequence[ProductCatalogItem],
        promotions: Sequence[Promotion],
    ) -> CommerceDecision:
        ...


ModelDecisionCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class CallableModelDecisionProvider:
    """Adapter for an optional local model or LLM structured-output function."""

    def __init__(
        self,
        model_callable: ModelDecisionCallable,
        fallback: Optional[CommerceDecisionProvider] = None,
    ):
        self.model_callable = model_callable
        self.fallback = fallback or DeterministicDecisionProvider()

    def decide(
        self,
        window: TranscriptWindow,
        candidates: CommerceCandidates,
        state: CommerceSessionState,
        catalog: Sequence[ProductCatalogItem],
        promotions: Sequence[Promotion],
    ) -> CommerceDecision:
        payload = _model_payload(window, candidates, state)
        try:
            raw_decision = self.model_callable(payload)
            return _decision_from_mapping(raw_decision)
        except Exception:
            return self.fallback.decide(window, candidates, state, catalog, promotions)


class DeterministicDecisionProvider:
    def __init__(
        self,
        product_threshold: float = 0.48,
        promo_threshold: float = 0.68,
        bundle_threshold: float = 0.48,
    ):
        self.product_threshold = product_threshold
        self.promo_threshold = promo_threshold
        self.bundle_threshold = bundle_threshold

    def decide(
        self,
        window: TranscriptWindow,
        candidates: CommerceCandidates,
        state: CommerceSessionState,
        catalog: Sequence[ProductCatalogItem],
        promotions: Sequence[Promotion],
    ) -> CommerceDecision:
        product = _first_above(candidates.products, self.product_threshold)
        promotion = _first_above(candidates.promotions, self.promo_threshold)

        bundle_skus = None
        bundle_confidence = 0.0
        if has_bundle_cue(window.text):
            bundle_skus, bundle_confidence = _decide_bundle(
                candidates.bundle_products,
                state,
                catalog,
                self.bundle_threshold,
            )

        flash_minutes = extract_flash_minutes(window.lookahead_text if has_flash_cue(window.text) else window.text)

        return CommerceDecision(
            product_sku=product.item.sku if product else None,
            product_confidence=product.score if product else 0.0,
            promo_code=promotion.promotion.promo_code if promotion else None,
            promo_confidence=promotion.score if promotion else 0.0,
            bundle_skus=bundle_skus,
            bundle_confidence=bundle_confidence,
            flash_minutes=flash_minutes,
            flash_confidence=0.87 if flash_minutes else 0.0,
        )


def has_bundle_cue(text: str) -> bool:
    normalized = normalize_search_text(text)
    return any(normalize_search_text(cue) in normalized for cue in BUNDLE_CUES)


def has_flash_cue(text: str) -> bool:
    normalized = normalize_search_text(text)
    return any(normalize_search_text(cue) in normalized for cue in FLASH_CUES)


def extract_flash_minutes(text: str) -> Optional[int]:
    normalized = normalize_search_text(text)
    if not any(normalize_search_text(word) in normalized for word in MINUTE_WORDS):
        return None
    if not has_flash_cue(text):
        return None
    numbers = extract_numbers(text)
    return numbers[-1] if numbers else None


def _model_payload(
    window: TranscriptWindow,
    candidates: CommerceCandidates,
    state: CommerceSessionState,
) -> Mapping[str, Any]:
    return {
        "window": {
            "index": window.index,
            "start": window.start,
            "text": window.text,
            "next_text": window.next_text,
        },
        "state": {
            "active_sku": state.active_sku,
            "active_bundle": state.active_bundle,
            "active_promo_code": state.active_promo_code,
            "mentioned_skus": state.mentioned_skus,
        },
        "product_candidates": [
            {
                "sku": candidate.item.sku,
                "product_name": candidate.item.product_name,
                "brand": candidate.item.brand,
                "category": candidate.item.category,
                "score": candidate.score,
            }
            for candidate in candidates.products
        ],
        "bundle_product_candidates": [
            {
                "sku": candidate.item.sku,
                "product_name": candidate.item.product_name,
                "score": candidate.score,
            }
            for candidate in candidates.bundle_products
        ],
        "promotion_candidates": [
            {
                "promo_code": candidate.promotion.promo_code,
                "description": candidate.promotion.promo_description,
                "score": candidate.score,
            }
            for candidate in candidates.promotions
        ],
        "expected_output": {
            "product_sku": "string|null",
            "product_confidence": "float",
            "promo_code": "string|null",
            "promo_confidence": "float",
            "bundle_skus": "list[string]|null",
            "bundle_confidence": "float",
            "flash_minutes": "int|null",
            "flash_confidence": "float",
        },
    }


def _decision_from_mapping(raw: Mapping[str, Any]) -> CommerceDecision:
    if not isinstance(raw, Mapping):
        raise TypeError("model decision must be a mapping")

    bundle_skus = raw.get("bundle_skus")
    if bundle_skus is not None:
        if not isinstance(bundle_skus, RuntimeSequence) or isinstance(bundle_skus, (str, bytes)):
            raise TypeError("bundle_skus must be a sequence")
        bundle_skus = [str(value) for value in bundle_skus if value]

    flash_minutes = raw.get("flash_minutes")
    return CommerceDecision(
        product_sku=_optional_string(raw.get("product_sku")),
        product_confidence=_optional_float(raw.get("product_confidence")),
        promo_code=_optional_string(raw.get("promo_code")),
        promo_confidence=_optional_float(raw.get("promo_confidence")),
        bundle_skus=bundle_skus,
        bundle_confidence=_optional_float(raw.get("bundle_confidence")),
        flash_minutes=None if flash_minutes is None else int(flash_minutes),
        flash_confidence=_optional_float(raw.get("flash_confidence")),
    )


def _optional_string(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def _optional_float(value: Any) -> float:
    return 0.0 if value in (None, "") else float(value)


def _first_above(candidates, threshold: float):
    if not candidates:
        return None
    first = candidates[0]
    return first if first.score >= threshold else None


def _decide_bundle(
    candidates: Sequence[ProductCandidate],
    state: CommerceSessionState,
    catalog: Sequence[ProductCatalogItem],
    threshold: float,
) -> tuple[Optional[List[str]], float]:
    skus: List[str] = []
    confidence = 0.0
    for candidate in candidates:
        if candidate.score < threshold:
            continue
        if candidate.item.sku not in skus:
            skus.append(candidate.item.sku)
            confidence = max(confidence, candidate.score)
        if len(skus) == 2:
            return skus, max(0.89, confidence)

    if len(skus) == 1:
        partner = _latest_compatible_history_sku(skus[0], state.mentioned_skus, catalog)
        if partner:
            return [skus[0], partner], max(0.84, confidence)

    return None, 0.0


def _latest_compatible_history_sku(
    sku: str,
    history: Sequence[str],
    catalog: Sequence[ProductCatalogItem],
) -> Optional[str]:
    items = {item.sku: item for item in catalog}
    current = items.get(sku)
    if not current:
        return None
    for prior_sku in reversed(history):
        if prior_sku == sku or prior_sku not in items:
            continue
        prior = items[prior_sku]
        if prior.sku in current.compatible_with or current.sku in prior.compatible_with:
            return prior.sku
    return None
