from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from src.ai.retrieval import ProductCandidate, PromotionCandidate, extract_numbers, normalize_search_text
from src.schemas import ProductCatalogItem, Promotion


BUNDLE_CUES = [
    "\u0e04\u0e39\u0e48\u0e01\u0e31\u0e19",
    "\u0e43\u0e0a\u0e49\u0e04\u0e39\u0e48",
    "\u0e15\u0e32\u0e21\u0e14\u0e49\u0e27\u0e22",
    "\u0e01\u0e48\u0e2d\u0e19\u0e41\u0e25\u0e49\u0e27",
    "\u0e04\u0e39\u0e48\u0e01\u0e31\u0e1a",
    "bundle",
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
    previous_texts: Sequence[str] = field(default_factory=tuple)

    @property
    def lookahead_text(self) -> str:
        return f"{self.text} {self.next_text}".strip()

    @property
    def context_text(self) -> str:
        return " ".join([*self.previous_texts, self.text, self.next_text]).strip()

    @property
    def history_text(self) -> str:
        return " ".join([*self.previous_texts, self.text]).strip()


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
    action_type: Optional[str] = None
    confidence: float = 0.0
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


class DeterministicDecisionProvider:
    def __init__(
        self,
        product_threshold: float = 0.48,
        promo_threshold: float = 0.68,
        bundle_threshold: float = 0.35,
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
        bundle_context = window.history_text
        if has_bundle_cue(bundle_context):
            bundle_skus, bundle_confidence = _decide_bundle(
                candidates.bundle_products,
                state,
                catalog,
                self.bundle_threshold,
                bundle_context,
            )

        flash_minutes = extract_flash_minutes(window.history_text)

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
    text: str = "",
) -> tuple[Optional[List[str]], float]:
    skus: List[str] = []
    confidence = 0.0
    for candidate in candidates:
        if candidate.score < threshold:
            continue
        if candidate.item.sku not in skus:
            skus.append(candidate.item.sku)
            confidence = max(confidence, candidate.score)
        if len(skus) >= 2:
            pair = _first_compatible_pair(skus, catalog)
            if pair:
                return _order_skus_by_mention(pair, candidates, catalog, text), max(0.89, confidence)

    for sku in skus:
        partner = _latest_compatible_history_sku(sku, state.mentioned_skus, catalog)
        if partner:
            pair = [sku, partner]
            return _order_skus_by_mention(pair, candidates, catalog, text), max(0.84, confidence)

    return None, 0.0


def _first_compatible_pair(
    skus: Sequence[str],
    catalog: Sequence[ProductCatalogItem],
) -> Optional[List[str]]:
    items = {item.sku: item for item in catalog}
    for index, first_sku in enumerate(skus):
        first = items.get(first_sku)
        if not first:
            continue
        for second_sku in skus[index + 1 :]:
            second = items.get(second_sku)
            if not second:
                continue
            if second.sku in first.compatible_with or first.sku in second.compatible_with:
                return [first_sku, second_sku]
    return None


def _order_skus_by_mention(
    skus: Sequence[str],
    candidates: Sequence[ProductCandidate],
    catalog: Sequence[ProductCatalogItem],
    text: str,
) -> List[str]:
    positions = {
        candidate.item.sku: _product_mention_position(candidate.item, text)
        for candidate in candidates
    }
    catalog_by_sku = {item.sku: item for item in catalog}
    for sku in skus:
        if sku not in positions and sku in catalog_by_sku:
            positions[sku] = _product_mention_position(catalog_by_sku[sku], text)
    known_positions = [positions.get(sku) for sku in skus if positions.get(sku) is not None]
    if len(known_positions) < 2:
        return list(skus)
    return sorted(
        list(skus),
        key=lambda sku: (
            positions.get(sku) is None,
            positions.get(sku, 10**9),
            skus.index(sku),
        ),
    )


def _product_mention_position(item: ProductCatalogItem, text: str) -> Optional[int]:
    query = normalize_search_text(text)
    query_compact = query.replace(" ", "")
    terms = _product_mention_terms(item)
    positions = [
        position
        for term in terms
        for position in [query_compact.find(term)]
        if position >= 0
    ]
    return min(positions) if positions else None


def _product_mention_terms(item: ProductCatalogItem) -> List[str]:
    raw_terms = [
        item.product_name,
        item.brand,
        *item.tags,
        *[tag.replace("_", " ") for tag in item.tags],
    ]
    terms = []
    for raw_term in raw_terms:
        compact = normalize_search_text(raw_term).replace(" ", "")
        if len(compact) >= 4:
            terms.append(compact)
        terms.extend(
            token
            for token in normalize_search_text(raw_term).split()
            if len(token) >= 4
        )
    return sorted(set(terms), key=len, reverse=True)


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
