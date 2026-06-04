from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

from src.ai.captioning import load_cached_transcript
from src.ai.decision import (
    CommerceCandidates,
    CommerceDecision,
    CommerceDecisionProvider,
    CommerceSessionState,
    DeterministicDecisionProvider,
    TranscriptWindow,
    has_bundle_cue,
)
from src.ai.retrieval import ProductCandidate, ProductPromoRetriever
from src.schemas import CaptionResult, CaptionSegment, CommerceAction, ProductCatalogItem, Promotion


PIN_PRODUCT_CARD = "PIN_PRODUCT_CARD"
SHOW_PROMO_CODE = "SHOW_PROMO_CODE"
SHOW_BUNDLE_RECOMMENDATION = "SHOW_BUNDLE_RECOMMENDATION"
START_FLASH_SALE_COUNTDOWN = "START_FLASH_SALE_COUNTDOWN"
DEFAULT_DECISION_HISTORY_CHARS = 900
CURRENT_PRODUCT_SUPPORT_THRESHOLD = 0.48


def load_caption_result(path: Path) -> CaptionResult:
    return load_cached_transcript(path)


def save_commerce_actions(actions: Sequence[CommerceAction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([action.to_dict() for action in actions], handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_promo_code(text: str, promotions: Sequence[Promotion]) -> Optional[str]:
    retriever = ProductPromoRetriever.build([], promotions)
    candidates = retriever.retrieve_promotions(text, top_k=1)
    return candidates[0].promotion.promo_code if candidates and candidates[0].score >= 0.68 else None


def eligible_items_for_promotion(
    promotion: Promotion,
    catalog: Sequence[ProductCatalogItem],
) -> List[ProductCatalogItem]:
    return [item for item in catalog if promotion.applies_to(item)]


def is_compatible(first: ProductCatalogItem, second: ProductCatalogItem) -> bool:
    return second.sku in first.compatible_with or first.sku in second.compatible_with


def generate_commerce_actions(
    captions: CaptionResult,
    catalog: Sequence[ProductCatalogItem],
    promotions: Sequence[Promotion],
    decision_provider: Optional[CommerceDecisionProvider] = None,
    retriever: Optional[ProductPromoRetriever] = None,
    max_decision_history_chars: int = DEFAULT_DECISION_HISTORY_CHARS,
) -> List[CommerceAction]:
    retriever = retriever or ProductPromoRetriever.build(catalog, promotions)
    decision_provider = decision_provider or DeterministicDecisionProvider()
    catalog_by_sku = {item.sku: item for item in catalog}
    promotions_by_code = {promotion.promo_code: promotion for promotion in promotions}
    actions: List[CommerceAction] = []
    emitted = set()
    state = CommerceSessionState()
    history_segments = []

    for index, segment in enumerate(captions.segments):
        history_segments.append(segment)
        history_segments = _trim_history_segments(
            history_segments,
            max_decision_history_chars,
        )
        previous_texts = tuple(prior.text for prior in history_segments[:-1])
        window = TranscriptWindow(
            index=index,
            start=segment.start,
            text=segment.text,
            next_text="",
            previous_texts=previous_texts,
        )
        decision_text = window.history_text
        promotion_text = _promotion_context_text(decision_text, state.active_sku, catalog_by_sku)
        current_product_candidates = retriever.retrieve_products(segment.text, top_k=5)
        history_product_candidates = retriever.retrieve_products(decision_text, top_k=5)
        candidates = CommerceCandidates(
            products=_merge_product_candidates(
                current_product_candidates,
                history_product_candidates,
                top_k=5,
            ),
            bundle_products=retriever.retrieve_products(decision_text, top_k=5),
            promotions=retriever.retrieve_promotions(promotion_text, top_k=5),
        )
        decision = decision_provider.decide(window, candidates, state, catalog, promotions)
        action_count_before = len(actions)

        if decision.product_sku and decision.product_sku in catalog_by_sku:
            product = catalog_by_sku[decision.product_sku]
            if _should_emit(decision, PIN_PRODUCT_CARD):
                pin_key = (PIN_PRODUCT_CARD, decision.product_sku)
                suppress_repin = pin_key in emitted and has_bundle_cue(segment.text)
                current_support = _candidate_score(
                    current_product_candidates,
                    decision.product_sku,
                )
                supported_new_pin = (
                    not state.active_sku
                    or decision.product_sku == state.active_sku
                    or current_support >= CURRENT_PRODUCT_SUPPORT_THRESHOLD
                )
                if (
                    decision.product_sku != state.active_sku
                    and not suppress_repin
                    and supported_new_pin
                ):
                    state.active_sku = decision.product_sku
                    state.active_bundle = None
                    if pin_key not in emitted:
                        emitted.add(pin_key)
                        if decision.product_sku not in state.mentioned_skus:
                            state.mentioned_skus.append(decision.product_sku)
                        state.pinned_skus.append(decision.product_sku)
                        actions.append(
                            CommerceAction(
                                timestamp=segment.start,
                                action_type=PIN_PRODUCT_CARD,
                                skus=[decision.product_sku],
                                confidence=round(decision.product_confidence, 3),
                                evidence_text=decision_text,
                                display_payload={
                                    "title": "Pin product card",
                                    "product": _product_payload(product),
                                },
                            )
                        )

        if (
            decision.promo_code
            and decision.promo_code in promotions_by_code
            and _should_emit(decision, SHOW_PROMO_CODE)
        ):
            promotion = promotions_by_code[decision.promo_code]
            promo_items = eligible_items_for_promotion(promotion, catalog)
            key = (SHOW_PROMO_CODE, promotion.promo_code)
            state.active_promo_code = promotion.promo_code
            if key not in emitted:
                emitted.add(key)
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=SHOW_PROMO_CODE,
                        skus=[item.sku for item in promo_items],
                        confidence=round(decision.promo_confidence, 3),
                        evidence_text=decision_text,
                        display_payload={
                            "title": "Promo code detected",
                            "promo_code": promotion.promo_code,
                            "promo_description": promotion.promo_description,
                            "discount_type": promotion.discount_type,
                            "discount_value": promotion.discount_value,
                            "live_only": promotion.live_only,
                            "eligible_categories": promotion.eligible_categories,
                            "eligible_tags": promotion.eligible_tags,
                            "eligible_skus": promotion.eligible_skus,
                        },
                    )
                )

        if (
            decision.bundle_skus
            and _valid_bundle(decision.bundle_skus, catalog_by_sku)
            and _should_emit(decision, SHOW_BUNDLE_RECOMMENDATION)
        ):
            bundle_skus = decision.bundle_skus[:2]
            key = (SHOW_BUNDLE_RECOMMENDATION, frozenset(bundle_skus))
            if key not in emitted:
                emitted.add(key)
                state.active_bundle = bundle_skus
                for sku in bundle_skus:
                    if sku not in state.mentioned_skus:
                        state.mentioned_skus.append(sku)
                first, second = catalog_by_sku[bundle_skus[0]], catalog_by_sku[bundle_skus[1]]
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=SHOW_BUNDLE_RECOMMENDATION,
                        skus=bundle_skus,
                        confidence=round(decision.bundle_confidence, 3),
                        evidence_text=decision_text,
                        display_payload={
                            "title": "Recommended bundle",
                            "products": [_product_payload(first), _product_payload(second)],
                            "reason": "Transcript cues matched compatible catalog products.",
                        },
                    )
                )

        if decision.flash_minutes and _should_emit(decision, START_FLASH_SALE_COUNTDOWN):
            key = (START_FLASH_SALE_COUNTDOWN, segment.start)
            if key not in emitted:
                emitted.add(key)
                flash_skus = state.active_bundle or ([state.active_sku] if state.active_sku else [])
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=START_FLASH_SALE_COUNTDOWN,
                        skus=flash_skus,
                        confidence=round(decision.flash_confidence, 3),
                        evidence_text=decision_text,
                        display_payload={
                            "title": "Flash sale countdown",
                            "duration_minutes": decision.flash_minutes,
                            "duration_seconds": decision.flash_minutes * 60,
                            "promo_code": state.active_promo_code,
                        },
                    )
                )

        if len(actions) > action_count_before:
            history_segments = []

    return sorted(actions, key=lambda action: (action.timestamp, action.action_type))


def _trim_history_segments(
    segments: Sequence[CaptionSegment],
    max_chars: int,
) -> List[CaptionSegment]:
    trimmed = list(segments)
    limit = max(0, int(max_chars))
    while len(trimmed) > 1 and len(_join_history_text(trimmed)) > limit:
        trimmed.pop(0)
    return trimmed


def _join_history_text(segments: Sequence[CaptionSegment]) -> str:
    return " ".join(segment.text for segment in segments if segment.text).strip()


def _merge_product_candidates(
    primary: Sequence[ProductCandidate],
    secondary: Sequence[ProductCandidate],
    top_k: int,
) -> List[ProductCandidate]:
    by_sku: Dict[str, ProductCandidate] = {}
    order: List[str] = []
    for candidate in [*primary, *secondary]:
        sku = candidate.item.sku
        if sku not in by_sku:
            order.append(sku)
            by_sku[sku] = candidate
            continue
        existing = by_sku[sku]
        if candidate.score > existing.score:
            by_sku[sku] = candidate

    return sorted(
        by_sku.values(),
        key=lambda candidate: (-candidate.score, order.index(candidate.item.sku)),
    )[:top_k]


def _candidate_score(
    candidates: Sequence[ProductCandidate],
    sku: str,
) -> float:
    for candidate in candidates:
        if candidate.item.sku == sku:
            return candidate.score
    return 0.0


def _product_payload(item: ProductCatalogItem) -> Dict[str, Any]:
    return {
        "sku": item.sku,
        "product_name": item.product_name,
        "brand": item.brand,
        "category": item.category,
        "price": item.price,
        "discount_price": item.discount_price,
        "stock": item.stock,
        "deeplink": item.deeplink,
    }


def _promotion_context_text(
    text: str,
    active_sku: Optional[str],
    catalog_by_sku: Dict[str, ProductCatalogItem],
) -> str:
    if not active_sku or active_sku not in catalog_by_sku:
        return text
    item = catalog_by_sku[active_sku]
    context = " ".join(
        [
            item.sku,
            item.product_name,
            item.brand,
            item.category,
            *item.tags,
        ]
    )
    return f"{text} {context}".strip()


def _should_emit(decision: CommerceDecision, action_type: str) -> bool:
    return decision.action_type is None or decision.action_type == action_type


def _valid_bundle(
    skus: Sequence[str],
    catalog_by_sku: Dict[str, ProductCatalogItem],
) -> bool:
    if len(skus) < 2:
        return False
    first = catalog_by_sku.get(skus[0])
    second = catalog_by_sku.get(skus[1])
    if not first or not second:
        return False
    return is_compatible(first, second)
