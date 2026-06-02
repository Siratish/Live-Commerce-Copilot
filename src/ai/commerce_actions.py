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
from src.ai.retrieval import ProductPromoRetriever
from src.schemas import CaptionResult, CommerceAction, ProductCatalogItem, Promotion


PIN_PRODUCT_CARD = "PIN_PRODUCT_CARD"
SHOW_PROMO_CODE = "SHOW_PROMO_CODE"
SHOW_BUNDLE_RECOMMENDATION = "SHOW_BUNDLE_RECOMMENDATION"
START_FLASH_SALE_COUNTDOWN = "START_FLASH_SALE_COUNTDOWN"


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
) -> List[CommerceAction]:
    retriever = retriever or ProductPromoRetriever.build(catalog, promotions)
    decision_provider = decision_provider or DeterministicDecisionProvider()
    catalog_by_sku = {item.sku: item for item in catalog}
    promotions_by_code = {promotion.promo_code: promotion for promotion in promotions}
    actions: List[CommerceAction] = []
    emitted = set()
    state = CommerceSessionState()

    for index, segment in enumerate(captions.segments):
        next_text = captions.segments[index + 1].text if index + 1 < len(captions.segments) else ""
        previous_texts = tuple(
            prior.text for prior in captions.segments[max(0, index - 3) : index]
        )
        window = TranscriptWindow(
            index=index,
            start=segment.start,
            text=segment.text,
            next_text=next_text,
            previous_texts=previous_texts,
        )
        bundle_text = window.lookahead_text if has_bundle_cue(segment.text) else segment.text
        promotion_text = _promotion_context_text(segment.text, state.active_sku, catalog_by_sku)
        candidates = CommerceCandidates(
            products=retriever.retrieve_products(segment.text, top_k=5),
            bundle_products=retriever.retrieve_products(bundle_text, top_k=5),
            promotions=retriever.retrieve_promotions(promotion_text, top_k=5),
        )
        decision = decision_provider.decide(window, candidates, state, catalog, promotions)

        if decision.product_sku and decision.product_sku in catalog_by_sku:
            product = catalog_by_sku[decision.product_sku]
            if decision.product_sku not in state.mentioned_skus:
                state.mentioned_skus.append(decision.product_sku)

            if _should_emit(decision, PIN_PRODUCT_CARD):
                pin_key = (PIN_PRODUCT_CARD, decision.product_sku)
                suppress_repin = pin_key in emitted and has_bundle_cue(segment.text)
                if decision.product_sku != state.active_sku and not suppress_repin:
                    state.active_sku = decision.product_sku
                    state.active_bundle = None
                    if pin_key not in emitted:
                        emitted.add(pin_key)
                        state.pinned_skus.append(decision.product_sku)
                        actions.append(
                            CommerceAction(
                                timestamp=segment.start,
                                action_type=PIN_PRODUCT_CARD,
                                skus=[decision.product_sku],
                                confidence=round(decision.product_confidence, 3),
                                evidence_text=segment.text,
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
                        evidence_text=segment.text,
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
            key = (SHOW_BUNDLE_RECOMMENDATION, tuple(bundle_skus))
            state.active_bundle = bundle_skus
            for sku in bundle_skus:
                if sku not in state.mentioned_skus:
                    state.mentioned_skus.append(sku)
            if key not in emitted:
                emitted.add(key)
                first, second = catalog_by_sku[bundle_skus[0]], catalog_by_sku[bundle_skus[1]]
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=SHOW_BUNDLE_RECOMMENDATION,
                        skus=bundle_skus,
                        confidence=round(decision.bundle_confidence, 3),
                        evidence_text=segment.text,
                        display_payload={
                            "title": "Recommended bundle",
                            "products": [_product_payload(first), _product_payload(second)],
                            "reason": "Model-assisted transcript decision matched compatible catalog products.",
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
                        evidence_text=segment.text,
                        display_payload={
                            "title": "Flash sale countdown",
                            "duration_minutes": decision.flash_minutes,
                            "duration_seconds": decision.flash_minutes * 60,
                            "promo_code": state.active_promo_code,
                        },
                    )
                )

    return sorted(actions, key=lambda action: (action.timestamp, action.action_type))


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
