from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re

from src.ai.captioning import load_cached_transcript
from src.data.catalog import catalog_by_sku
from src.schemas import CaptionResult, CaptionSegment, CommerceAction, ProductCatalogItem


PIN_PRODUCT_CARD = "PIN_PRODUCT_CARD"
SHOW_PRICE_DROP = "SHOW_PRICE_DROP"
SHOW_PROMO_CODE = "SHOW_PROMO_CODE"
SHOW_BUNDLE_RECOMMENDATION = "SHOW_BUNDLE_RECOMMENDATION"
START_FLASH_SALE_COUNTDOWN = "START_FLASH_SALE_COUNTDOWN"
SHOW_BUY_DEEPLINK = "SHOW_BUY_DEEPLINK"

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
THAI_TONE_MARKS = str.maketrans("", "", "่้๊๋")
WORD_RE = re.compile(r"[a-z0-9]+|[\u0e00-\u0e7f]+")
NUMBER_RE = re.compile(r"\d+")

PRODUCT_ALIASES: Dict[str, List[str]] = {
    "SKU001": [
        "vitamin c serum",
        "วิตามินซีซีรั่ม",
        "วิตามิน ซี ซีรั่ม",
        "ซีรั่มวิตามินซี",
    ],
    "SKU002": [
        "green tea cleanser",
        "กรีนทีคลีนเซอร์",
        "cleanser",
        "cleanse",
        "คลีนเซอร์",
    ],
}

PROMO_VARIANTS: Dict[str, List[str]] = {
    "LIVE25": ["live25", "live 25", "life25", "life 25", "ไลฟ์25", "ไลฟ์ 25"],
    "WELLNESS15": ["wellness15", "wellness 15"],
    "TECH100": ["tech100", "tech 100"],
}


@dataclass(frozen=True)
class ProductMention:
    item: ProductCatalogItem
    confidence: float
    match_text: str
    position: int


def load_caption_result(path: Path) -> CaptionResult:
    return load_cached_transcript(path)


def save_commerce_actions(actions: Sequence[CommerceAction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([action.to_dict() for action in actions], handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def compact_text(text: str) -> str:
    return "".join(tokenize(text))


def tokenize(text: str) -> List[str]:
    normalized = (text or "").translate(THAI_DIGITS).translate(THAI_TONE_MARKS).lower()
    return WORD_RE.findall(normalized)


def extract_numbers(text: str) -> List[int]:
    return [int(value) for value in NUMBER_RE.findall((text or "").translate(THAI_DIGITS))]


def aliases_for_product(item: ProductCatalogItem) -> List[str]:
    aliases = [
        item.product_name,
        item.product_name.replace("Vitamin C", "วิตามินซี"),
        item.brand,
        *item.tags,
        *[tag.replace("_", " ") for tag in item.tags],
        *PRODUCT_ALIASES.get(item.sku, []),
    ]
    seen = set()
    result = []
    for alias in aliases:
        normalized = normalize_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def find_product_mentions(
    text: str,
    catalog: Sequence[ProductCatalogItem],
    min_confidence: float = 0.74,
) -> List[ProductMention]:
    normalized = normalize_text(text)
    compact = compact_text(text)
    mentions: List[ProductMention] = []

    for item in catalog:
        best: Optional[ProductMention] = None
        for alias in aliases_for_product(item):
            alias_compact = compact_text(alias)
            if not alias_compact:
                continue
            position = compact.find(alias_compact)
            if position < 0 and alias in normalized:
                position = normalized.find(alias)
            if position < 0:
                continue

            confidence = 0.94
            if alias == normalize_text(item.product_name):
                confidence = 0.99
            elif alias in {normalize_text(tag) for tag in item.tags}:
                confidence = 0.78
            elif alias == normalize_text(item.brand):
                confidence = 0.70

            mention = ProductMention(
                item=item,
                confidence=confidence,
                match_text=alias,
                position=position,
            )
            if best is None or mention.confidence > best.confidence:
                best = mention
        if best and best.confidence >= min_confidence:
            mentions.append(best)

    return sorted(mentions, key=lambda mention: (mention.position, -mention.confidence))


def normalize_promo_code(text: str, catalog: Sequence[ProductCatalogItem]) -> Optional[str]:
    compact = compact_text(text)
    available_codes = {item.promo_code for item in catalog}
    for code in available_codes:
        if compact_text(code) in compact:
            return code

    for code, variants in PROMO_VARIANTS.items():
        if code not in available_codes:
            continue
        for variant in variants:
            if compact_text(variant) in compact:
                return code
    return None


def is_compatible(first: ProductCatalogItem, second: ProductCatalogItem) -> bool:
    return second.sku in first.compatible_with or first.sku in second.compatible_with


def _product_payload(item: ProductCatalogItem) -> Dict[str, Any]:
    return {
        "sku": item.sku,
        "product_name": item.product_name,
        "brand": item.brand,
        "category": item.category,
        "price": item.price,
        "discount_price": item.discount_price,
        "promo_code": item.promo_code,
        "promo_description": item.promo_description,
        "stock": item.stock,
        "deeplink": item.deeplink,
    }


def _find_price_window(
    segments: Sequence[CaptionSegment],
    start_index: int,
    item: ProductCatalogItem,
    window: int = 3,
) -> Optional[Tuple[float, str]]:
    window_segments = segments[start_index : start_index + window]
    original_index = None
    discount_index = None
    evidence: List[str] = []

    for offset, segment in enumerate(window_segments):
        numbers = extract_numbers(segment.text)
        if item.price in numbers and original_index is None:
            original_index = start_index + offset
            evidence.append(segment.text)
        if item.discount_price in numbers and discount_index is None:
            discount_index = start_index + offset
            evidence.append(segment.text)

    if original_index is None or discount_index is None:
        return None

    timestamp = (
        segments[discount_index].start
        if original_index == start_index
        else segments[original_index].start
    )
    return timestamp, " ".join(evidence)


def _extract_flash_minutes(text: str) -> Optional[int]:
    normalized = normalize_text(text)
    if "นาที" not in normalized and "minute" not in normalized:
        return None
    if not any(token in normalized for token in ["เหลือเวลา", "เหลือ", "flash", "ไลฟ์"]):
        return None
    numbers = extract_numbers(text)
    return numbers[-1] if numbers else None


def generate_commerce_actions(
    captions: CaptionResult,
    catalog: Sequence[ProductCatalogItem],
) -> List[CommerceAction]:
    items_by_sku = catalog_by_sku(catalog)
    actions: List[CommerceAction] = []
    emitted = set()
    active_sku: Optional[str] = None
    active_promo_code: Optional[str] = None
    active_bundle: Optional[List[str]] = None

    for index, segment in enumerate(captions.segments):
        mentions = find_product_mentions(segment.text, catalog)
        strong_mentions = [mention for mention in mentions if mention.confidence >= 0.9]

        if strong_mentions:
            first = strong_mentions[0]
            if first.item.sku != active_sku:
                active_sku = first.item.sku
                active_bundle = None
                key = (PIN_PRODUCT_CARD, active_sku)
                if key not in emitted:
                    emitted.add(key)
                    actions.append(
                        CommerceAction(
                            timestamp=segment.start,
                            action_type=PIN_PRODUCT_CARD,
                            skus=[active_sku],
                            confidence=first.confidence,
                            evidence_text=segment.text,
                            display_payload={
                                "title": "Pin product card",
                                "product": _product_payload(first.item),
                            },
                        )
                    )

                price_window = _find_price_window(captions.segments, index, first.item)
                if price_window:
                    timestamp, evidence = price_window
                    price_key = (SHOW_PRICE_DROP, first.item.sku)
                    if price_key not in emitted:
                        emitted.add(price_key)
                        actions.append(
                            CommerceAction(
                                timestamp=timestamp,
                                action_type=SHOW_PRICE_DROP,
                                skus=[first.item.sku],
                                confidence=0.95,
                                evidence_text=evidence,
                                display_payload={
                                    "title": "Live price drop",
                                    "sku": first.item.sku,
                                    "product_name": first.item.product_name,
                                    "original_price": first.item.price,
                                    "discount_price": first.item.discount_price,
                                    "currency": "THB",
                                },
                            )
                        )

        promo_code = normalize_promo_code(segment.text, catalog)
        if promo_code:
            active_promo_code = promo_code
            promo_items = [item for item in catalog if item.promo_code == promo_code]
            key = (SHOW_PROMO_CODE, promo_code)
            if key not in emitted:
                emitted.add(key)
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=SHOW_PROMO_CODE,
                        skus=[item.sku for item in promo_items],
                        confidence=0.92,
                        evidence_text=segment.text,
                        display_payload={
                            "title": "Promo code detected",
                            "promo_code": promo_code,
                            "promo_description": promo_items[0].promo_description
                            if promo_items
                            else "",
                        },
                    )
                )

        ordered_mentions = [mention.item for mention in mentions if mention.confidence >= 0.74]
        if len(ordered_mentions) >= 2:
            first, second = ordered_mentions[0], ordered_mentions[1]
            if is_compatible(first, second):
                bundle_skus = [first.sku, second.sku]
                active_bundle = bundle_skus
                key = (SHOW_BUNDLE_RECOMMENDATION, tuple(bundle_skus))
                if key not in emitted:
                    emitted.add(key)
                    actions.append(
                        CommerceAction(
                            timestamp=segment.start,
                            action_type=SHOW_BUNDLE_RECOMMENDATION,
                            skus=bundle_skus,
                            confidence=0.89,
                            evidence_text=segment.text,
                            display_payload={
                                "title": "Recommended bundle",
                                "products": [_product_payload(first), _product_payload(second)],
                                "reason": "Host mentioned using these compatible products together.",
                            },
                        )
                    )

        flash_minutes = _extract_flash_minutes(segment.text)
        if flash_minutes:
            key = (START_FLASH_SALE_COUNTDOWN, segment.start)
            if key not in emitted:
                emitted.add(key)
                flash_skus = active_bundle or ([active_sku] if active_sku else [])
                actions.append(
                    CommerceAction(
                        timestamp=segment.start,
                        action_type=START_FLASH_SALE_COUNTDOWN,
                        skus=flash_skus,
                        confidence=0.87,
                        evidence_text=segment.text,
                        display_payload={
                            "title": "Flash sale countdown",
                            "duration_minutes": flash_minutes,
                            "duration_seconds": flash_minutes * 60,
                            "promo_code": active_promo_code,
                        },
                    )
                )

        if any(term in normalize_text(segment.text) for term in ["กดซื้อ", "ลิงค์", "link", "checkout"]):
            buy_skus = active_bundle or ([active_sku] if active_sku else [])
            if buy_skus:
                key = (SHOW_BUY_DEEPLINK, tuple(buy_skus))
                if key not in emitted:
                    emitted.add(key)
                    actions.append(
                        CommerceAction(
                            timestamp=segment.start,
                            action_type=SHOW_BUY_DEEPLINK,
                            skus=buy_skus,
                            confidence=0.90,
                            evidence_text=segment.text,
                            display_payload={
                                "title": "Show buy link",
                                "deeplinks": [
                                    {
                                        "sku": sku,
                                        "product_name": items_by_sku[sku].product_name,
                                        "deeplink": items_by_sku[sku].deeplink,
                                    }
                                    for sku in buy_skus
                                    if sku in items_by_sku
                                ],
                            },
                        )
                    )

    return sorted(actions, key=lambda action: (action.timestamp, action.action_type))
