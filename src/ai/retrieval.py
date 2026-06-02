from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, Set
import heapq
import re

from src.schemas import ProductCatalogItem, Promotion


THAI_DIGITS = str.maketrans("\u0e50\u0e51\u0e52\u0e53\u0e54\u0e55\u0e56\u0e57\u0e58\u0e59", "0123456789")
THAI_TONE_MARKS = str.maketrans("", "", "\u0e48\u0e49\u0e4a\u0e4b")
WORD_RE = re.compile(r"[a-z0-9]+|[\u0e00-\u0e7f]+")
NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class ProductCandidate:
    item: ProductCatalogItem
    score: float


@dataclass(frozen=True)
class PromotionCandidate:
    promotion: Promotion
    score: float


@dataclass(frozen=True)
class _SearchRecord:
    obj: object
    text: str
    compact: str
    primary_compact: str
    ngrams: Set[str]
    primary_ngrams: Set[str]


def normalize_search_text(text: str) -> str:
    normalized = (text or "").translate(THAI_DIGITS).translate(THAI_TONE_MARKS)
    normalized = normalized.lower().replace("-", " ")
    return " ".join(WORD_RE.findall(normalized))


def compact_search_text(text: str) -> str:
    return "".join(WORD_RE.findall(normalize_search_text(text)))


def extract_numbers(text: str) -> List[int]:
    normalized = (text or "").translate(THAI_DIGITS)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    values = [int(value) for value in NUMBER_RE.findall(normalized)]
    for match in re.finditer(r"\d(?:[\s-]+\d{1,3})+", normalized):
        compact = re.sub(r"\D", "", match.group(0))
        if compact:
            values.append(int(compact))
    return values


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    return char_ngrams_from_compact(compact_search_text(text), n=n)


def char_ngrams_from_compact(compact: str, n: int = 3) -> Set[str]:
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def dice_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return (2.0 * len(left.intersection(right))) / (len(left) + len(right))


def containment_similarity(query_ngrams: Set[str], candidate_ngrams: Set[str]) -> float:
    if not query_ngrams or not candidate_ngrams:
        return 0.0
    return len(query_ngrams.intersection(candidate_ngrams)) / len(candidate_ngrams)


def best_substring_similarity(query_compact: str, candidate_compact: str) -> float:
    if not query_compact or not candidate_compact:
        return 0.0
    if candidate_compact in query_compact:
        return 1.0

    best = SequenceMatcher(None, query_compact, candidate_compact).ratio()
    candidate_len = len(candidate_compact)
    min_len = max(2, candidate_len - 2)
    max_len = min(len(query_compact), candidate_len + 3)
    for window_len in range(min_len, max_len + 1):
        for start in range(0, len(query_compact) - window_len + 1):
            window = query_compact[start : start + window_len]
            best = max(best, SequenceMatcher(None, window, candidate_compact).ratio())
    return best


def _join(values: Iterable[str]) -> str:
    return " ".join(value for value in values if value)


def product_search_text(item: ProductCatalogItem) -> str:
    return _join(
        [
            item.sku,
            item.product_name,
            item.brand,
            item.category,
            item.description,
            *item.tags,
            *[tag.replace("_", " ") for tag in item.tags],
            *item.compatible_with,
        ]
    )


def promotion_search_text(promotion: Promotion) -> str:
    return _join(
        [
            promotion.promo_code,
            promotion.promo_description,
            promotion.discount_type,
            str(promotion.discount_value),
            "live only" if promotion.live_only else "",
            *promotion.eligible_categories,
            *promotion.eligible_tags,
            *promotion.eligible_skus,
        ]
    )


class ProductPromoRetriever:
    def __init__(
        self,
        product_records: Sequence[_SearchRecord],
        promotion_records: Sequence[_SearchRecord],
    ):
        self.product_records = list(product_records)
        self.promotion_records = list(promotion_records)

    @classmethod
    def build(
        cls,
        catalog: Sequence[ProductCatalogItem],
        promotions: Sequence[Promotion],
    ) -> "ProductPromoRetriever":
        product_records = []
        for item in catalog:
            text = product_search_text(item)
            compact = compact_search_text(text)
            primary_compact = compact_search_text(item.product_name)
            product_records.append(
                _SearchRecord(
                    obj=item,
                    text=text,
                    compact=compact,
                    primary_compact=primary_compact,
                    ngrams=char_ngrams_from_compact(compact),
                    primary_ngrams=char_ngrams_from_compact(primary_compact),
                )
            )

        promotion_records = []
        for promotion in promotions:
            text = promotion_search_text(promotion)
            compact = compact_search_text(text)
            primary_compact = compact_search_text(promotion.promo_code)
            promotion_records.append(
                _SearchRecord(
                    obj=promotion,
                    text=text,
                    compact=compact,
                    primary_compact=primary_compact,
                    ngrams=char_ngrams_from_compact(compact),
                    primary_ngrams=char_ngrams_from_compact(primary_compact),
                )
            )
        return cls(product_records, promotion_records)

    def retrieve_products(self, text: str, top_k: int = 5) -> List[ProductCandidate]:
        query_compact = compact_search_text(text)
        query_ngrams = char_ngrams(text)
        query_numbers = extract_numbers(text)
        scored = [
            ProductCandidate(
                item=record.obj,  # type: ignore[arg-type]
                score=_score_product_record(record, query_compact, query_ngrams, query_numbers),
            )
            for record in self.product_records
        ]
        return [
            candidate
            for candidate in heapq.nlargest(top_k, scored, key=lambda item: item.score)
            if candidate.score > 0
        ]

    def retrieve_promotions(self, text: str, top_k: int = 5) -> List[PromotionCandidate]:
        query_compact = compact_search_text(text)
        query_ngrams = char_ngrams(text)
        query_numbers = extract_numbers(text)
        scored = [
            PromotionCandidate(
                promotion=record.obj,  # type: ignore[arg-type]
                score=_score_promotion_record(record, query_compact, query_ngrams, query_numbers),
            )
            for record in self.promotion_records
        ]
        return [
            candidate
            for candidate in heapq.nlargest(top_k, scored, key=lambda item: item.score)
            if candidate.score > 0
        ]


def _score_product_record(
    record: _SearchRecord,
    query_compact: str,
    query_ngrams: Set[str],
    query_numbers: Sequence[int],
) -> float:
    item = record.obj
    broad_score = dice_similarity(query_ngrams, record.ngrams)
    primary_score = dice_similarity(query_ngrams, record.primary_ngrams)
    primary_coverage = containment_similarity(query_ngrams, record.primary_ngrams)
    score = max(
        broad_score,
        primary_score,
        primary_coverage,
    )
    if record.primary_compact and record.primary_compact in query_compact:
        score = 1.0
    elif primary_score >= 0.35 or primary_coverage >= 0.65:
        score = max(score, best_substring_similarity(query_compact, record.primary_compact))

    if item.price in query_numbers or item.discount_price in query_numbers:  # type: ignore[attr-defined]
        boosted = min(0.95, max(score + 0.35, 0.5))
        score = max(score, boosted if score > 0.08 else 0.0)
    return score


def _score_promotion_record(
    record: _SearchRecord,
    query_compact: str,
    query_ngrams: Set[str],
    query_numbers: Sequence[int],
) -> float:
    promotion = record.obj
    promo_numbers = extract_numbers(promotion.promo_code) + [promotion.discount_value]  # type: ignore[attr-defined]
    if not _has_promo_intent(record.primary_compact, query_compact, query_numbers, promo_numbers):
        return 0.0

    score = max(
        dice_similarity(query_ngrams, record.ngrams),
        best_substring_similarity(query_compact, record.primary_compact),
    )
    if set(query_numbers).intersection(promo_numbers):
        score = max(score, min(0.95, score + 0.35), 0.8)
    return score


def _has_promo_intent(
    promo_code_compact: str,
    query_compact: str,
    query_numbers: Sequence[int],
    promo_numbers: Sequence[int],
) -> bool:
    if promo_code_compact and promo_code_compact in query_compact:
        return True
    if not set(query_numbers).intersection(promo_numbers):
        return False
    return _has_code_cue(query_compact) or _has_discount_cue(query_compact)


def _has_code_cue(query_compact: str) -> bool:
    return any(
        cue in query_compact
        for cue in [
            "code",
            "promo",
            "promotion",
            "coupon",
            compact_search_text("\u0e42\u0e04\u0e49\u0e14"),
            compact_search_text("\u0e04\u0e39\u0e1b\u0e2d\u0e07"),
            compact_search_text("\u0e42\u0e1b\u0e23"),
        ]
    )


def _has_discount_cue(query_compact: str) -> bool:
    return any(
        cue in query_compact
        for cue in [
            compact_search_text("\u0e2a\u0e48\u0e27\u0e19\u0e25\u0e14"),
            compact_search_text("\u0e25\u0e14"),
            "discount",
        ]
    )
