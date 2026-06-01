from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import csv

from src.schemas import ProductCatalogItem, Promotion


def load_product_catalog(path: Path) -> List[ProductCatalogItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [ProductCatalogItem.from_dict(row) for row in rows]


def load_promotions(path: Path) -> List[Promotion]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [Promotion.from_dict(row) for row in rows]


def catalog_by_sku(items: Iterable[ProductCatalogItem]) -> Dict[str, ProductCatalogItem]:
    return {item.sku: item for item in items}
