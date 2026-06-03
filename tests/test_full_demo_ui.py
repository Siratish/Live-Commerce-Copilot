from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.data.catalog import load_product_catalog, load_promotions
from src.schemas import ProductCatalogItem, Promotion
from src.utils.full_demo_ui import (
    build_catalog_scene_html,
    save_product_catalog_csv,
    save_promotions_csv,
)


class FullDemoUiTests(unittest.TestCase):
    def test_catalog_scene_html_includes_products_and_promotions(self) -> None:
        catalog = [
            ProductCatalogItem(
                sku="SKU001",
                product_name="Vitamin C Serum",
                brand="DemoBeauty",
                category="Skincare",
                price=399,
                discount_price=299,
                stock=120,
                description="Brightening serum",
                tags=["serum", "skincare"],
                compatible_with=["SKU002"],
                deeplink="https://trueid.example/live/product/SKU001",
            )
        ]
        promotions = [
            Promotion(
                promo_code="LIVE25",
                promo_description="Extra 25% discount",
                discount_type="percent",
                discount_value=25,
                live_only=True,
                eligible_categories=["Skincare"],
                eligible_tags=["skincare"],
                eligible_skus=["SKU001"],
                required_min_stock=1,
            )
        ]

        html = build_catalog_scene_html(catalog, promotions)

        self.assertIn("Vitamin C Serum", html)
        self.assertIn("LIVE25", html)
        self.assertIn("THB 299", html)

    def test_session_catalog_csv_roundtrip(self) -> None:
        item = ProductCatalogItem(
            sku="SKU009",
            product_name="Notebook Product",
            brand="DemoBrand",
            category="Skincare",
            price=500,
            discount_price=399,
            stock=20,
            description="Notebook-added product",
            tags=["demo", "skincare"],
            compatible_with=["SKU001"],
            deeplink="https://trueid.example/live/product/SKU009",
        )
        promotion = Promotion(
            promo_code="NEWLIVE",
            promo_description="Notebook promo",
            discount_type="percent",
            discount_value=10,
            live_only=True,
            eligible_categories=["Skincare"],
            eligible_tags=["skincare"],
            eligible_skus=["SKU009"],
            required_min_stock=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.csv"
            promotions_path = Path(temp_dir) / "promotions.csv"
            save_product_catalog_csv([item], catalog_path)
            save_promotions_csv([promotion], promotions_path)

            loaded_catalog = load_product_catalog(catalog_path)
            loaded_promotions = load_promotions(promotions_path)

        self.assertEqual(loaded_catalog[0].sku, "SKU009")
        self.assertEqual(loaded_catalog[0].tags, ["demo", "skincare"])
        self.assertEqual(loaded_catalog[0].compatible_with, ["SKU001"])
        self.assertEqual(loaded_promotions[0].promo_code, "NEWLIVE")
        self.assertTrue(loaded_promotions[0].live_only)


if __name__ == "__main__":
    unittest.main()
