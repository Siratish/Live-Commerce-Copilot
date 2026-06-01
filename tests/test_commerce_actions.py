from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from src.ai.commerce_actions import (
    PIN_PRODUCT_CARD,
    SHOW_BUNDLE_RECOMMENDATION,
    SHOW_BUY_DEEPLINK,
    SHOW_PRICE_DROP,
    SHOW_PROMO_CODE,
    START_FLASH_SALE_COUNTDOWN,
    generate_commerce_actions,
    load_caption_result,
    normalize_promo_code,
)
from src.data.catalog import load_product_catalog
from src.utils.action_timeline import build_action_timeline_html


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = REPO_ROOT / "data" / "demo" / "audio_1_captions.json"
CATALOG = REPO_ROOT / "data" / "demo" / "product_catalog.csv"
AUDIO = REPO_ROOT / "data" / "demo" / "audio" / "1.mp3"


class CommerceActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.captions = load_caption_result(CAPTIONS)
        self.catalog = load_product_catalog(CATALOG)
        self.actions = generate_commerce_actions(self.captions, self.catalog)

    def test_expected_action_sequence(self) -> None:
        sequence = [(action.action_type, action.skus, round(action.timestamp, 2)) for action in self.actions]
        self.assertEqual(
            sequence,
            [
                (PIN_PRODUCT_CARD, ["SKU001"], 8.30),
                (SHOW_PRICE_DROP, ["SKU001"], 13.80),
                (SHOW_PROMO_CODE, ["SKU001", "SKU002", "SKU007"], 29.36),
                (PIN_PRODUCT_CARD, ["SKU002"], 49.08),
                (SHOW_PRICE_DROP, ["SKU002"], 53.68),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU002", "SKU001"], 63.08),
                (START_FLASH_SALE_COUNTDOWN, ["SKU002", "SKU001"], 70.58),
                (SHOW_BUY_DEEPLINK, ["SKU002", "SKU001"], 74.08),
            ],
        )

    def test_price_extraction_payloads(self) -> None:
        price_actions = [action for action in self.actions if action.action_type == SHOW_PRICE_DROP]
        self.assertEqual(price_actions[0].display_payload["original_price"], 399)
        self.assertEqual(price_actions[0].display_payload["discount_price"], 299)
        self.assertEqual(price_actions[1].display_payload["original_price"], 259)
        self.assertEqual(price_actions[1].display_payload["discount_price"], 199)

    def test_promo_normalization_variants(self) -> None:
        self.assertEqual(normalize_promo_code("Code Life 25", self.catalog), "LIVE25")
        self.assertEqual(normalize_promo_code("ใช้ Live 25", self.catalog), "LIVE25")
        self.assertEqual(normalize_promo_code("โค้ดไลฟ์ 25", self.catalog), "LIVE25")

    def test_bundle_is_catalog_compatible(self) -> None:
        bundle = next(action for action in self.actions if action.action_type == SHOW_BUNDLE_RECOMMENDATION)
        self.assertEqual(bundle.skus, ["SKU002", "SKU001"])
        products = bundle.display_payload["products"]
        self.assertEqual([product["sku"] for product in products], ["SKU002", "SKU001"])

    def test_flash_sale_countdown_payload(self) -> None:
        flash = next(action for action in self.actions if action.action_type == START_FLASH_SALE_COUNTDOWN)
        self.assertEqual(flash.display_payload["duration_minutes"], 5)
        self.assertEqual(flash.display_payload["duration_seconds"], 300)
        self.assertEqual(flash.display_payload["promo_code"], "LIVE25")

    def test_cli_smoke_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.pipeline.run_commerce_actions",
                    "--captions",
                    str(CAPTIONS),
                    "--catalog",
                    str(CATALOG),
                    "--audio",
                    str(AUDIO),
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )
            summary = json.loads(completed.stdout.decode("utf-8"))
            self.assertEqual(summary["action_count"], 8)
            self.assertTrue((Path(temp_dir) / "commerce_actions.json").exists())
            self.assertTrue((Path(temp_dir) / "commerce_actions_timeline.html").exists())

    def test_action_timeline_html_embeds_actions(self) -> None:
        html = build_action_timeline_html(self.actions, audio_path=AUDIO)
        self.assertIn("<audio controls", html)
        self.assertIn("SHOW_BUNDLE_RECOMMENDATION", html)
        self.assertIn("data:audio/mpeg;base64,", html)


if __name__ == "__main__":
    unittest.main()
