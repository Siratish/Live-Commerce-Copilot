from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
import unittest

import src.ai.commerce_actions as commerce_actions
from src.ai.commerce_actions import (
    PIN_PRODUCT_CARD,
    SHOW_BUNDLE_RECOMMENDATION,
    SHOW_PRICE_DROP,
    SHOW_PROMO_CODE,
    START_FLASH_SALE_COUNTDOWN,
    eligible_items_for_promotion,
    generate_commerce_actions,
    load_caption_result,
    normalize_promo_code,
)
from src.ai.decision import (
    CallableModelDecisionProvider,
    CommerceCandidates,
    CommerceDecision,
    CommerceSessionState,
    DEFAULT_TYPHOON_MODEL_ID,
    TranscriptWindow,
    Typhoon25DecisionProvider,
    TyphoonSDecisionProvider,
    extract_json_mapping,
)
from src.ai.retrieval import ProductPromoRetriever
from src.data.catalog import load_product_catalog, load_promotions
from src.pipeline.run_commerce_actions import create_decision_provider
from src.schemas import CaptionResult, CaptionSegment, ProductCatalogItem
from src.utils.action_timeline import build_action_timeline_html


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = REPO_ROOT / "data" / "demo" / "audio_1_captions.json"
TURBO_CAPTIONS = REPO_ROOT / "data" / "demo" / "audio_1_captions_turbo.json"
CATALOG = REPO_ROOT / "data" / "demo" / "product_catalog.csv"
PROMOTIONS = REPO_ROOT / "data" / "demo" / "promotions.csv"
AUDIO = REPO_ROOT / "data" / "demo" / "audio" / "1.mp3"


class CommerceActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.captions = load_caption_result(CAPTIONS)
        self.catalog = load_product_catalog(CATALOG)
        self.promotions = load_promotions(PROMOTIONS)
        self.actions = generate_commerce_actions(self.captions, self.catalog, self.promotions)

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
            ],
        )

    def test_price_extraction_payloads(self) -> None:
        price_actions = [action for action in self.actions if action.action_type == SHOW_PRICE_DROP]
        self.assertEqual(price_actions[0].display_payload["original_price"], 399)
        self.assertEqual(price_actions[0].display_payload["discount_price"], 299)
        self.assertEqual(price_actions[1].display_payload["original_price"], 259)
        self.assertEqual(price_actions[1].display_payload["discount_price"], 199)

    def test_promo_normalization_variants(self) -> None:
        self.assertEqual(normalize_promo_code("Code Life 25", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("Code Life 2-5", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("ใช้ Live 25", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("โค้ดไลฟ์ 25", self.promotions), "LIVE25")

    def test_promotion_eligibility_is_separate_from_product_catalog(self) -> None:
        live25 = next(promotion for promotion in self.promotions if promotion.promo_code == "LIVE25")
        eligible = eligible_items_for_promotion(live25, self.catalog)
        self.assertEqual([item.sku for item in eligible], ["SKU001", "SKU002", "SKU007"])
        self.assertNotIn("SKU003", [item.sku for item in eligible])

    def test_alias_tables_are_not_used(self) -> None:
        self.assertFalse(hasattr(commerce_actions, "PRODUCT_ALIASES"))
        self.assertFalse(hasattr(commerce_actions, "PROMO_VARIANTS"))

    def test_retrieval_resolves_asr_noise_without_alias_tables(self) -> None:
        retriever = ProductPromoRetriever.build(self.catalog, self.promotions)
        product = retriever.retrieve_products("ตอนนี้เป็น Green Tea Cleanseur จาก Demo Beauty", top_k=1)[0]
        self.assertEqual(product.item.sku, "SKU002")
        self.assertGreaterEqual(product.score, 0.48)
        self.assertEqual(normalize_promo_code("Code Life 2-5", self.promotions), "LIVE25")

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
                    "--promotions",
                    str(PROMOTIONS),
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
            self.assertEqual(summary["action_count"], 7)
            self.assertTrue((Path(temp_dir) / "commerce_actions.json").exists())
            self.assertTrue((Path(temp_dir) / "commerce_actions_timeline.html").exists())

    def test_action_timeline_html_embeds_actions(self) -> None:
        html = build_action_timeline_html(self.actions, audio_path=AUDIO)
        self.assertIn("<audio controls", html)
        self.assertIn("SHOW_BUNDLE_RECOMMENDATION", html)
        self.assertIn("data:audio/mpeg;base64,", html)

    def test_turbo_transcript_variants_still_generate_expected_actions(self) -> None:
        captions = load_caption_result(TURBO_CAPTIONS)
        actions = generate_commerce_actions(captions, self.catalog, self.promotions)
        sequence = [(action.action_type, action.skus, round(action.timestamp, 2)) for action in actions]
        self.assertEqual(
            sequence,
            [
                (PIN_PRODUCT_CARD, ["SKU001"], 7.86),
                (SHOW_PRICE_DROP, ["SKU001"], 13.58),
                (SHOW_PROMO_CODE, ["SKU001", "SKU002", "SKU007"], 29.34),
                (PIN_PRODUCT_CARD, ["SKU002"], 49.04),
                (SHOW_PRICE_DROP, ["SKU002"], 53.64),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU002", "SKU001"], 62.94),
                (START_FLASH_SALE_COUNTDOWN, ["SKU002", "SKU001"], 70.44),
            ],
        )

    def test_invalid_model_decisions_are_rejected_by_validation(self) -> None:
        class BadDecisionProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                return CommerceDecision(
                    product_sku="UNKNOWN_SKU",
                    promo_code="UNKNOWN_PROMO",
                    bundle_skus=["SKU001", "UNKNOWN_SKU"],
                    flash_minutes=None,
                )

        captions = CaptionResult(
            language="th",
            duration_seconds=2.0,
            segments=[
                CaptionSegment(
                    start=0.0,
                    end=2.0,
                    text="Green Tea Cleanseur 199 บาท",
                    source="test",
                )
            ],
        )
        actions = generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=BadDecisionProvider(),
        )
        self.assertEqual(actions, [])

    def test_callable_model_provider_accepts_structured_output(self) -> None:
        provider = CallableModelDecisionProvider(
            lambda payload: {
                "action_type": "SHOW_BUNDLE_RECOMMENDATION",
                "confidence": 0.86,
                "product_sku": "SKU002",
                "product_confidence": 0.91,
                "promo_code": "LIVE25",
                "promo_confidence": 0.88,
                "bundle_skus": ["SKU002", "SKU001"],
                "bundle_confidence": 0.86,
                "flash_minutes": 5,
                "flash_confidence": 0.84,
            }
        )
        decision = provider.decide(
            TranscriptWindow(index=0, start=0.0, text="test"),
            CommerceCandidates(products=[], bundle_products=[], promotions=[]),
            CommerceSessionState(),
            self.catalog,
            self.promotions,
        )
        self.assertEqual(decision.product_sku, "SKU002")
        self.assertEqual(decision.promo_code, "LIVE25")
        self.assertEqual(decision.bundle_skus, ["SKU002", "SKU001"])
        self.assertEqual(decision.flash_minutes, 5)
        self.assertEqual(decision.action_type, SHOW_BUNDLE_RECOMMENDATION)

    def test_model_payload_includes_catalog_promotions_and_previous_texts(self) -> None:
        captured = {}

        def capture_payload(payload):
            captured.update(payload)
            return {"action_type": "NO_ACTION", "confidence": 0}

        provider = CallableModelDecisionProvider(capture_payload)
        decision = provider.decide(
            TranscriptWindow(
                index=3,
                start=9.0,
                text="วันนี้ใช้คู่กัน",
                next_text="ลดเหลือ 199 บาท",
                previous_texts=("ตัวแรกเป็นเซรั่ม", "ราคา 399", "ใช้โค้ด LIVE25"),
            ),
            CommerceCandidates(products=[], bundle_products=[], promotions=[]),
            CommerceSessionState(active_sku="SKU001", active_promo_code="LIVE25"),
            self.catalog,
            self.promotions,
        )

        self.assertIsNone(decision.action_type)
        self.assertEqual(captured["window"]["previous_texts"], ["ตัวแรกเป็นเซรั่ม", "ราคา 399", "ใช้โค้ด LIVE25"])
        self.assertEqual(len(captured["catalog"]), len(self.catalog))
        self.assertEqual(len(captured["promotions"]), len(self.promotions))
        self.assertIn("discount_price", captured["catalog"][0])
        self.assertNotIn("deeplink", captured["catalog"][0])
        self.assertIn("eligible_categories", captured["promotions"][0])

    def test_action_windows_include_previous_three_caption_segments(self) -> None:
        seen_previous = []

        class CaptureWindowProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                seen_previous.append(tuple(window.previous_texts))
                return CommerceDecision()

        captions = CaptionResult(
            language="th",
            duration_seconds=4.0,
            segments=[
                CaptionSegment(start=0.0, end=1.0, text="หนึ่ง", source="test"),
                CaptionSegment(start=1.0, end=2.0, text="สอง", source="test"),
                CaptionSegment(start=2.0, end=3.0, text="สาม", source="test"),
                CaptionSegment(start=3.0, end=4.0, text="สี่", source="test"),
            ],
        )
        generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=CaptureWindowProvider(),
        )

        self.assertEqual(
            seen_previous,
            [
                (),
                ("หนึ่ง",),
                ("หนึ่ง", "สอง"),
                ("หนึ่ง", "สอง", "สาม"),
            ],
        )

    def test_model_action_type_limits_output_to_one_action_per_segment(self) -> None:
        class SingleActionProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                return CommerceDecision(
                    action_type=PIN_PRODUCT_CARD,
                    confidence=0.9,
                    product_sku="SKU001",
                    product_confidence=0.9,
                    promo_code="LIVE25",
                    promo_confidence=0.9,
                    original_price=399,
                    discount_price=299,
                )

        captions = CaptionResult(
            language="th",
            duration_seconds=2.0,
            segments=[
                CaptionSegment(
                    start=0.0,
                    end=2.0,
                    text="Vitamin C Serum ราคา 399 เหลือ 299 ใช้โค้ด LIVE25",
                    source="test",
                )
            ],
        )
        actions = generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=SingleActionProvider(),
        )

        self.assertEqual([action.action_type for action in actions], [PIN_PRODUCT_CARD])

    def test_model_price_drop_action_can_use_validated_price_details(self) -> None:
        class PriceActionProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                return CommerceDecision(
                    action_type=SHOW_PRICE_DROP,
                    confidence=0.88,
                    product_sku="SKU001",
                    original_price=399,
                    discount_price=299,
                )

        captions = CaptionResult(
            language="th",
            duration_seconds=2.0,
            segments=[
                CaptionSegment(
                    start=0.0,
                    end=2.0,
                    text="ตัวนี้ลดราคาเฉพาะในไลฟ์",
                    source="test",
                )
            ],
        )
        actions = generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=PriceActionProvider(),
        )

        self.assertEqual([action.action_type for action in actions], [SHOW_PRICE_DROP])
        self.assertEqual(actions[0].display_payload["original_price"], 399)
        self.assertEqual(actions[0].display_payload["discount_price"], 299)

    def test_typhoon_s_provider_accepts_json_output(self) -> None:
        def fake_generator(messages):
            return """
            ```json
            {
              "product_sku": "SKU002",
              "product_confidence": 0.92,
              "promo_code": "LIVE25",
              "promo_confidence": 0.81,
              "bundle_skus": ["SKU002", "SKU001"],
              "bundle_confidence": 0.88,
              "flash_minutes": 5,
              "flash_confidence": 0.77
            }
            ```
            """

        provider = Typhoon25DecisionProvider(text_generator=fake_generator)
        decision = provider.decide(
            TranscriptWindow(index=0, start=0.0, text="test"),
            CommerceCandidates(products=[], bundle_products=[], promotions=[]),
            CommerceSessionState(),
            self.catalog,
            self.promotions,
        )
        self.assertEqual(decision.product_sku, "SKU002")
        self.assertEqual(decision.promo_code, "LIVE25")
        self.assertEqual(decision.bundle_skus, ["SKU002", "SKU001"])
        self.assertEqual(decision.flash_minutes, 5)

    def test_typhoon_s_provider_falls_back_on_invalid_json(self) -> None:
        provider = Typhoon25DecisionProvider(text_generator=lambda messages: "not json")
        window = TranscriptWindow(index=0, start=0.0, text="Green Tea Cleanser")
        candidates = CommerceCandidates(
            products=ProductPromoRetriever.build(self.catalog, self.promotions).retrieve_products(
                "Green Tea Cleanser",
                top_k=5,
            ),
            bundle_products=[],
            promotions=[],
        )
        decision = provider.decide(
            window,
            candidates,
            CommerceSessionState(),
            self.catalog,
            self.promotions,
        )
        self.assertEqual(decision.product_sku, "SKU002")

    def test_typhoon_s_json_extraction_accepts_wrapped_text(self) -> None:
        parsed = extract_json_mapping('extra {"product_sku": "SKU001"} text')
        self.assertEqual(parsed["product_sku"], "SKU001")

    def test_typhoon_s_provider_unload_is_safe_without_loaded_model(self) -> None:
        provider = Typhoon25DecisionProvider(text_generator=lambda messages: "{}")
        provider.unload_model()
        self.assertIsNone(provider._model)
        self.assertIsNone(provider._tokenizer)

    def test_create_decision_provider_supports_typhoon_s(self) -> None:
        provider = create_decision_provider("typhoon_s")
        self.assertIsInstance(provider, Typhoon25DecisionProvider)
        self.assertEqual(provider.model_id, DEFAULT_TYPHOON_MODEL_ID)

    def test_create_decision_provider_supports_typhoon25_alias(self) -> None:
        provider = create_decision_provider("typhoon25")
        self.assertIsInstance(provider, Typhoon25DecisionProvider)
        self.assertEqual(provider.model_id, "scb10x/typhoon2.5-qwen3-4b")

    def test_typhoon_s_provider_name_is_legacy_alias(self) -> None:
        self.assertIs(TyphoonSDecisionProvider, Typhoon25DecisionProvider)

    def test_retrieval_scales_to_10k_skus(self) -> None:
        synthetic_catalog = _synthetic_catalog(10_000)
        start = time.perf_counter()
        retriever = ProductPromoRetriever.build(synthetic_catalog, self.promotions)
        results = retriever.retrieve_products("Green Tea Cleanseur fast gentle daily cleanser", top_k=3)
        elapsed = time.perf_counter() - start
        self.assertEqual(results[0].item.sku, "SKU002")
        self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()


def _synthetic_catalog(size: int) -> list[ProductCatalogItem]:
    base = [
        ProductCatalogItem(
            sku=f"SYN{i:05d}",
            product_name=f"Synthetic Product {i}",
            brand="SyntheticBrand",
            category="Synthetic",
            price=100 + (i % 900),
            discount_price=80 + (i % 700),
            stock=50,
            description=f"Synthetic filler catalog item {i}",
            tags=["synthetic", f"group_{i % 20}"],
            compatible_with=[],
            deeplink=f"https://trueid.example/live/product/SYN{i:05d}",
        )
        for i in range(size - 1)
    ]
    base.append(
        ProductCatalogItem(
            sku="SKU002",
            product_name="Green Tea Cleanser",
            brand="DemoBeauty",
            category="Skincare",
            price=259,
            discount_price=199,
            stock=80,
            description="Gentle daily cleanser with green tea extract",
            tags=["cleanser", "gentle", "green_tea", "skincare"],
            compatible_with=["SKU001"],
            deeplink="https://trueid.example/live/product/SKU002",
        )
    )
    return base
