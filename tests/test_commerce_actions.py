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
from src.ai.retrieval import ProductPromoRetriever, extract_numbers
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
                (PIN_PRODUCT_CARD, ["SKU001"], 7.86),
                (SHOW_PROMO_CODE, ["SKU001", "SKU002", "SKU007"], 29.34),
                (PIN_PRODUCT_CARD, ["SKU002"], 49.04),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU002", "SKU001"], 62.94),
                (START_FLASH_SALE_COUNTDOWN, ["SKU002", "SKU001"], 72.74),
            ],
        )

    def test_product_cards_include_catalog_prices(self) -> None:
        product_cards = [action for action in self.actions if action.action_type == PIN_PRODUCT_CARD]
        first_product = product_cards[0].display_payload["product"]
        second_product = product_cards[1].display_payload["product"]
        self.assertEqual(first_product["price"], 399)
        self.assertEqual(first_product["discount_price"], 299)
        self.assertEqual(second_product["price"], 259)
        self.assertEqual(second_product["discount_price"], 199)

    def test_promo_normalization_variants(self) -> None:
        self.assertEqual(normalize_promo_code("Code Life 25", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("Code Life 2-5", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("ใช้ Live 25", self.promotions), "LIVE25")
        self.assertEqual(normalize_promo_code("โค้ดไลฟ์ 25", self.promotions), "LIVE25")
        self.assertIsNone(normalize_promo_code("True ID Tech Live Deal", self.promotions))

    def test_number_extraction_handles_comma_and_split_digits(self) -> None:
        self.assertIn(1290, extract_numbers("ราคาเต็ม 1,290 บาท"))
        self.assertIn(25, extract_numbers("Code Life 2-5"))
        self.assertIn(100, extract_numbers("โค้ดเทก 1 00 ลดเพิ่มอีก 100 บาท"))

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
            self.assertEqual(summary["action_count"], 5)
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
                (SHOW_PROMO_CODE, ["SKU001", "SKU002", "SKU007"], 29.34),
                (PIN_PRODUCT_CARD, ["SKU002"], 49.04),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU002", "SKU001"], 62.94),
                (START_FLASH_SALE_COUNTDOWN, ["SKU002", "SKU001"], 72.74),
            ],
        )

    def test_split_bundle_cue_across_realtime_chunks_generates_bundle(self) -> None:
        captions = CaptionResult(
            language="th",
            duration_seconds=77.3,
            segments=[
                CaptionSegment(
                    start=8.3,
                    end=13.7,
                    text="ตัวแรกคือวิตามินซีซีรัมจาก Demo Beauty ปกติราคา 399 บาท",
                    source="test",
                ),
                CaptionSegment(
                    start=49.2,
                    end=53.44,
                    text="ตอนนี้เป็น Green Tea Cleanser จาก Demo Beauty ค่ะ",
                    source="test",
                ),
                CaptionSegment(
                    start=53.8,
                    end=56.46,
                    text="ปกติราคา 259 บาท",
                    source="test",
                ),
                CaptionSegment(
                    start=61.26,
                    end=62.8,
                    text="เหมาะสำหรับใช้ทุกวันค่ะ",
                    source="test",
                ),
                CaptionSegment(
                    start=63.1,
                    end=65.74,
                    text="ถ้าใช้คู่กันแนะนำให้ใช้ Queen Tea Crancer ก่อน",
                    source="test",
                ),
                CaptionSegment(
                    start=65.74,
                    end=68.22,
                    text="แล้วตามด้วยวิตามิน C Serum หลัง ๆ หน้านะคะ",
                    source="test",
                ),
                CaptionSegment(
                    start=70.5,
                    end=72.58,
                    text="ตอนนี้โปรไลท์ 2 หาเหลือเวลาอิ",
                    source="test",
                ),
                CaptionSegment(
                    start=72.9,
                    end=77.3,
                    text="5 นาที เท่านั้นนะคะ ใครสนใจสามารถกดซื้อผ่านลิงค์ใน True ID Like ได้เลยค่ะ",
                    source="test",
                ),
            ],
        )
        actions = generate_commerce_actions(captions, self.catalog, self.promotions)
        sequence = [(action.action_type, action.skus, round(action.timestamp, 2)) for action in actions]
        self.assertEqual(
            sequence,
            [
                (PIN_PRODUCT_CARD, ["SKU001"], 8.3),
                (PIN_PRODUCT_CARD, ["SKU002"], 49.2),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU002", "SKU001"], 63.1),
                (START_FLASH_SALE_COUNTDOWN, ["SKU002", "SKU001"], 72.9),
            ],
        )

    def test_audio2_tech_transcript_generates_expected_actions(self) -> None:
        captions = CaptionResult(
            language="th",
            duration_seconds=80.48,
            segments=[
                CaptionSegment(
                    start=0.0,
                    end=6.16,
                    text="สวัสดีครับทุกคน วันนี้เราอยู่กับ True ID Tech Live Deal สำหรับคนที่ใช้มือถือทุกวันนะครับ",
                    source="test",
                ),
                CaptionSegment(
                    start=6.16,
                    end=14.64,
                    text="สินค้าตัวแรกคือ Fast Charge Power Bank 1 000-0-0-0-0-MH จาก Demotech ปกติราคา 799 บาท",
                    source="test",
                ),
                CaptionSegment(
                    start=14.64,
                    end=21.92,
                    text="วันนี้ลดเหลือ 599 บาท ตัวนี้รองรับ Fast Charging และมีพอร์ต USB-C เหมาะสำหรับคนที่เดินทางบ่อย",
                    source="test",
                ),
                CaptionSegment(
                    start=21.92,
                    end=29.84,
                    text="หรือดูวิดีโอบนมือถือเป็นเวลานานครับ ใช้โคตรเทกส 1 00 ลดเพิ่มอีก 100 บาท เมื่อซื้อผ่าน Live นี้นะครับ",
                    source="test",
                ),
                CaptionSegment(
                    start=36.88,
                    end=49.32,
                    text="ในกล่องมีเฉพาะ Power Bank และคู่มือการใช้งานครับ ต่อไปเป็น Wireless Earbuds Lite จาก Demo Sound ราคาเต็ม 1,290 บาท วันนี้ลดเหลือ 990 บาท",
                    source="test",
                ),
                CaptionSegment(
                    start=71.36,
                    end=80.48,
                    text="สำหรับ BUNDLE ผมแนะนำ Power Bank คู่กับ Wireless Earbuds เพราะเหมาะกับคนเดินทาง ดูหนัง ฟังเพลง และไม่อยากให้แบดหมดระหว่างวันครับ",
                    source="test",
                ),
            ],
        )
        actions = generate_commerce_actions(captions, self.catalog, self.promotions)
        sequence = [(action.action_type, action.skus, round(action.timestamp, 2)) for action in actions]
        self.assertEqual(
            sequence,
            [
                (PIN_PRODUCT_CARD, ["SKU005"], 6.16),
                (SHOW_PROMO_CODE, ["SKU005", "SKU006", "SKU008"], 21.92),
                (PIN_PRODUCT_CARD, ["SKU006"], 36.88),
                (SHOW_BUNDLE_RECOMMENDATION, ["SKU005", "SKU006"], 71.36),
            ],
        )
        promo = next(action for action in actions if action.action_type == SHOW_PROMO_CODE)
        self.assertEqual(promo.display_payload["promo_code"], "TECH100")
        earbuds_card = next(action for action in actions if action.skus == ["SKU006"])
        self.assertEqual(earbuds_card.display_payload["product"]["price"], 1290)
        self.assertEqual(earbuds_card.display_payload["product"]["discount_price"], 990)

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
        self.assertIn("history_text", captured["window"])
        self.assertEqual(len(captured["catalog"]), len(self.catalog))
        self.assertEqual(len(captured["promotions"]), len(self.promotions))
        self.assertIn("discount_price", captured["catalog"][0])
        self.assertNotIn("deeplink", captured["catalog"][0])
        self.assertIn("eligible_categories", captured["promotions"][0])

    def test_action_windows_include_unresolved_history_segments(self) -> None:
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

    def test_action_history_resets_after_emitted_action(self) -> None:
        seen_previous = []

        class ResetAfterActionProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                seen_previous.append(tuple(window.previous_texts))
                if window.index == 0:
                    return CommerceDecision(
                        action_type=PIN_PRODUCT_CARD,
                        product_sku="SKU001",
                        product_confidence=1.0,
                    )
                return CommerceDecision()

        captions = CaptionResult(
            language="th",
            duration_seconds=2.0,
            segments=[
                CaptionSegment(start=0.0, end=1.0, text="Vitamin C Serum", source="test"),
                CaptionSegment(start=1.0, end=2.0, text="next unresolved segment", source="test"),
            ],
        )
        actions = generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=ResetAfterActionProvider(),
        )

        self.assertEqual([action.action_type for action in actions], [PIN_PRODUCT_CARD])
        self.assertEqual(seen_previous, [(), ()])

    def test_action_history_trims_oldest_segments_when_too_long(self) -> None:
        seen_history = []

        class CaptureHistoryProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                seen_history.append(window.history_text)
                return CommerceDecision()

        captions = CaptionResult(
            language="th",
            duration_seconds=3.0,
            segments=[
                CaptionSegment(start=0.0, end=1.0, text="aaaa", source="test"),
                CaptionSegment(start=1.0, end=2.0, text="bbbb", source="test"),
                CaptionSegment(start=2.0, end=3.0, text="cccc", source="test"),
            ],
        )
        generate_commerce_actions(
            captions,
            self.catalog,
            self.promotions,
            decision_provider=CaptureHistoryProvider(),
            max_decision_history_chars=9,
        )

        self.assertEqual(seen_history, ["aaaa", "aaaa bbbb", "bbbb cccc"])

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

    def test_model_unsupported_action_type_is_ignored(self) -> None:
        class UnsupportedActionProvider:
            def decide(self, window, candidates, state, catalog, promotions):
                return CommerceDecision(
                    action_type="UNSUPPORTED_PRICE_ACTION",
                    confidence=0.88,
                    product_sku="SKU001",
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
            decision_provider=UnsupportedActionProvider(),
        )

        self.assertEqual(actions, [])

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
