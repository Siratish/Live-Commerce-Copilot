from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional
import json
import sys

from src.ai.commerce_actions import generate_commerce_actions, load_caption_result, save_commerce_actions
from src.ai.decision import (
    DEFAULT_TYPHOON_MODEL_ID,
    CommerceDecisionProvider,
    DeterministicDecisionProvider,
    Typhoon25DecisionProvider,
)
from src.data.catalog import load_product_catalog, load_promotions
from src.utils.action_timeline import save_action_timeline_html


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_actions_from_paths(
    captions_path: Path,
    catalog_path: Path,
    promotions_path: Path,
    output_dir: Path,
    audio_path: Optional[Path] = None,
    decision_provider: Optional[CommerceDecisionProvider] = None,
    decision_provider_name: str = "deterministic",
    decision_model: Optional[str] = None,
    decision_max_new_tokens: int = 256,
    decision_temperature: float = 0.1,
) -> Dict[str, Any]:
    captions = load_caption_result(captions_path)
    catalog = load_product_catalog(catalog_path)
    promotions = load_promotions(promotions_path)
    decision_provider = decision_provider or create_decision_provider(
        provider_name=decision_provider_name,
        model_id=decision_model,
        max_new_tokens=decision_max_new_tokens,
        temperature=decision_temperature,
    )
    actions = generate_commerce_actions(
        captions,
        catalog,
        promotions,
        decision_provider=decision_provider,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / "commerce_actions.json"
    html_path = output_dir / "commerce_actions_timeline.html"
    save_commerce_actions(actions, actions_path)
    save_action_timeline_html(actions, html_path, audio_path=audio_path)

    return {
        "action_count": len(actions),
        "decision_provider": decision_provider_name,
        "decision_model": decision_model,
        "actions": [action.to_dict() for action in actions],
        "outputs": {
            "json": str(actions_path),
            "html": str(html_path),
        },
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate live-commerce actions from captions.")
    parser.add_argument("--captions", required=True, help="Path to captions JSON.")
    parser.add_argument("--catalog", required=True, help="Path to product catalog CSV.")
    parser.add_argument(
        "--promotions",
        default=str(REPO_ROOT / "data" / "demo" / "promotions.csv"),
        help="Path to promotion rules CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs"),
        help="Directory for generated action outputs.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Optional audio path for synchronized timeline HTML.",
    )
    parser.add_argument(
        "--decision-provider",
        choices=["deterministic", "typhoon25", "typhoon_s"],
        default="deterministic",
        help="Decision provider for commerce action extraction.",
    )
    parser.add_argument(
        "--decision-model",
        default=DEFAULT_TYPHOON_MODEL_ID,
        help="Model id for AI decision providers.",
    )
    parser.add_argument(
        "--decision-max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for AI decision JSON generation.",
    )
    parser.add_argument(
        "--decision-temperature",
        type=float,
        default=0.1,
        help="Generation temperature for AI decision JSON generation.",
    )
    return parser


def build_decision_provider(args) -> Optional[CommerceDecisionProvider]:
    return create_decision_provider(
        provider_name=args.decision_provider,
        model_id=args.decision_model,
        max_new_tokens=args.decision_max_new_tokens,
        temperature=args.decision_temperature,
    )


def create_decision_provider(
    provider_name: str = "deterministic",
    model_id: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> CommerceDecisionProvider:
    if provider_name == "deterministic":
        return DeterministicDecisionProvider()
    if provider_name in {"typhoon25", "typhoon_s"}:
        return Typhoon25DecisionProvider(
            model_id=model_id or DEFAULT_TYPHOON_MODEL_ID,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    raise ValueError(f"unsupported decision provider: {provider_name}")


def main() -> None:
    args = build_parser().parse_args()
    summary = run_actions_from_paths(
        captions_path=resolve_repo_path(args.captions) or Path(args.captions),
        catalog_path=resolve_repo_path(args.catalog) or Path(args.catalog),
        promotions_path=resolve_repo_path(args.promotions) or Path(args.promotions),
        output_dir=resolve_repo_path(args.output_dir) or Path(args.output_dir),
        audio_path=resolve_repo_path(args.audio) if args.audio else None,
        decision_provider=build_decision_provider(args),
        decision_provider_name=args.decision_provider,
        decision_model=args.decision_model if args.decision_provider in {"typhoon25", "typhoon_s"} else None,
        decision_max_new_tokens=args.decision_max_new_tokens,
        decision_temperature=args.decision_temperature,
    )
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(payload.encode("utf-8"))


if __name__ == "__main__":
    main()
