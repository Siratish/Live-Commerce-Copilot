from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional
import json
import sys

from src.ai.commerce_actions import generate_commerce_actions, load_caption_result, save_commerce_actions
from src.data.catalog import load_product_catalog
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
    output_dir: Path,
    audio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    captions = load_caption_result(captions_path)
    catalog = load_product_catalog(catalog_path)
    actions = generate_commerce_actions(captions, catalog)

    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / "commerce_actions.json"
    html_path = output_dir / "commerce_actions_timeline.html"
    save_commerce_actions(actions, actions_path)
    save_action_timeline_html(actions, html_path, audio_path=audio_path)

    return {
        "action_count": len(actions),
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
        "--output-dir",
        default=str(REPO_ROOT / "outputs"),
        help="Directory for generated action outputs.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Optional audio path for synchronized timeline HTML.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_actions_from_paths(
        captions_path=resolve_repo_path(args.captions) or Path(args.captions),
        catalog_path=resolve_repo_path(args.catalog) or Path(args.catalog),
        output_dir=resolve_repo_path(args.output_dir) or Path(args.output_dir),
        audio_path=resolve_repo_path(args.audio) if args.audio else None,
    )
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(payload.encode("utf-8"))


if __name__ == "__main__":
    main()
