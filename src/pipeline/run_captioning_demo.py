from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional
import json

from src.ai.captioning import (
    CaptioningEngine,
    CaptioningSettings,
    available_asr_models,
    caption_metrics,
    write_caption_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _fallback_yaml_load(path: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key = line.rstrip(":")
            config[key] = {}
            current_section = key
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        config[current_section][key.strip()] = _coerce_scalar(value)
    return config


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _fallback_yaml_load(path)

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_from_config(
    config_path: Path,
    mode_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    asr_provider_override: Optional[str] = None,
    asr_model_override: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    captioning = config.get("captioning", {})
    outputs = config.get("outputs", {})

    settings = CaptioningSettings(
        mode=mode_override or str(captioning.get("mode", "cached")),
        language=str(captioning.get("language", "en")),
        cached_transcript_path=resolve_repo_path(
            captioning.get("cached_transcript_path")
        ),
        audio_path=resolve_repo_path(captioning.get("audio_path")),
        asr_provider=asr_provider_override
        or str(captioning.get("asr_provider", "openai_whisper")),
        asr_model=asr_model_override
        or (
            None
            if captioning.get("asr_model") is None
            else str(captioning.get("asr_model"))
        ),
        whisper_model=str(captioning.get("whisper_model", "tiny")),
        asr_chunk_length_seconds=int(captioning.get("asr_chunk_length_seconds", 30)),
        asr_batch_size=int(captioning.get("asr_batch_size", 16)),
        allow_cached_fallback=bool(captioning.get("allow_cached_fallback", True)),
    )
    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else resolve_repo_path(outputs.get("directory")) or REPO_ROOT / "outputs"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    result = CaptioningEngine(settings).transcribe()
    paths = write_caption_outputs(
        result=result,
        output_dir=output_dir,
        json_name=str(outputs.get("captions_json", "captions.json")),
        vtt_name=str(outputs.get("captions_vtt", "captions.vtt")),
        srt_name=str(outputs.get("captions_srt", "captions.srt")),
        metrics_name=str(outputs.get("metrics_json", "caption_metrics.json")),
    )
    metrics = caption_metrics(result)
    return {
        "language": result.language,
        "segment_count": len(result.segments),
        "mode": settings.mode,
        "asr_provider": settings.asr_provider,
        "asr_model": settings.asr_model or settings.whisper_model,
        "audio_path": str(settings.audio_path) if settings.audio_path else None,
        "metrics": metrics,
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the captioning demo pipeline.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "demo.yaml"),
        help="Path to the YAML demo config.",
    )
    parser.add_argument(
        "--mode",
        choices=["cached", "auto", "whisper", "openai_whisper", "typhoon", "typhoon_whisper"],
        default=None,
        help="Override captioning mode from config.",
    )
    parser.add_argument(
        "--asr-provider",
        choices=["openai_whisper", "typhoon_whisper"],
        default=None,
        help="ASR provider for auto mode.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=(
            "ASR model alias or full model id. Typhoon aliases include "
            "large-v3, turbo, medium, and isan-medium."
        ),
    )
    parser.add_argument(
        "--list-asr-models",
        action="store_true",
        help="Print supported ASR model aliases and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_asr_models:
        print(json.dumps(available_asr_models(), indent=2))
        return
    summary = run_from_config(
        config_path=Path(args.config),
        mode_override=args.mode,
        output_dir_override=args.output_dir,
        asr_provider_override=args.asr_provider,
        asr_model_override=args.asr_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
