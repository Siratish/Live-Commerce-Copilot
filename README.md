# Live Commerce Copilot

Colab-safe AI live-commerce copilot demo for turning live-stream speech into
captions and synchronized commerce actions.

This implementation focuses on timestamped captions, transcript-driven product
and promotion actions, bundle recommendations, and flash-sale countdowns. It
intentionally does not include product visual grounding, viewer Q&A, or chat
analytics yet.

## What Runs Today

- Loads a deterministic cached transcript from `data/demo/captions/audio_1_captions.json`.
- Validates timestamp order and caption text.
- Exports captions as JSON, WebVTT, and SRT.
- Writes a small metrics summary for notebook display.
- Provides an optional Whisper path for future real audio inference.
- Generates transcript-driven live-commerce actions from the mock product catalog.

The default config uses `auto` mode against `data/demo/audio/1.mp3` with Thai
captioning (`language: th`): it tries the configured ASR provider first, then
falls back to the cached Thai transcript if `ffmpeg` or model dependencies are
unavailable. The fallback keeps the demo runnable without GPU, model downloads,
or API keys.

## Quick Start

```bash
python -m src.pipeline.run_captioning_demo --config config/demo.yaml
```

Expected outputs:

- `outputs/captions.json`
- `outputs/captions.vtt`
- `outputs/captions.srt`
- `outputs/caption_metrics.json`

Generate commerce actions from the audio-1 transcript:

```bash
python -m src.pipeline.run_commerce_actions --captions outputs/captions.json --catalog data/demo/catalog/product_catalog.csv --promotions data/demo/catalog/promotions.csv --audio data/demo/audio/1.mp3
```

Expected action outputs:

- `outputs/commerce_actions.json`
- `outputs/commerce_actions_timeline.html`

Products and promotions are modeled separately. `data/demo/catalog/product_catalog.csv`
contains product facts, prices, tags, compatibility, and deeplinks.
`data/demo/catalog/promotions.csv` defines promo codes, discount type/value, live-only
status, eligible categories/tags/SKUs, and stock requirements.

## ASR Model Choices

The captioning config supports one ASR provider for this demo:

- `openai_whisper`: local `openai-whisper` package aliases such as `tiny`,
  `base`, `small`, `medium`, `large`, `large-v3`, and `turbo`.

List the aliases:

```bash
python -m src.pipeline.run_captioning_demo --list-asr-models
```

Run OpenAI Whisper Turbo while preserving cached fallback:

```bash
python -m src.pipeline.run_captioning_demo --config config/demo.yaml --asr-provider openai_whisper --asr-model turbo
```

For live file and mic demos, audio is chunked with the pause-aware windowing
utilities before each chunk is sent to ASR. The window closes when the speaker
pauses, or when the max window duration is reached for long continuous speech.
This mirrors the production streaming path without relying on a fixed-length
split for every utterance.

Tune the max utterance window with:

```bash
python -m src.pipeline.run_captioning_demo --config config/demo.yaml --asr-provider openai_whisper --asr-model turbo --asr-chunk-length-seconds 4
```

The commerce action engine is retrieval plus decision:

- `src/ai/retrieval.py` builds searchable product and promotion records from
  catalog facts and uses character n-gram scoring for ASR-noisy Thai/English
  transcript text.
- `src/ai/decision.py` defines the deterministic decision provider used by the
  demo. It uses retrieved candidates, recent transcript history, active session
  state, bundle cues, flash-sale cues, and catalog compatibility rules to choose
  at most one action per caption segment.
- `src/ai/commerce_actions.py` validates every decision against catalog and
  promotion rules before emitting UI actions, so unknown SKUs, unsupported promo
  codes, and incompatible bundles are rejected.

## Tests

```bash
python -m unittest discover -s tests
```

If `pytest` is installed, this also works:

```bash
pytest
```

## Optional Real ASR

The default demo does not require speech model dependencies. To experiment with
real audio transcription later:

```bash
pip install openai-whisper "numpy>=1.23.0"
python -m src.pipeline.run_captioning_demo --config config/demo.yaml --mode auto
```

`auto` tries the configured ASR provider when audio, `ffmpeg`, and the required
packages are available, then falls back to the cached transcript when fallback is
enabled.

## Production Note

In production, a stream adapter would send short audio frames into a pause-aware
utterance chunker, then flush ASR when the speaker pauses or a max latency budget
is reached. This slice keeps file and cached transcript input first so the
submission can execute from end to end in a clean Colab runtime.
