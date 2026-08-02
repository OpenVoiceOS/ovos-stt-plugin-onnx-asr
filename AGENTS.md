# ovos-stt-plugin-onnx-asr

OpenVoiceOS Speech-to-Text plugin backed by the `onnx-asr` library. Runs offline via ONNX Runtime, supporting NeMo (Canary, Parakeet), Whisper, and GigaAM models without PyTorch.

## Setup

```bash
pip install .
```

Runtime deps: `ovos-plugin-manager>=2.1.1,<3.0.0`, `onnx_asr`.

## Test

No test suite exists. The only runnable check is the package build (mirrors CI):

```bash
python setup.py sdist bdist_wheel && pip install .
```

A manual smoke test lives in the `__main__` block of `ovos_stt_plugin_onnxasr/__init__.py`, but it hardcodes a local WAV path and must be edited before use.

## Lint/Typecheck

None configured.

## Layout

- `ovos_stt_plugin_onnxasr/__init__.py` — `OnnxASR(STT)`, the single plugin class. Loads the model in `__init__` via `onnx_asr.load_model(model_id, quantization=...)`; transcribes in `execute()` from `AudioData`.
- `ovos_stt_plugin_onnxasr/version.py` — semver block (managed by CI).
- `setup.py` — packaging; entry point.

Entry-point group: `opm.stt` → `ovos-stt-plugin-onnx-asr = ovos_stt_plugin_onnxasr:OnnxASR`. This is an OVOS Plugin Manager (OPM) STT plugin.

## Conventions

- Branches: work on `dev`, stable on `master`. NEVER `main`.
- Never edit `version.py`; gh-automations bumps semver from conventional-commit prefixes (`feat:`, `fix:`, `feat!:`).
- New repos private by default.
- Commit identity: `JarbasAi <jarbasai@mailfence.com>`.
- Reference `OpenVoiceOS/gh-automations` reusable workflows at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary (no history, no dates) in docs, commits, comments.
- CI is provided by OpenVoiceOS/gh-automations.

## Gotchas

- `available_languages` returns an empty `set()`; OVOS cannot enumerate supported languages and language validation may fail (see open issue #2 "Error when defining language").
- `execute()` passes both `language` and `target_language` to `recognize()`; not all `onnx-asr` models accept these kwargs, so model choice affects behavior.
- The `__main__` block contains a hardcoded absolute path (`/home/miro/PycharmProjects/...`) and is dev scaffolding, not a real entry point.
- `setup.py` `url` points to `OpenVoiceOS/ovos-stt-plugin-onnx-asr`, but the repo lives at `TigreGotico/ovos-stt-plugin-onnx-asr`.
- No `.gitignore`; an `ovos_stt_plugin_onnx_asr.egg-info/` build artifact exists on disk (untracked).
