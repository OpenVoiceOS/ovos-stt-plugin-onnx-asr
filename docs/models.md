# Model types

The plugin loads models with `onnx-asr`. A model is loadable when its
`config.json` declares a `model_type` that `onnx-asr` knows.

Almost all model families in the built-in per-language registry are native to
`onnx-asr`: NeMo Conformer, FastConformer and Citrinet with a CTC, RNN-T, TDT or
AED decoder, Whisper, Vosk, GigaAM and T-one.

One family is not native: wav2vec2. The plugin carries the `wav2vec2-ctc` model
type and registers it with `onnx-asr` at start-up.

## wav2vec2 models

Registration is automatic. wav2vec2 and XLS-R CTC fine-tunes exported to ONNX
load by repo id, with no extra setup:

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-zulu-onnx"
    }
  }
}
```

These are CTC models. They transcribe one language, so the `language` option has
no effect on them.

Ready-to-use conversions are published in the
[OpenVoiceOS](https://huggingface.co/OpenVoiceOS/models?search=wav2vec2) org.

## When registration stops

The plugin does not register a model type that `onnx-asr` already provides. If
the installed `onnx-asr` has the `wav2vec2-ctc` type, the plugin keeps its own
class unused and lets `onnx-asr` load the model. No configuration change is
necessary.

If the internals of `onnx-asr` change and the plugin cannot register the type,
the plugin writes a warning and continues. Native model types keep working, but
wav2vec2 models do not load.
