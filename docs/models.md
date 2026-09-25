# Model types

The plugin loads models with `onnx-asr`. A model is loadable when its
`config.json` declares a `model_type` that `onnx-asr` knows.

`onnx-asr` knows some model types itself. The plugin adds the others at start-up,
so a plain `pip install onnx-asr` is sufficient. No model type needs a different
installation, and you do not have to switch anything on.

Adding a model type makes `onnx-asr` resolve and load models of that architecture.
It does not tell you that a given conversion transcribes correctly. Test the model
you intend to use.

## Supported architectures

| Architecture | Model type | Model type comes from |
|---|---|---|
| NeMo Conformer, FastConformer, Citrinet | `nemo-conformer-ctc`, `nemo-conformer-rnnt`, `nemo-conformer-tdt`, `nemo-conformer-aed` | `onnx-asr` |
| Whisper | `whisper`, `whisper-ort` | `onnx-asr` |
| Vosk, Kaldi transducer | `vosk`, `kaldi-rnnt` | `onnx-asr` |
| GigaAM | `gigaam-v2-ctc`, `gigaam-v2-rnnt`, `gigaam-v3-*` | `onnx-asr` |
| T-one | `t-one-ctc` | `onnx-asr` |
| wav2vec2, XLS-R CTC | `wav2vec2-ctc` | this plugin |
| wav2vec2 with language adapters (MMS) | `wav2vec2-adapters` | this plugin |
| ESPnet E-Branchformer | `espnet-ctc`, `espnet-aed` | this plugin |
| Granite Speech NAR | `granite-nar` | this plugin |
| Moonshine | `moonshine`, `moonshine-tiny`, `moonshine-base` | this plugin |
| Omnilingual ASR CTC | `omnilingual-ctc` | this plugin |
| Paraformer (FunASR) | `paraformer` | this plugin |
| SenseVoice (FunASR) | `sensevoice` | this plugin |
| Speech-LLM | `speech-llm` | this plugin |

"This plugin" means the plugin carries the model type until `onnx-asr` releases
it. The model type then comes from `onnx-asr`, with no change to your
configuration.

## How to load a model

Give the repo id of an ONNX conversion. The model type comes from its
`config.json`:

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

Ready-to-use conversions are published in the
[OpenVoiceOS](https://huggingface.co/OpenVoiceOS/models) org.

CTC models such as `wav2vec2-ctc` transcribe one language. The `language` option
has no effect on them.

## Short names

The plugin gives a short name to one OpenVoiceOS conversion of each architecture,
so that the name alone loads a model:

| Short name | Repository |
|---|---|
| `espnet-ctc` | `OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx` |
| `espnet-aed` | `OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx` |
| `granite-nar` | `OpenVoiceOS/granite-speech-4.1-2b-nar-onnx` |
| `moonshine-tiny` | `OpenVoiceOS/moonshine-tiny-onnx` |
| `moonshine-base` | `OpenVoiceOS/moonshine-base-onnx` |
| `omnilingual-ctc` | `OpenVoiceOS/omnilingual-asr-ctc-1b-onnx` |
| `paraformer` | `OpenVoiceOS/paraformer-zh-onnx` |
| `sensevoice` | `OpenVoiceOS/sensevoice-small-onnx` |
| `speech-llm` | `OpenVoiceOS/qwen3-asr-0.6b-onnx` |
| `wav2vec2-adapters` | `OpenVoiceOS/mms-1b-all-onnx` |

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "sensevoice"
    }
  }
}
```

The table is a convenience of this plugin over the OpenVoiceOS conversions. It is
not a list of every model an architecture can load: give the repo id to load any
other conversion. A short name that `onnx-asr` maps itself keeps the repository
`onnx-asr` gives it.

An architecture with one conversion per language, such as `wav2vec2-ctc`, gets no
short name, because no single name can name a model. Give the repo id for those.

## When the plugin stops adding a model type

The plugin never replaces a model type that `onnx-asr` provides. When the
installed `onnx-asr` has the type, the plugin keeps its own class unused and lets
`onnx-asr` load the model.

Each architecture is added on its own. If one architecture cannot be added, the
plugin writes a warning and adds the others.

Adding `wav2vec2-adapters` is the one exception to that isolation. The
architecture downloads its language adapters one at a time, which needs a change
to the model loader that `onnx-asr` uses for every model. The plugin makes that
change only when it adds `wav2vec2-adapters`, and every other model type keeps the
behaviour it had.

If the internals of `onnx-asr` change and the plugin cannot add its model types,
the plugin writes a warning and continues. Model types native to `onnx-asr` keep
working. The other model types do not load.
