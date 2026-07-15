# OpenVoiceOS STT Plugin - ONNX ASR

An OpenVoiceOS Speech-to-Text plugin backed by the lightweight [onnx-asr](https://github.com/istupakov/onnx-asr) library. This plugin runs offline and supports high-performance models like Nvidia Canary, Parakeet, and OpenAI Whisper via ONNX Runtime.

## Description

This plugin enables OpenVoiceOS to use state-of-the-art ASR models exported to ONNX. It leverages the `onnx-asr` package which provides a unified interface for running various architectures (NeMo, Whisper, GigaAM, etc.) without heavy dependencies like PyTorch.

## Install

To install the plugin, use `pip`. You also need to ensure the backend dependencies are installed.

```bash
pip install ovos-stt-plugin-onnx-asr
```

## Configuration

Configure the plugin in your `mycroft.conf` or user config.

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "nemo-canary-1b-v2",
      "quantization": "int8"
    }
  }
}

```

### Configuration Options

| Option | Default | Description |
| --- | --- | --- |
| `model` | `nemo-canary-1b-v2` | The model ID to load. Can be a specific alias (like `nemo-parakeet-tdt-0.6b-v3`) or a Hugging Face repo ID. |
| `quantization` | `null` | Set to `"int8"` to load the quantized weights for faster, lower-memory CPU inference. Requires the repo to ship `*.int8.onnx` files; loading fails if they are absent. int8 trades a small accuracy drop (typically a few WER points, less on larger models) for ~3-4x smaller models. |
| `use_cuda` | `false` | Run on the GPU via the CUDA execution provider (with a CPU fallback). |
| `providers` | `null` | Explicit list of onnxruntime execution providers, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]` or `["TensorrtExecutionProvider"]`. Takes precedence over `use_cuda`. |

### GPU acceleration

To run on the GPU, install `onnxruntime-gpu` (in place of the default `onnxruntime`) with a matching CUDA/cuDNN runtime, then set `use_cuda`:

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "nemo-parakeet-tdt-0.6b-v3",
      "use_cuda": true
    }
  }
}
```

For finer control (e.g. TensorRT) set `providers` directly; it overrides `use_cuda`.

## Supported Models

The `model` option accepts either a built-in `onnx-asr` alias or any Hugging Face
repo id whose `config.json` declares a supported `model_type` (NeMo
Conformer/FastConformer with CTC, RNN-T, TDT or Canary/AED decoder, Whisper, Vosk,
GigaAM, T-one or wav2vec2-CTC). Streaming, LLM-decoder and TTS checkpoints are not
supported.

`language` is only meaningful for Whisper and Canary models (and
`target_language` for Canary); the plugin passes them automatically only to those
families.

### Built-in aliases (Nvidia NeMo, Whisper, GigaAM, Vosk, T-one)

| Alias | Language(s) |
| --- | --- |
| `nemo-canary-1b-v2` | Multilingual (en, de, fr, es, …) |
| `nemo-parakeet-tdt-0.6b-v3` | Multilingual (25 European langs) |
| `nemo-parakeet-tdt-0.6b-v2` / `nemo-parakeet-ctc-0.6b` / `nemo-parakeet-rnnt-0.6b` | English |
| `gigaam-v3-ctc` / `gigaam-v3-rnnt` / `gigaam-v2-ctc` / `gigaam-v2-rnnt` | Russian |
| `nemo-fastconformer-ru-ctc` / `nemo-fastconformer-ru-rnnt` | Russian |
| `alphacep/vosk-model-ru` / `alphacep/vosk-model-small-ru` / `t-tech/t-one` | Russian |
| `whisper-base` / `onnx-community/whisper-large-v3-turbo` | Multilingual |

### OpenVoiceOS curated models

Language-specific models live in the
[OpenVoiceOS/stt-asr-onnx](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx)
collection and are loaded by repo id. Most ship both fp32 and int8 weights, so
`quantization: "int8"` works; a few older ones are fp32-only (setting int8 then
fails to load). Whisper-based entries in the collection are fp32-only.

| `model` | Language | Architecture |
| --- | --- | --- |
| `OpenVoiceOS/stt-gl-conformer-ctc-large-onnx` | Galician | Conformer-CTC |
| `OpenVoiceOS/stt-fa-fastconformer-hybrid-large-onnx` | Persian | FastConformer (RNN-T + CTC) |
| `OpenVoiceOS/parakeet-tdt_ctc-0.6b-ja-onnx` | Japanese | Parakeet TDT+CTC |
| `OpenVoiceOS/parakeet-ctc-0.6b-vietnamese-onnx` | Vietnamese | Parakeet CTC |
| `OpenVoiceOS/parakeet-rnnt-110m-da-dk-onnx` | Danish | Parakeet RNN-T |
| `OpenVoiceOS/parakeet-rnnt-1.1b-cv17-es-ep18-1270h-onnx` | Spanish | Parakeet RNN-T |
| `OpenVoiceOS/stt-ca-es-conformer-transducer-large-onnx` | Catalan / Spanish | Conformer RNN-T |
| `OpenVoiceOS/stt-eu-conformer-transducer-large-onnx` | Basque | Conformer RNN-T |
| `OpenVoiceOS/whisper-large-v3-ca-punctuated-3370h-onnx` | Catalan | Whisper |
| `OpenVoiceOS/whisper-small-pt-onnx` / `whisper-medium-pt-onnx` / `whisper-large-v3-pt-onnx` | Portuguese | Whisper |
| `OpenVoiceOS/parakeet-{tdt,rnnt,ctc}-1.1b-onnx` / `parakeet-tdt_ctc-110m-onnx` | English | Parakeet |

For the full list of built-in aliases and benchmarks, see the [onnx-asr repository](https://github.com/istupakov/onnx-asr).

### wav2vec2 models

The plugin bundles a small runtime shim that teaches `onnx-asr` about the
`wav2vec2-ctc` model type, so wav2vec2 / XLS-R CTC fine-tunes exported to ONNX load
by repo id with no extra setup. (The shim is inert once
[onnx-asr#1](https://github.com/istupakov/onnx-asr/pull/1) lands in an installed
`onnx-asr` release.) These are CTC models, so `language` is not used.

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "model": "OpenVoiceOS/wav2vec2-base-10k-voxpopuli-ft-en-onnx"
    }
  }
}
```

Ready-to-use conversions are published under the
[OpenVoiceOS](https://huggingface.co/OpenVoiceOS/models?search=wav2vec2) org,
including VoxPopuli base fine-tunes (cs, de, en, es, fi, fr, hr, hu, it, nl, pl, ro,
sk, sl) and XLS-R fine-tunes for Swedish, Icelandic, Faroese, Finnish, Northern Sami
and Serbian.

## Credits

* [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) - The underlying library doing the heavy lifting.

