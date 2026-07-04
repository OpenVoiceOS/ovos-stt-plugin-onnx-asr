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

Curated single-language and regional models live in the
[OpenVoiceOS/stt-asr-onnx](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx)
collection, converted from reputable NeMo Conformer/Parakeet checkpoints and
loaded by repo id. Repos are named `<author>-<model>-onnx` to avoid collisions
between same-named finetunes. Most ship both fp32 and int8 weights, so
`quantization: "int8"` works (a few large models are fp32-only). See the
collection for the exhaustive, up-to-date list; the highlights below are grouped
by language.

**Indian languages — AI4Bharat IndicConformer** (`ai4bharat-indicconformer-<lang>-onnx`,
CTC), 22 languages: `as` `bn` `brx` `doi` `gu` `hi` `kn` `kok` `ks` `mai` `ml`
`mni` `mr` `ne` `or` `pa` `sa` `sat` `sd` `ta` `te` `ur`. IISc **Vaani
FastConformer** (`artpark-iisc-vaani-fastconformer-<lang>-onnx`) adds `hi` `kn`
`ml` `or` `te` and a `multi` multilingual model.

**European — NVIDIA Conformer** (`nvidia-<lang>-conformer-{ctc,transducer}-large-onnx`):
`en` (also `-small`, `-transducer-xlarge`), `de`, `fr`, `es`, `it`, `ru`, `ca`,
`hr` (Croatian), `be` (Belarusian), `eo` (Esperanto), `rw` (Kinyarwanda), and
`nvidia-kab-conformer-transducer-large-onnx` (Kabyle).

**Parakeet (per-language)**

| `model` | Language | Architecture |
| --- | --- | --- |
| `OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx` | Persian | FastConformer (RNN-T + CTC) |
| `OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx` | Japanese | Parakeet TDT+CTC |
| `OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx` | Vietnamese | Parakeet CTC |
| `OpenVoiceOS/nvidia-parakeet-rnnt-110m-da-dk-onnx` | Danish | Parakeet RNN-T |
| `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-{pl,pt,nl,et,sl}-onnx` | Polish / Portuguese / Dutch / Estonian / Slovenian | Parakeet TDT |
| `OpenVoiceOS/nvidia-parakeet-{tdt,rnnt,ctc}-1.1b-onnx` / `nvidia-parakeet-tdt_ctc-110m-onnx` | English | Parakeet |

**Iberian**

| `model` | Language | Architecture |
| --- | --- | --- |
| `OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx` | Galician | Conformer-CTC |
| `OpenVoiceOS/parakeet-rnnt-1.1b-cv17-es-ep18-1270h-onnx` | Spanish | Parakeet RNN-T |
| `OpenVoiceOS/stt-ca-es-conformer-transducer-large-onnx` | Catalan / Spanish | Conformer RNN-T |
| `OpenVoiceOS/bsc-lt-los-conformer-transducer-large-onnx` (+ `-punctuated`) | Languages of Spain (es/ca/gl/eu) | Conformer RNN-T |
| `OpenVoiceOS/stt-eu-conformer-transducer-large-onnx` / `stt-eu-conformer-ctc-large-onnx` / `hitz-eu-conformer-transducer-large-v2-onnx` | Basque | Conformer |
| `OpenVoiceOS/hitz-eseu-conformer-transducer-large-onnx` / `hitz-bbs-s2tc-conformer-transducer-large-onnx` | Basque + Spanish | Conformer RNN-T |

The collection also contains Whisper conversions for several languages (Portuguese,
Catalan, and various national-lab finetunes); load those the same way by repo id.

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


[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
