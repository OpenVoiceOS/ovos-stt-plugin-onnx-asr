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
| `quantization` | `null` | Set to `"int8"` to use quantized models for faster inference and lower memory usage. |
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

This plugin supports any model compatible with `onnx-asr`. Common models include:

**Nvidia NeMo (Multilingual & English)**

* `nemo-canary-1b-v2` (Multilingual, supports English, German, French, Spanish, etc.)
* `nemo-parakeet-tdt-0.6b-v3` (Multilingual)
* `nemo-parakeet-ctc-0.6b` (English)

**OpenAI Whisper**

* `onnx-community/whisper-large-v3-turbo`

**Russian Models (GigaAM)**

* `gigaam-v3-ctc`
* `gigaam-v3-rnnt`

For a full list of supported models and benchmarks, see the [onnx-asr repository](https://github.com/istupakov/onnx-asr).

## Credits

* [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) - The underlying library doing the heavy lifting.

