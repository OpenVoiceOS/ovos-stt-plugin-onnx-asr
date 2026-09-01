# OpenVoiceOS STT Plugin - ONNX ASR

An OpenVoiceOS Speech-to-Text plugin backed by the lightweight [onnx-asr](https://github.com/istupakov/onnx-asr) library. This plugin runs offline and supports high-performance models like Nvidia Canary, Parakeet, and OpenAI Whisper via ONNX Runtime.

## Description

This plugin enables OpenVoiceOS to use state-of-the-art ASR models exported to ONNX. It leverages the `onnx-asr` package which provides a unified interface for running various architectures (NeMo, Whisper, GigaAM, etc.) without heavy dependencies like PyTorch.

## Install

To install the plugin, use `pip`. You also need to ensure the backend dependencies are installed.

```bash
pip install ovos-stt-plugin-onnx-asr
```

To run this plugin as a standalone server in a container, see
[docs/docker.md](docs/docker.md) and [docs/deployment.md](docs/deployment.md).
For which model families load and how, see [docs/models.md](docs/models.md).

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

### Per-language model resolution

The plugin ships a built-in best-model-per-language registry (`defaults.LANG_DEFAULTS`, ~90 languages — dedicated fine-tunes from the [OpenVoiceOS/stt-asr-onnx](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx) collection, with parakeet-tdt-0.6b-v3 and whisper-base as multilingual coverage). Whisper ONNX exports are supported like any other model. The model for a request's language resolves in this order, trying the full BCP-47 tag before the nearest usable tag at each level:

1. `lang2model` in the plugin config
2. `ONNX_ASR_DEFAULT_<LANG>` environment variables — underscores map to dashes, so `ONNX_ASR_DEFAULT_PT=...` sets `pt` and `ONNX_ASR_DEFAULT_PT_BR=...` sets `pt-BR` (handy for containers)
3. the configured `model`
4. the built-in registry
5. `nemo-canary-1b-v2`, for a language the registry does not hold

Language tags are compared as OVOS-INTENT-2 §2 specifies, via [`ovos-spec-tools`](https://github.com/OpenVoiceOS/ovos-spec-tools): case insensitively, with underscores accepted for hyphens, falling back to the nearest usable tag. So `en_US`, `en-GB` and `EN` all reach the `en` entry, while `ga` (Irish) never reaches the `gaa` (Ga) one.

A configured `model` serves every language, because it is your instruction and a registry entry is only the best guess for a language you said nothing about. To let the registry pick per language, leave `model` unset and name exceptions in `lang2model`.

Models load lazily on the first request for their language and stay cached in memory, so a single instance (or one `ovos-stt-server` container) serves every language with the best available model.

A registry model whose repository holds no quantized weights loads fp32, even with `quantization` set, so that setting cannot take a language away.

### Configuration Options

| Option | Default | Description |
| --- | --- | --- |
| `model` | `nemo-canary-1b-v2` | The model ID to load. Can be a specific alias (like `nemo-parakeet-tdt-0.6b-v3`) or a Hugging Face repo ID. Set it and it serves every language, ahead of the built-in registry; leave it unset for per-language routing. |
| `lang2model` | `{}` | Optional per-language routing map, e.g. `{"ru": "gigaam-v2-rnnt", "gl": "OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx"}`. The model for a language loads lazily on the first request and stays cached in memory, so a single instance (or a single `ovos-stt-server` container) can serve every configured language with the best model for each. Unmapped languages fall back to `model`. |
| `quantization` | `null` | Set to `"int8"` to load the quantized weights for faster, lower-memory CPU inference. Requires the repo to ship `*.int8.onnx` files; loading fails if they are absent, except for a model the built-in registry picked, which then loads fp32. int8 trades a small accuracy drop (typically a few WER points, less on larger models) for ~3-4x smaller models. |
| `use_cuda` | `false` | Run on the GPU via the CUDA execution provider (with a CPU fallback). |
| `providers` | `null` | Explicit list of onnxruntime execution providers, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]` or `["TensorrtExecutionProvider"]`. Takes precedence over `use_cuda`. |
| `cpu_models_only` | `false` | Restrict model selection to models practical on CPU-only hardware. See below. |

### CPU-only deployments

A satellite or server that will only ever run on CPU should set `cpu_models_only`
rather than trust every operator, or every future config change, to keep clear of
a model that expects a GPU. On, it drops any model whose id advertises a
parameter count of 0.6B or more (the size the catalogue already writes into model
ids like `nemo-canary-1b-v2` or `qwen3-asr-0.6b-onnx`) from the built-in
per-language registry, and it swaps the ultimate fallback from
`nemo-canary-1b-v2` (1B) to `whisper-base` (74M, the plugin's other multilingual
coverage model).

```json
{
  "stt": {
    "module": "ovos-stt-plugin-onnx-asr",
    "ovos-stt-plugin-onnx-asr": {
      "cpu_models_only": true
    }
  }
}
```

Naming an excluded model explicitly, through `model` or `lang2model`, is refused
at startup rather than silently swapped for a different one: the config is
wrong, and running a different model than the one named is worse than saying so.
`cpu_models_only` is off by default, so leaving it unset keeps every model in
the catalogue selectable, exactly as before the option existed.

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
GigaAM, T-one or wav2vec2-CTC — plus, with the fork pin below: `speech-llm`,
`espnet-ctc`/`espnet-aed`, `granite-nar`, `sensevoice`, `moonshine` and
`wav2vec2-adapters`). Streaming and TTS checkpoints are not supported.

`language` is meaningful for Whisper, Canary/AED, speech-llm and
wav2vec2-adapters models (`target_language` for Canary); the plugin passes it
automatically only to those families.

### Choosing a model: architecture tiers

Models in the [STT/ASR - onnx collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx-699321e8732462509c642fbe)
span several architecture families. They trade accuracy against size and speed in
a fairly consistent order. RTF = real-time factor on a mid-range desktop CPU
(AMD Ryzen 5 7600); lower is faster, below 1.0 is faster than real time.

| Tier | Families | Size | CPU speed | When to use |
| --- | --- | --- | --- | --- |
| Speech-LLM | `speech-llm` (Qwen3-ASR, Canary-Qwen, Granite, Voxtral, AMALIA), `granite-nar` | 1–19 GB | RTF ~0.3–1.3 (int8); NAR variant ~0.3 fp32 | Best accuracy available. Use on a server with RAM to spare, or when transcription quality matters more than latency. The 9B models effectively need a GPU. |
| Attention encoder-decoder | `nemo-conformer-aed` (Canary, Cohere Transcribe), Whisper large / `espnet-aed` | 1–8 GB | RTF ~0.2–0.9 (int8) | Strong accuracy with punctuation and casing. Good server default. Cohere Transcribe covers 14 languages at RTF ~0.19 int8. |
| Conformer CTC / transducer | `nemo-conformer-ctc/rnnt/tdt` (Parakeet), GigaAM, Vosk | 0.1–2.5 GB | RTF ~0.05–0.3 | The practical sweet spot for assistants: fast, small, accurate for their languages. `nemo-parakeet-tdt-0.6b-v3` is the plugin default for good reason. |
| Compact Whisper | whisper small/medium, distil, lite-whisper | 0.2–1 GB (int8) | RTF ~0.3–1.0 | Many language fine-tunes only exist as Whisper checkpoints. Use the fine-tune for your language when one exists. |
| Tiny specialists | `sensevoice` (zh/en/ja/ko/yue), `moonshine` (en) | 60–950 MB | RTF ~0.02–0.04 | Fastest options by far. Use on constrained hardware (Raspberry Pi class) for their languages. |
| wav2vec2 CTC / MMS | `wav2vec2-ctc`, `wav2vec2-adapters` (MMS, 1100+ languages) | ~1 GB int8 base | RTF ~0.15 | Lowest accuracy tier: character-level output, no punctuation or casing, narrow training domains. **But for hundreds of languages this is the only model that exists.** Something is better than nothing. |

Rules of thumb:

- A dedicated fine-tune for your language beats a bigger multilingual model more
  often than not. Check the collection for your language tag first.
- If your language has a Whisper or Conformer fine-tune, prefer it over MMS.
  Use `wav2vec2-adapters` (MMS) when nothing else covers the language.
- int8 quantization roughly quarters the size. For autoregressive models it is
  usually ~2x faster on CPU; for single-pass models (NAR, CTC) it can be slower —
  check the model card, each states its measured numbers.
- Speech-LLM and AED models emit punctuation and casing; CTC-family models
  generally do not.

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
`quantization: "int8"` works (a few large models are fp32-only; NeMo CTC
exports like the Citrinet and FastConformer-Hybrid pc fleets ship fp32 only).

Every model in the collection, grouped by language:

| Language | `model` | Architecture |
| --- | --- | --- |
| Indian languages (multilingual) | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-multi-onnx` | FastConformer |
| Arabic | `OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pc_v1.0_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Arabic | `OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Armenian | `OpenVoiceOS/stt_hy-AM_citrinet_512_armenian-CV17.0_onnx` | Citrinet CTC |
| Armenian | `OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Assamese | `OpenVoiceOS/ai4bharat-indicconformer-as-onnx` | IndicConformer CTC |
| Bangla | `OpenVoiceOS/ai4bharat-indicconformer-bn-onnx` | IndicConformer CTC |
| Basque | `OpenVoiceOS/hitz-eu-conformer-transducer-large-v2-onnx` | Conformer RNN-T |
| Basque | `OpenVoiceOS/stt-eu-conformer-ctc-large-onnx` | Conformer CTC |
| Basque | `OpenVoiceOS/stt-eu-conformer-transducer-large-onnx` | Conformer RNN-T |
| Belarusian | `OpenVoiceOS/nvidia-be-conformer-ctc-large-onnx` | Conformer CTC |
| Belarusian | `OpenVoiceOS/nvidia-be-conformer-transducer-large-onnx` | Conformer RNN-T |
| Belarusian | `OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Bodo | `OpenVoiceOS/ai4bharat-indicconformer-brx-onnx` | IndicConformer CTC |
| Catalan | `OpenVoiceOS/neongeckocom-stt_ca_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Catalan | `OpenVoiceOS/nvidia-ca-conformer-ctc-large-onnx` | Conformer CTC |
| Catalan | `OpenVoiceOS/nvidia-ca-conformer-transducer-large-onnx` | Conformer RNN-T |
| Catalan | `OpenVoiceOS/whisper-large-v3-ca-punctuated-3370h-onnx` | Whisper |
| Catalan / Spanish | `OpenVoiceOS/stt-ca-es-conformer-transducer-large-onnx` | Conformer RNN-T |
| Catalan / Spanish | `OpenVoiceOS/whisper-large-v3-tiny-caesar-onnx` | Whisper |
| Chinese | `OpenVoiceOS/stt_zh_citrinet_1024_gamma_0_25_onnx` | Citrinet CTC |
| Chinese | `OpenVoiceOS/stt_zh_citrinet_512_onnx` | Citrinet CTC |
| Croatian | `OpenVoiceOS/nvidia-hr-conformer-ctc-large-onnx` | Conformer CTC |
| Croatian | `OpenVoiceOS/nvidia-hr-conformer-transducer-large-onnx` | Conformer RNN-T |
| Croatian | `OpenVoiceOS/stt_hr_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Danish | `OpenVoiceOS/nvidia-parakeet-rnnt-110m-da-dk-onnx` | Parakeet RNN-T |
| Dogri | `OpenVoiceOS/ai4bharat-indicconformer-doi-onnx` | IndicConformer CTC |
| Dutch | `OpenVoiceOS/neongeckocom-stt_nl_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Dutch | `OpenVoiceOS/stt_nl_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Dutch | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-nl-onnx` | Parakeet TDT |
| English | `OpenVoiceOS/neongeckocom-stt_en_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| English | `OpenVoiceOS/nvidia-en-conformer-ctc-large-onnx` | Conformer CTC |
| English | `OpenVoiceOS/nvidia-en-conformer-ctc-small-onnx` | Conformer CTC |
| English | `OpenVoiceOS/nvidia-en-conformer-transducer-large-onnx` | Conformer RNN-T |
| English | `OpenVoiceOS/nvidia-en-conformer-transducer-xlarge-onnx` | Conformer RNN-T |
| English | `OpenVoiceOS/nvidia-parakeet-ctc-1.1b-onnx` | Parakeet CTC |
| English | `OpenVoiceOS/nvidia-parakeet-rnnt-1.1b-onnx` | Parakeet RNN-T |
| English | `OpenVoiceOS/nvidia-parakeet-tdt-1.1b-onnx` | Parakeet TDT |
| English | `OpenVoiceOS/nvidia-parakeet-tdt_ctc-110m-onnx` | Parakeet TDT+CTC |
| English | `OpenVoiceOS/stt_en_citrinet_1024_gamma_0_25_onnx` | Citrinet CTC |
| English | `OpenVoiceOS/stt_en_citrinet_256_gamma_0_25_onnx` | Citrinet CTC |
| English | `OpenVoiceOS/stt_en_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| English | `OpenVoiceOS/stt_en_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Esperanto | `OpenVoiceOS/nvidia-eo-conformer-ctc-large-onnx` | Conformer CTC |
| Esperanto | `OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx` | Conformer RNN-T |
| Estonian | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-et-onnx` | Parakeet TDT |
| Fanti | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-fante-onnx` | wav2vec2 CTC |
| Filipino | `OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx` | FastConformer-Hybrid |
| Fon | `OpenVoiceOS/misterkissi-whisper-small-fongbe-onnx` | Whisper |
| French | `OpenVoiceOS/neongeckocom-stt_fr_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| French | `OpenVoiceOS/nvidia-fr-conformer-ctc-large-onnx` | Conformer CTC |
| French | `OpenVoiceOS/nvidia-fr-conformer-transducer-large-onnx` | Conformer RNN-T |
| French | `OpenVoiceOS/stt_fr_citrinet_1024_gamma_0_25_onnx` | Citrinet CTC |
| French | `OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Galician | `OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx` | Conformer CTC |
| Georgian | `OpenVoiceOS/stt_ka_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| German | `OpenVoiceOS/neongeckocom-stt_de_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| German | `OpenVoiceOS/nvidia-de-conformer-ctc-large-onnx` | Conformer CTC |
| German | `OpenVoiceOS/nvidia-de-conformer-transducer-large-onnx` | Conformer RNN-T |
| German | `OpenVoiceOS/stt_de_citrinet_1024_onnx` | Citrinet CTC |
| German | `OpenVoiceOS/stt_de_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Gujarati | `OpenVoiceOS/ai4bharat-indicconformer-gu-onnx` | IndicConformer CTC |
| Haitian Creole | `OpenVoiceOS/misterkissi-whisper-small-haitian-creole-onnx` | Whisper |
| Hindi | `OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` | IndicConformer CTC |
| Hindi | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-hi-onnx` | FastConformer |
| Ikposo | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-kposo-onnx` | wav2vec2 CTC |
| Ikposo | `OpenVoiceOS/misterkissi-whisper-small-kposo-onnx` | Whisper |
| Ga | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx` | wav2vec2 CTC — Ga, the Kwa language of Accra (`gaa`), not Irish |
| Italian | `OpenVoiceOS/neongeckocom-stt_it_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Italian | `OpenVoiceOS/nvidia-it-conformer-ctc-large-onnx` | Conformer CTC |
| Italian | `OpenVoiceOS/nvidia-it-conformer-transducer-large-onnx` | Conformer RNN-T |
| Italian | `OpenVoiceOS/stt_it_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Japanese | `OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx` | Parakeet TDT+CTC |
| Kabyle | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-1b-kabyle-onnx` | wav2vec2 CTC |
| Kabyle | `OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx` | Conformer RNN-T |
| Kannada | `OpenVoiceOS/ai4bharat-indicconformer-kn-onnx` | IndicConformer CTC |
| Kannada | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-kn-onnx` | FastConformer |
| Kasem | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-kasem-onnx` | wav2vec2 CTC |
| Kashmiri | `OpenVoiceOS/ai4bharat-indicconformer-ks-onnx` | IndicConformer CTC |
| Kazakh / Russian | `OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Kinyarwanda | `OpenVoiceOS/nvidia-rw-conformer-ctc-large-onnx` | Conformer CTC |
| Kinyarwanda | `OpenVoiceOS/nvidia-rw-conformer-transducer-large-onnx` | Conformer RNN-T |
| Konkani | `OpenVoiceOS/ai4bharat-indicconformer-kok-onnx` | IndicConformer CTC |
| Korean | `OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx` | Citrinet CTC |
| Lingala | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-lingala-onnx` | wav2vec2 CTC |
| Lomwe | `OpenVoiceOS/misterkissi-whisper-small-lomwe-onnx` | Whisper |
| Maithili | `OpenVoiceOS/ai4bharat-indicconformer-mai-onnx` | IndicConformer CTC |
| Malagasy | `OpenVoiceOS/misterkissi-whisper-small-malagasy-onnx` | Whisper |
| Malayalam | `OpenVoiceOS/ai4bharat-indicconformer-ml-onnx` | IndicConformer CTC |
| Malayalam | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-ml-onnx` | FastConformer |
| Manipuri | `OpenVoiceOS/ai4bharat-indicconformer-mni-onnx` | IndicConformer CTC |
| Marathi | `OpenVoiceOS/ai4bharat-indicconformer-mr-onnx` | IndicConformer CTC |
| Nepali | `OpenVoiceOS/ai4bharat-indicconformer-ne-onnx` | IndicConformer CTC |
| Nyankole | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-runyankore-onnx` | wav2vec2 CTC |
| Odia | `OpenVoiceOS/ai4bharat-indicconformer-or-onnx` | IndicConformer CTC |
| Odia | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-or-onnx` | FastConformer |
| Persian | `OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx` | FastConformer-Hybrid |
| Persian | `OpenVoiceOS/stt_fa_fastconformer_hybrid_large_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Polish | `OpenVoiceOS/stt_pl_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Polish | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx` | Parakeet TDT |
| Portuguese | `OpenVoiceOS/neongeckocom-stt_pt_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Portuguese | `OpenVoiceOS/stt_pt_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Portuguese | `OpenVoiceOS/whisper-large-v3-pt-onnx` | Whisper |
| Portuguese | `OpenVoiceOS/whisper-medium-pt-onnx` | Whisper |
| Portuguese | `OpenVoiceOS/whisper-small-pt-onnx` | Whisper |
| Portuguese | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx` | Parakeet TDT |
| Punjabi | `OpenVoiceOS/ai4bharat-indicconformer-pa-onnx` | IndicConformer CTC |
| Russian | `OpenVoiceOS/nvidia-ru-conformer-ctc-large-onnx` | Conformer CTC |
| Russian | `OpenVoiceOS/nvidia-ru-conformer-transducer-large-onnx` | Conformer RNN-T |
| Russian | `alphacep/vosk-model-ru` | Vosk |
| Sanskrit | `OpenVoiceOS/ai4bharat-indicconformer-sa-onnx` | IndicConformer CTC |
| Santali | `OpenVoiceOS/ai4bharat-indicconformer-sat-onnx` | IndicConformer CTC |
| Shona | `OpenVoiceOS/misterkissi-whisper-small-shona-onnx` | Whisper |
| Sindhi | `OpenVoiceOS/ai4bharat-indicconformer-sd-onnx` | IndicConformer CTC |
| Slovenian | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx` | Parakeet TDT |
| Southern Sotho | `OpenVoiceOS/misterkissi-whisper-small-sesotho-onnx` | Whisper |
| Spanish | `OpenVoiceOS/bsc-lt-los-conformer-transducer-large-onnx` | Conformer RNN-T |
| Spanish | `OpenVoiceOS/bsc-lt-los-conformer-transducer-large-punctuated-onnx` | Conformer RNN-T |
| Spanish | `OpenVoiceOS/hitz-bbs-s2tc-conformer-transducer-large-onnx` | Conformer RNN-T |
| Spanish | `OpenVoiceOS/hitz-eseu-conformer-transducer-large-onnx` | Conformer RNN-T |
| Spanish | `OpenVoiceOS/neongeckocom-stt_es_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Spanish | `OpenVoiceOS/nvidia-es-conformer-ctc-large-onnx` | Conformer CTC |
| Spanish | `OpenVoiceOS/nvidia-es-conformer-transducer-large-onnx` | Conformer RNN-T |
| Spanish | `OpenVoiceOS/parakeet-rnnt-1.1b-cv17-es-ep18-1270h-onnx` | Parakeet RNN-T |
| Spanish | `OpenVoiceOS/stt_es_citrinet_1024_gamma_0_25_onnx` | Citrinet CTC |
| Spanish | `OpenVoiceOS/stt_es_citrinet_512_onnx` | Citrinet CTC |
| Spanish | `OpenVoiceOS/stt_es_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Tajik | `alphacep/vosk-model-tg` | Vosk |
| Tamil | `OpenVoiceOS/ai4bharat-indicconformer-ta-onnx` | IndicConformer CTC |
| Telugu | `OpenVoiceOS/ai4bharat-indicconformer-te-onnx` | IndicConformer CTC |
| Telugu | `OpenVoiceOS/artpark-iisc-vaani-fastconformer-te-onnx` | FastConformer |
| Tigre | `OpenVoiceOS/misterkissi-whisper-small-tigre-onnx` | Whisper |
| Tsonga | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-tsonga-onnx` | wav2vec2 CTC |
| Tswana | `OpenVoiceOS/misterkissi-whisper-small-setswana-onnx` | Whisper |
| Ukrainian | `OpenVoiceOS/neongeckocom-stt_uk_citrinet_512_gamma_0_25_onnx` | Citrinet CTC |
| Ukrainian | `OpenVoiceOS/stt_uk_citrinet_1024_gamma_0_25_onnx` | Citrinet CTC |
| Umbundu | `OpenVoiceOS/misterkissi-whisper-small-umbundu-onnx` | Whisper |
| Unknown language [ua] | `OpenVoiceOS/stt_ua_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Urdu | `OpenVoiceOS/ai4bharat-indicconformer-ur-onnx` | IndicConformer CTC |
| Uzbek | `OpenVoiceOS/asr-uz-fastconformer-large-onnx` | FastConformer |
| Uzbek | `OpenVoiceOS/stt_uz_fastconformer_hybrid_large_pc_onnx` | FastConformer-Hybrid (CTC, punct+case) |
| Vagla | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-vagla-onnx` | wav2vec2 CTC |
| Vai | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-vai-onnx` | wav2vec2 CTC |
| Vai | `OpenVoiceOS/misterkissi-whisper-small-vai-onnx` | Whisper |
| Vietnamese | `OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx` | Parakeet CTC |
| Xhosa | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-xhosa-onnx` | wav2vec2 CTC |
| Zulu | `OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-zulu-onnx` | wav2vec2 CTC |

Every curated repo carries a model card with `base_model` metadata pointing to the
upstream checkpoint and the training source. See the collection for the exhaustive,
up-to-date list.

For the full list of built-in aliases and benchmarks, see the [onnx-asr repository](https://github.com/istupakov/onnx-asr).

### wav2vec2 models

wav2vec2 and XLS-R CTC fine-tunes load by repo id with no extra setup. See
[docs/models.md](docs/models.md).

### Fork model families (draft, pinned to a fork branch)

`istupakov/onnx-asr` does not yet support three model families used by some of
our HF exports: **ESPnet E-Branchformer** (CTC and attention-decoder variants)
and **Speech-LLM** (audio encoder + projector + causal LM, covers both plain
Qwen3-ASR-style prompting and the SALM variant used by Canary-Qwen). Until
that lands upstream, this plugin can only load those repos against
[TigreGotico/onnx-asr](https://github.com/TigreGotico/onnx-asr)'s `integration`
branch — see `requirements.txt`. This is why the PR that added this section
stays a draft: a git ref in `install_requires` breaks PyPI publishing, so it
must revert to a normal version floor once the upstream PRs merge and release.

Upstream tracking PRs (parity benchmarks and RTF numbers in each):

* [istupakov/onnx-asr#3](https://github.com/TigreGotico/onnx-asr/pull/3) — speech-llm model family (Qwen3-ASR)
* [istupakov/onnx-asr#4](https://github.com/TigreGotico/onnx-asr/pull/4) — ESPnet E-Branchformer models (espnet-ctc / espnet-aed) + w2v-BERT 2.0 preprocessor
* [istupakov/onnx-asr#5](https://github.com/TigreGotico/onnx-asr/pull/5) — SALM encoder shape for speech-llm (Canary-Qwen-2.5B)

No plugin config changes are needed to use these models: set `model` to the
HF repo id as usual, the model type comes from the repo's `config.json`. The
plugin also passes a `language` hint to Speech-LLM models the same way it
does for Whisper and Canary.

| Model | Family | HF repo | Fork branch |
| --- | --- | --- | --- |
| INESC-ID e-branchformer (European Portuguese), CTC | `espnet-ctc` | [OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx](https://huggingface.co/OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx) | `integration` |
| INESC-ID e-branchformer (European Portuguese), attention decoder | `espnet-aed` | [OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx](https://huggingface.co/OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx) | `integration` |
| Qwen3-ASR 0.6B | `speech-llm` | [OpenVoiceOS/qwen3-asr-0.6b-onnx](https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx) | `integration` |
| Qwen3-ASR 1.7B | `speech-llm` | [OpenVoiceOS/qwen3-asr-1.7b-onnx](https://huggingface.co/OpenVoiceOS/qwen3-asr-1.7b-onnx) | `integration` |
| Canary-Qwen 2.5B (SALM) | `speech-llm` | [OpenVoiceOS/canary-qwen-2.5b-onnx](https://huggingface.co/OpenVoiceOS/canary-qwen-2.5b-onnx) | `integration` (needs the SALM commit, included) |
| Granite Speech 3.3 2B | `speech-llm` | [OpenVoiceOS/granite-speech-3.3-2b-onnx](https://huggingface.co/OpenVoiceOS/granite-speech-3.3-2b-onnx) | `integration` |
| INESC-ID Whisper large-v3 (European Portuguese fine-tune) | Whisper | [OpenVoiceOS/inesc-id-whisperlv3-ft-ep-onnx](https://huggingface.co/OpenVoiceOS/inesc-id-whisperlv3-ft-ep-onnx) | none — Whisper is already supported |
| Camões Whisper (Portuguese) | Whisper | [OpenVoiceOS/camoes-whisper-asr-onnx](https://huggingface.co/OpenVoiceOS/camoes-whisper-asr-onnx) | none — Whisper is already supported |

## Credits

* [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) - The underlying library doing the heavy lifting.


[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
