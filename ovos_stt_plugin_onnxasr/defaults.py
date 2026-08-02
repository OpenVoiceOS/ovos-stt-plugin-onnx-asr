"""Built-in best-model-per-language registry.

Maps lowercase BCP-47 tags (full tag or primary subtag) to the best onnx-asr
compatible model we know of for that language — dedicated fine-tunes first,
multilingual models as coverage fillers. Most exports live in the
`OpenVoiceOS/stt-asr-onnx` HuggingFace collection:
https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx

Resolution order (see ``OnnxASR``): plugin config ``lang2model`` >
``ONNX_ASR_DEFAULT_<LANG>`` environment variables > this registry >
the configured default ``model``. Full tags win over primary subtags at
every level, so ``pt-br`` can point somewhere other than ``pt``.
"""
import os
from typing import Dict, Optional

# multilingual coverage models
_PARAKEET_V3 = "nemo-parakeet-tdt-0.6b-v3"  # 25 European languages
_WHISPER = "whisper-base"                   # openai whisper-base, 99 languages
_VAANI = "OpenVoiceOS/artpark-iisc-vaani-fastconformer-multi-onnx"

LANG_DEFAULTS: Dict[str, str] = {}

# parakeet-tdt-0.6b-v3 languages (its dedicated per-language fine-tunes below
# override the ones that have them)
for _l in ("bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl "
           "es sv ru uk").split():
    LANG_DEFAULTS[_l] = _PARAKEET_V3

# whisper-base as long-tail coverage
for _l in ("ar zh ja ko tr vi th id ms he fa sq sr mk bs is no nn cy af "
           "sw ur bn as").split():
    LANG_DEFAULTS[_l] = _WHISPER

# dedicated fine-tunes — better than the multilingual fillers on their language
LANG_DEFAULTS.update({
    "ru": "gigaam-v2-rnnt",
    "pt": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx",
    "nl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-nl-onnx",
    "et": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-et-onnx",
    "sl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx",
    "es": "OpenVoiceOS/stt_es_fastconformer_hybrid_large_pc_onnx",
    "de": "OpenVoiceOS/stt_de_fastconformer_hybrid_large_pc_onnx",
    "fr": "OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx",
    "it": "OpenVoiceOS/stt_it_fastconformer_hybrid_large_pc_onnx",
    "hr": "OpenVoiceOS/stt_hr_fastconformer_hybrid_large_pc_onnx",
    "ca": "OpenVoiceOS/nvidia-ca-conformer-transducer-large-onnx",
    "eu": "OpenVoiceOS/hitz-eu-conformer-transducer-large-v2-onnx",
    "gl": "OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx",
    "be": "OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx",
    "eo": "OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx",
    "rw": "OpenVoiceOS/nvidia-rw-conformer-transducer-large-onnx",
    "kab": "OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx",
    "fa": "OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx",
    "ja": "OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx",
    "vi": "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
    "uz": "OpenVoiceOS/asr-uz-fastconformer-large-onnx",
    "tl": "OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx",
    "en": "nemo-parakeet-tdt-0.6b-v2",
    "uk": "OpenVoiceOS/stt_ua_fastconformer_hybrid_large_pc_onnx",
    "ar": "OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx",
    "hy": "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx",
    "ka": "OpenVoiceOS/stt_ka_fastconformer_hybrid_large_pc_onnx",
    "kk": "OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx",
    "pl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx",
    # only dedicated exports for these languages are Citrinet; still better
    # than the whisper-base filler
    "ko": "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx",
    "zh": "OpenVoiceOS/stt_zh_citrinet_1024_gamma_0_25_onnx",
    # QuartzNet15x5: dated architecture, but the only dedicated exports for
    # these languages (Carlos Daniel Hernandez Mena's Ravnursson/Samromur/MASRI
    # trained checkpoints); beats the whisper-base filler
    "fo": "OpenVoiceOS/carlosdanielhernandezmena-stt_fo_quartznet15x5_sp_ep163_100h_onnx",
    "is": "OpenVoiceOS/carlosdanielhernandezmena-stt_is_quartznet15x5_ft_ep56_875h_onnx",
    "mt": "OpenVoiceOS/carlosdanielhernandezmena-stt_mt_quartznet15x5_sp_ep255_64h_onnx",
})

# Indic: AI4Bharat IndicConformer per-language exports beat the Vaani multi model.
# "as" (Assamese) is deliberately absent — its export repo holds no ONNX weights
# yet, so Assamese stays on whisper until one is published.
for _l in ("bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd "
           "ta te ur").split():
    LANG_DEFAULTS[_l] = f"OpenVoiceOS/ai4bharat-indicconformer-{_l}-onnx"

# African & long-tail community fine-tunes (misterkissi exports)
LANG_DEFAULTS.update({
    "ht": "OpenVoiceOS/misterkissi-whisper-small-haitian-creole-onnx",
    "mg": "OpenVoiceOS/misterkissi-whisper-small-malagasy-onnx",
    "sn": "OpenVoiceOS/misterkissi-whisper-small-shona-onnx",
    "st": "OpenVoiceOS/misterkissi-whisper-small-sesotho-onnx",
    "tn": "OpenVoiceOS/misterkissi-whisper-small-setswana-onnx",
    "umb": "OpenVoiceOS/misterkissi-whisper-small-umbundu-onnx",
    "tig": "OpenVoiceOS/misterkissi-whisper-small-tigre-onnx",
    "fon": "OpenVoiceOS/misterkissi-whisper-small-fongbe-onnx",
    "vai": "OpenVoiceOS/misterkissi-whisper-small-vai-onnx",
    "ga": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx",
    "ln": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-lingala-onnx",
    "zu": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-zulu-onnx",
    "xh": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-xhosa-onnx",
    "ts": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-tsonga-onnx",
    "nyn": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-runyankore-onnx",
    "fat": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-fante-onnx",
    "xsm": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-kasem-onnx",
})

ENV_PREFIX = "ONNX_ASR_DEFAULT_"


def env_lang_defaults() -> Dict[str, str]:
    """
    Read per-language default models from ``ONNX_ASR_DEFAULT_<LANG>`` env vars.

    The suffix is a lang tag with underscores in place of dashes, so
    ``ONNX_ASR_DEFAULT_PT`` sets ``pt`` and ``ONNX_ASR_DEFAULT_PT_BR`` sets
    ``pt-br`` (full BCP-47 tags win over primary subtags at lookup time).
    """
    langs = {}
    for key, val in os.environ.items():
        if key.startswith(ENV_PREFIX) and val:
            tag = key[len(ENV_PREFIX):].replace("_", "-").lower()
            if tag:
                langs[tag] = val
    return langs


def resolve_model(lang: str,
                  lang2model: Dict[str, str],
                  default_model: Optional[str] = None) -> Optional[str]:
    """
    Pick the model for ``lang`` (a BCP-47 tag, any case).

    Tries the full tag then the primary subtag at each level:
    ``lang2model`` (plugin config) > env vars > built-in registry;
    falls back to ``default_model``.
    """
    full = lang.lower()
    primary = full.split("-")[0]
    env = env_lang_defaults()
    for table in (lang2model, env, LANG_DEFAULTS):
        for tag in (full, primary):
            if tag in table:
                return table[tag]
    return default_model
