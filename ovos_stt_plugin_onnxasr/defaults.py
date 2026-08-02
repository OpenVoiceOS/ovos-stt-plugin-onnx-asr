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
           "sw ur bn").split():
    LANG_DEFAULTS[_l] = _WHISPER

# dedicated fine-tunes — better than the multilingual fillers on their language
LANG_DEFAULTS.update({
    "ru": "gigaam-v2-rnnt",
    "pt": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx",
    "pl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx",
    "nl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-nl-onnx",
    "et": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-et-onnx",
    "sl": "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-sl-onnx",
    "es": "OpenVoiceOS/nvidia-es-conformer-transducer-large-onnx",
    "de": "OpenVoiceOS/nvidia-de-conformer-transducer-large-onnx",
    "fr": "OpenVoiceOS/nvidia-fr-conformer-transducer-large-onnx",
    "it": "OpenVoiceOS/nvidia-it-conformer-transducer-large-onnx",
    "hr": "OpenVoiceOS/nvidia-hr-conformer-transducer-large-onnx",
    "ca": "OpenVoiceOS/nvidia-ca-conformer-transducer-large-onnx",
    "eu": "OpenVoiceOS/hitz-eu-conformer-transducer-large-v2-onnx",
    "gl": "OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx",
    "be": "OpenVoiceOS/nvidia-be-conformer-transducer-large-onnx",
    "eo": "OpenVoiceOS/nvidia-eo-conformer-transducer-large-onnx",
    "rw": "OpenVoiceOS/nvidia-rw-conformer-transducer-large-onnx",
    "kab": "OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx",
    "fa": "OpenVoiceOS/nvidia-fa-fastconformer-hybrid-large-onnx",
    "ja": "OpenVoiceOS/nvidia-parakeet-tdt_ctc-0.6b-ja-onnx",
    "vi": "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
    "uz": "OpenVoiceOS/asr-uz-fastconformer-large-onnx",
    "tl": "OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx",
})

# Indic: AI4Bharat IndicConformer per-language exports beat the Vaani multi model
for _l in ("as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd "
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
