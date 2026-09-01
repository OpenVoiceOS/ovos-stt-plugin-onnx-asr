"""Built-in best-model-per-language registry.

Maps BCP-47 tags to the best onnx-asr
compatible model we know of for that language — dedicated fine-tunes first,
multilingual models as coverage fillers. Most exports live in the
`OpenVoiceOS/stt-asr-onnx` HuggingFace collection:
https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx

Resolution order (see ``OnnxASR``), from strongest to weakest:

1. ``lang2model`` in the plugin config — the operator names a model for a language.
2. ``ONNX_ASR_DEFAULT_<LANG>`` environment variables — the same choice, made at
   deployment level instead of in the config file.
3. ``model`` in the plugin config — the operator names one model for every
   language. It is weaker than the two per-language layers and stronger than this
   registry, because a registry entry is a guess and ``model`` is an instruction.
4. this registry — the best model known for a language the operator said nothing
   about.
5. the built-in fallback model, for a language this registry does not hold.

Full tags win over primary subtags at every level, so ``pt-br`` can point somewhere
other than ``pt``. Tag comparison follows OVOS-INTENT-2 §2 via ``ovos-spec-tools``,
so it is case insensitive, accepts underscores, and falls back to the nearest
usable tag rather than to a shared prefix.
"""
import os
import re
from typing import Dict, Optional

from ovos_spec_tools.language import closest_lang, standardize_lang

DEFAULT_MODEL = "nemo-canary-1b-v2"
"""Model to serve a language that nothing else resolves."""

DEFAULT_CPU_MODEL = "whisper-base"
"""Model to serve a language that nothing else resolves, under ``cpu_models_only``.

:data:`DEFAULT_MODEL` names its own parameter count (1B) in its id, which
:func:`is_cpu_friendly` reads as too large for a CPU-only deployment. This one
is the plugin's other multilingual coverage filler (99 languages, 74M
parameters), so a CPU-only deployment that names no ``model`` still gets a
sensible default rather than one the flag it set would reject."""

CPU_MODEL_PARAM_LIMIT = 0.6
"""Smallest parameter count, in billions, that a model's name may advertise
before :func:`is_cpu_friendly` calls it unsuitable for CPU-only inference.

OVOS deployments routinely run STT on satellite-class hardware (a Raspberry Pi
or similar), where a model at or above this size is impractically slow or
exhausts memory. The registry carries no dedicated size field per model, so
this reads the parameter count the catalogue already writes into every model's
own id (``nemo-canary-1b-v2``, ``qwen3-asr-0.6b-onnx``) rather than keeping a
second, hand-maintained list that could drift from what the ids say."""

_SIZE_RE = re.compile(r"(?:^|[-_])(\d+(?:\.\d+)?)b(?:[-_]|$)", re.IGNORECASE)


def model_param_count(model_id: str) -> Optional[float]:
    """
    Give the parameter count, in billions, that ``model_id`` advertises in its
    own name, or None when the id carries no such marker.

    A model without a size marker in its name is not assumed to be large.
    """
    match = _SIZE_RE.search(model_id)
    return float(match.group(1)) if match else None


def is_cpu_friendly(model_id: str) -> bool:
    """
    True unless ``model_id``'s name advertises a parameter count at or above
    :data:`CPU_MODEL_PARAM_LIMIT`.
    """
    count = model_param_count(model_id)
    return count is None or count < CPU_MODEL_PARAM_LIMIT


def cpu_friendly_only(table: Dict[str, str]) -> Dict[str, str]:
    """Return ``table`` without the languages a CPU-unfriendly model serves."""
    return {lang: model for lang, model in table.items()
            if is_cpu_friendly(model)}

KNOWN_BAD_MODELS: Dict[str, str] = {}
"""Model id -> why the model must never be a default.

A model that runs but returns text nobody can read is worse than no model: the
request succeeds, so nothing reports a fault. An entry here is dropped from
:data:`LANG_DEFAULTS`, which leaves the language to the operator's own
configuration. Keep one line of evidence per entry.
"""

FP32_ONLY_MODELS = frozenset({
    "OpenVoiceOS/asr-uz-fastconformer-large-onnx",
    "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-fante-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-kasem-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-lingala-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-runyankore-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-tsonga-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-xhosa-onnx",
    "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-zulu-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-fongbe-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-haitian-creole-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-malagasy-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-sesotho-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-setswana-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-shona-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-tigre-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-umbundu-onnx",
    "OpenVoiceOS/misterkissi-whisper-small-vai-onnx",
    "OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx",
    "OpenVoiceOS/stt-tl-fastconformer-hybrid-large-onnx",
    "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx",
    "OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx",
})
"""Registry models whose repository holds fp32 weights only.

``quantization: "int8"`` asks for files these repositories do not hold, and the
load fails. The operator sets one quantization for the whole plugin, but this
registry picks the model, so that setting must not take a language away.
:func:`quantization_for` reads this set and loads fp32 for these models.

The set holds registry models only, and a test keeps it equal to what the
repositories really hold."""

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
#
# pt/pt-PT alternatives (not defaults, need the TigreGotico/onnx-asr `integration`
# fork branch until espnet-ctc/espnet-aed upstreams — see README "Fork model
# families" section): set via `lang2model` or `ONNX_ASR_DEFAULT_PT`/`_PT_PT`.
#   OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx   (espnet-ctc, fast)
#   OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx   (espnet-aed, more accurate)
#   OpenVoiceOS/inesc-id-whisperlv3-ft-ep-onnx          (Whisper, needs no fork)
#   OpenVoiceOS/camoes-whisper-asr-onnx                 (Whisper, needs no fork)
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
    # Paraformer is a dedicated Chinese model; SenseVoice covers Chinese,
    # Cantonese, Japanese, Korean and English in one model. Both beat the
    # Citrinet exports, which are the older architecture and hold fp32 weights
    # only.
    "zh": "OpenVoiceOS/paraformer-zh-onnx",
    # langcodes reads an explicit Hant script as a large distance from plain
    # "zh", which it resolves as Hans, so Traditional tags do not reach the
    # entry above on nearest-tag matching alone. The model hears the same
    # speech either way and writes Simplified, so name the tags instead of
    # loosening the distance every language is matched with.
    "zh-Hant": "OpenVoiceOS/paraformer-zh-onnx",
    # SenseVoice covers Korean and is the stronger model, but its export
    # currently rejects a third of all audio lengths, so Korean stays on the
    # Citrinet export until that is resolved.
    "ko": "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx",
})

# Indic: AI4Bharat IndicConformer per-language exports beat the Vaani multi model.
# "as" (Assamese) is deliberately absent — its export repo holds no ONNX weights
# yet, so Assamese stays on whisper until one is published.
for _l in ("bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd "
           "ta te ur").split():
    LANG_DEFAULTS[_l] = f"OpenVoiceOS/ai4bharat-indicconformer-{_l}-onnx"

# African & long-tail community fine-tunes (misterkissi exports).
# The w2v2 group below shares one source: XLS-R 300m fine-tunes by the same
# author. They are accurate on speech that resembles their training data and
# degrade sharply away from it: the Zulu member scores 12.5% WER on its own test
# set and 91% on FLEURS. Treat the reported WER of any member as an in-domain
# figure, and read a transcription of your own audio before relying on one.
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
    # "ga" in this model id is Ga, the Kwa language of Accra (ISO 639-3
    # "gaa"), not Irish. The two-letter tag "ga" is Irish, and this model
    # transcribes Irish at 99.7% WER while reaching 4.96% WER on Ga.
    "gaa": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx",
    "ln": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-lingala-onnx",
    "zu": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-zulu-onnx",
    "xh": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-xhosa-onnx",
    "ts": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-tsonga-onnx",
    "nyn": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-runyankore-onnx",
    "fat": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-fante-onnx",
    "xsm": "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-kasem-onnx",
})

def drop_known_bad(table: Dict[str, str]) -> Dict[str, str]:
    """
    Return ``table`` without the languages that a known-bad model serves.

    See :data:`KNOWN_BAD_MODELS`.
    """
    return {lang: model for lang, model in table.items()
            if model not in KNOWN_BAD_MODELS}


# Applied at the end, so that adding an id to KNOWN_BAD_MODELS is all it takes
# to keep the model out of every language it serves.
LANG_DEFAULTS = drop_known_bad(LANG_DEFAULTS)

ENV_PREFIX = "ONNX_ASR_DEFAULT_"


def quantization_for(model_id: str, quantization: Optional[str]) -> Optional[str]:
    """
    Give the quantization to load ``model_id`` with.

    Returns ``quantization``, except for a registry model whose repository holds
    fp32 weights only: that one loads fp32. See :data:`FP32_ONLY_MODELS`.
    """
    if quantization and model_id in FP32_ONLY_MODELS:
        return None
    return quantization


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
            tag = key[len(ENV_PREFIX):]
            if tag:
                langs[standardize_lang(tag)] = val
    return langs


def _match(tag: str, table: Dict[str, str]) -> Optional[str]:
    """
    Give the key of ``table`` that serves ``tag``, or None.

    ``tag`` must already be standardized. Table keys are standardized here, so a
    config written as ``{"pt_BR": ...}`` matches a request for ``pt-BR``.
    """
    keys = {standardize_lang(k): k for k in table}
    if tag in keys:
        return keys[tag]
    nearest = closest_lang(tag, list(keys))
    return keys[nearest] if nearest else None


def resolve_model(lang: str,
                  lang2model: Dict[str, str],
                  default_model: Optional[str] = None,
                  configured_model: Optional[str] = None,
                  registry: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Pick the model for ``lang`` (a BCP-47 tag, any case, hyphens or underscores).

    Matches the full tag then the nearest usable tag at each per-language level:
    ``lang2model`` (plugin config) > ``ONNX_ASR_DEFAULT_<LANG>`` env vars >
    ``configured_model`` > the built-in registry > ``default_model``.

    Tag comparison follows OVOS-INTENT-2 §2: it is case insensitive, treats
    underscores as hyphens, and falls back to the nearest tag within the
    ``langcodes`` distance the spec calls usable. So ``en_US`` reaches an ``en``
    entry, and a tag no entry is close to reaches ``default_model`` rather than
    an unrelated language that happens to share a prefix.

    Args:
        lang: the language to serve.
        lang2model: the ``lang2model`` map from the plugin config.
        default_model: model for a language nothing else resolves.
        configured_model: the ``model`` the operator wrote in the config, if any.
            It names one model for every language, so it wins over the registry:
            an operator who asks for a model gets that model. Leave it out to let
            the registry pick per language.
        registry: the built-in per-language table to fall back to. Defaults to
            :data:`LANG_DEFAULTS`; a caller filtering the registry (e.g. for
            ``cpu_models_only``) passes the filtered table here instead.
    """
    table = LANG_DEFAULTS if registry is None else registry
    tag = standardize_lang(lang)
    env = env_lang_defaults()
    for t in (lang2model, env):
        match = _match(tag, t)
        if match:
            return t[match]
    if configured_model:
        return configured_model
    match = _match(tag, table)
    if match:
        return table[match]
    return default_model
