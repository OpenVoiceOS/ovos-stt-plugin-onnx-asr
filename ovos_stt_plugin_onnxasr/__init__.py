from collections import OrderedDict
from threading import RLock
from typing import List, Optional

import onnx_asr
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData

from ovos_utils.log import LOG

from ovos_stt_plugin_onnxasr._compat import ensure_model_types
from ovos_stt_plugin_onnxasr.defaults import LANG_DEFAULTS, env_lang_defaults, resolve_model


class OnnxASR(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the model types the plugin carries with onnx-asr.
        ensure_model_types()
        self.default_model_id = self.config.get("model", "nemo-canary-1b-v2")
        # Optional per-language routing: {"lang2model": {"ru": "gigaam-v2-rnnt", ...}}.
        # Keys are BCP-47 tags (full tags like "pt-br" or primary subtags like
        # "pt"; full tags win). Anything not configured here falls back to
        # ONNX_ASR_DEFAULT_<LANG> env vars, then the built-in best-model-per-
        # language registry (defaults.LANG_DEFAULTS), then ``model``. Models
        # load lazily on first request for their language and stay cached, so
        # one server instance can serve every language.
        self.lang2model = {
            k.lower(): v
            for k, v in (self.config.get("lang2model") or {}).items()
        }
        self._models = OrderedDict()
        # Model ids that failed to load. Kept so a broken or unreachable model
        # is not retried on every request for its language.
        self._failed_models = set()
        # Caps how many onnx-asr models stay resident at once. Each model is
        # multi-GB, and one gets loaded per language actually requested, so
        # an unbounded cache on a multi-language deployment grows without
        # limit. Unset (the default) keeps every model loaded forever.
        # When set, the least-recently-used model is evicted the moment a new
        # one would exceed the cap. This bounds what the cache retains, not
        # peak memory: a model being transcribed through is held by the thread
        # using it, so eviction cannot reach it.
        self.max_loaded_models = self._parse_max_loaded_models(
            self.config.get("max_loaded_models"))
        # The server answers requests from a threadpool; the cache is mutated
        # from every one of those threads.
        self._lock = RLock()
        # the default model keeps loading eagerly so misconfiguration still
        # fails at startup rather than on the first request
        self.get_model(self.default_model_id)

    @staticmethod
    def _parse_max_loaded_models(value) -> Optional[int]:
        """
        Normalize the ``max_loaded_models`` config value to a positive int.

        Configuration reaches plugins from hand-edited files and from
        environment injection, so a value can arrive as a string. A value that
        cannot bound a cache -- unparseable, or below 1 -- is reported and
        treated as unset, because refusing to start would take down a server
        over a tuning knob.
        """
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            LOG.error(f"ignoring invalid max_loaded_models: {value!r}")
            return None
        if parsed < 1:
            LOG.error(f"ignoring max_loaded_models={parsed}, it must be at "
                      f"least 1; the cache is unbounded")
            return None
        return parsed

    def get_model(self, model_id: str):
        """
        Load (or fetch from cache) the onnx-asr model for ``model_id``.

        Returns:
            tuple: (model, accepts_language, accepts_target_language)

        The server answers requests from a threadpool, so the cache is guarded
        by a lock and the loaded entry is returned directly. Reading it back
        out of the cache would race with another thread evicting it.
        """
        with self._lock:
            if model_id in self._models:
                self._models.move_to_end(model_id)
                return self._models[model_id]
            # Evict down to one free slot BEFORE loading. Loading first would
            # put cap + 1 models in memory at the moment memory is tightest,
            # which is the OOM this cap exists to prevent.
            self._evict_lru(headroom=1)

        quantization = self.config.get("quantization")
        providers = self._get_providers()
        if providers:
            LOG.info(f"onnx-asr using providers: {providers}")
        LOG.info(f"loading onnx-asr model: {model_id}")
        model = onnx_asr.load_model(model_id,
                                    quantization=quantization,
                                    providers=providers)
        # onnx-asr accepts a `language` hint only for Whisper and Canary
        # models, and `target_language` only for Canary. Passing them to
        # other architectures is outside the RecognizeOptions contract,
        # so gate them on the loaded model family.
        asr_name = type(model.asr).__name__
        accepts_language = "Whisper" in asr_name or asr_name == "NemoConformerAED"
        accepts_target_language = asr_name == "NemoConformerAED"
        entry = (model, accepts_language, accepts_target_language)

        with self._lock:
            self._models[model_id] = entry
            self._models.move_to_end(model_id)
            self._evict_lru()
        return entry

    def _evict_lru(self, headroom: int = 0):
        """
        Drop least-recently-used models until the cache holds at most
        ``max_loaded_models - headroom``. No-op when the cap is unset.

        Call with the lock held.
        """
        if not self.max_loaded_models:
            return
        limit = max(self.max_loaded_models - headroom, 0)
        while len(self._models) > limit:
            evicted_id, _ = self._models.popitem(last=False)
            LOG.info(f"evicting onnx-asr model (max_loaded_models={self.max_loaded_models}): {evicted_id}")

    def _load_with_fallback(self, model_id: str, tag: str):
        """
        Load ``model_id``, falling back to the configured default model when
        it cannot be loaded.

        A per-language model can be unavailable for reasons that have nothing
        to do with the request: the download failed, the id is wrong, the host
        is offline. Refusing the request in that case loses a language the
        server can still serve, so an unavailable model degrades to the
        default one instead. The substitution is logged at warning level, so
        it is visible rather than silent.

        A failing model is remembered and not retried, otherwise every request
        for that language pays the failure again. The default model itself has
        nothing to fall back to, so its failure is raised.
        """
        candidates = [model_id]
        if self.default_model_id not in candidates:
            candidates.append(self.default_model_id)

        for candidate in candidates:
            if candidate in self._failed_models:
                continue
            try:
                return self.get_model(candidate)
            except Exception as err:
                self._failed_models.add(candidate)
                LOG.error(f"failed to load onnx-asr model '{candidate}' "
                          f"for language '{tag}': {err}")

        raise RuntimeError(
            f"no usable onnx-asr model for language '{tag}': "
            f"tried {candidates}"
        )

    def _get_providers(self) -> Optional[List[str]]:
        """
        Resolve the onnxruntime execution providers from plugin config.

        ``providers`` (an explicit list of onnxruntime provider names) takes
        precedence; otherwise ``use_cuda: true`` selects CUDA with a CPU
        fallback. When neither is set, returns None so onnx-asr/onnxruntime
        pick their default (CPU).

        GPU execution requires ``onnxruntime-gpu`` installed in place of the
        default ``onnxruntime`` (plus a matching CUDA/cuDNN runtime).

        Returns:
            list[str] | None: Ordered provider list, or None for the default.
        """
        providers = self.config.get("providers")
        if providers:
            return list(providers)
        if self.config.get("use_cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return None

    @property
    def available_languages(self) -> set:
        langs = set(LANG_DEFAULTS) | set(env_lang_defaults()) | set(self.lang2model)
        return langs

    def execute(self, audio: AudioData, language: Optional[str] = None):
        """
        Transcribes the provided audio using the configured model and language.

        Parameters:
            audio (AudioData): Audio input to be processed.
            language (Optional[str]): Language code to use for transcription; if omitted, the instance's current language is used.

        Returns:
            transcription (str): Final recognized text for the processed audio.
        """
        tag = (language or self.lang).lower()
        model_id = resolve_model(tag, self.lang2model, self.default_model_id)
        # onnx-asr models use bare ISO 639-1 codes ("en"); OVOS hands us full
        # BCP-47 tags ("en-US"), which raise KeyError inside the decoders.
        lang = tag.split("-")[0]
        model, accepts_language, accepts_target_language = self._load_with_fallback(model_id, tag)
        kwargs = {}
        if accepts_language:
            kwargs["language"] = lang
        if accepts_target_language:
            kwargs["target_language"] = lang
        text = model.recognize(
            audio.get_np_float32(),
            sample_rate=audio.sample_rate,
            **kwargs
        )
        return text

if __name__ == "__main__":
    b = OnnxASR({"lang": "en",
                 "model": "nemo-canary-1b-v2",
                 "quantization": "int8"})

    eu = "/home/miro/PycharmProjects/ovos-stt-plugin-vosk/jfk.wav"
    audio = AudioData.from_file(eu)

    a = b.execute(audio, language="en")
    print(a)
