from typing import List, Optional

import onnx_asr
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData

from ovos_utils.log import LOG

from ovos_stt_plugin_onnxasr.defaults import LANG_DEFAULTS, env_lang_defaults, resolve_model


class OnnxASR(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self._models = {}
        # the default model keeps loading eagerly so misconfiguration still
        # fails at startup rather than on the first request
        self.get_model(self.default_model_id)

    def get_model(self, model_id: str):
        """
        Load (or fetch from cache) the onnx-asr model for ``model_id``.

        Returns:
            tuple: (model, accepts_language, accepts_target_language)
        """
        if model_id not in self._models:
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
            self._models[model_id] = (model, accepts_language, accepts_target_language)
        return self._models[model_id]

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
        model, accepts_language, accepts_target_language = self.get_model(model_id)
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
