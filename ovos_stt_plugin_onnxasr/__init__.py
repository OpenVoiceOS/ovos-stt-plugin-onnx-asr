from typing import List, Optional

import onnx_asr
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData
from ovos_utils import classproperty
from ovos_utils.log import LOG

from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc

# Best offline model per language when the config does not name one.
# Heavier languages use NVIDIA FastConformer-hybrid large pc exports (CTC head,
# punctuation + casing); the rest use light Citrinet-512 exports. All verified
# against the original NeMo checkpoints (https://huggingface.co/OpenVoiceOS).
LANG_MODELS = {
    "ar": "OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pcd_v1.0_onnx",
    "be": "OpenVoiceOS/stt_be_fastconformer_hybrid_large_pc_onnx",
    "ca": "OpenVoiceOS/stt_ca_citrinet_512_gamma_0_25_onnx",
    "de": "OpenVoiceOS/stt_de_fastconformer_hybrid_large_pc_onnx",
    "en": "OpenVoiceOS/stt_en_fastconformer_hybrid_large_pc_onnx",
    "es": "OpenVoiceOS/stt_es_fastconformer_hybrid_large_pc_onnx",
    "fa": "OpenVoiceOS/stt_fa_fastconformer_hybrid_large_onnx",
    "fr": "OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx",
    "hr": "OpenVoiceOS/stt_hr_fastconformer_hybrid_large_pc_onnx",
    "hy": "OpenVoiceOS/stt_hy_fastconformer_hybrid_large_pc_onnx",
    "it": "OpenVoiceOS/stt_it_citrinet_512_gamma_0_25_onnx",
    "ka": "OpenVoiceOS/stt_ka_fastconformer_hybrid_large_pc_onnx",
    "kk": "OpenVoiceOS/stt_kk_ru_fastconformer_hybrid_large_onnx",
    "ko": "OpenVoiceOS/stt_kr_citrinet1024_PublicCallCenter_1000H_onnx",
    "nl": "OpenVoiceOS/stt_nl_citrinet_512_gamma_0_25_onnx",
    "pl": "OpenVoiceOS/stt_pl_fastconformer_hybrid_large_pc_onnx",
    "pt": "OpenVoiceOS/stt_pt_citrinet_512_gamma_0_25_onnx",
    "ru": "nemo-fastconformer-ru-ctc",
    "uk": "OpenVoiceOS/stt_ua_fastconformer_hybrid_large_pc_onnx",
    "uz": "OpenVoiceOS/stt_uz_fastconformer_hybrid_large_pc_onnx",
    "zh": "OpenVoiceOS/stt_zh_citrinet_1024_gamma_0_25_onnx",
}


class OnnxASR(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Teach onnx-asr about the wav2vec2-ctc type until upstream PR #1 ships it.
        ensure_wav2vec2_ctc()
        lang = (self.lang or "").split("-")[0]
        model_id = self.config.get("model") or LANG_MODELS.get(lang, "nemo-canary-1b-v2")
        quantization = self.config.get("quantization")
        providers = self._get_providers()
        if providers:
            LOG.info(f"onnx-asr using providers: {providers}")
        self.onnx_model = onnx_asr.load_model(model_id,
                                              quantization=quantization,
                                              providers=providers)
        # onnx-asr accepts a `language` hint only for Whisper and Canary models,
        # and `target_language` only for Canary. Passing them to other
        # architectures is outside the RecognizeOptions contract, so gate them
        # on the loaded model family.
        asr_name = type(self.onnx_model.asr).__name__
        self._accepts_language = "Whisper" in asr_name or asr_name == "NemoConformerAED"
        self._accepts_target_language = asr_name == "NemoConformerAED"

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

    @classproperty
    def available_languages(cls) -> set:
        return set(LANG_MODELS)

    def execute(self, audio: AudioData, language: Optional[str] = None):
        """
        Transcribes the provided audio using the configured model and language.

        Parameters:
            audio (AudioData): Audio input to be processed.
            language (Optional[str]): Language code to use for transcription; if omitted, the instance's current language is used.

        Returns:
            transcription (str): Final recognized text for the processed audio.
        """
        lang = language or self.lang
        kwargs = {}
        if self._accepts_language:
            kwargs["language"] = lang
        if self._accepts_target_language:
            kwargs["target_language"] = lang
        text = self.onnx_model.recognize(
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
