from typing import Optional

import onnx_asr
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData
from ovos_utils import classproperty


class OnnxASR(STT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_id = self.config.get("model", "nemo-canary-1b-v2")
        self.onnx_model = onnx_asr.load_model(model_id)

    @classproperty
    def available_languages(cls) -> set:
        return set()

    def execute(self, audio: AudioData, language: Optional[str] = None):
        """
        Transcribes the provided audio using the configured model and language.

        Parameters:
            audio (AudioData): Audio input to be processed.
            language (Optional[str]): Language code to use for transcription; if omitted, the instance's current language is used.

        Returns:
            transcription (str): Final recognized text for the processed audio.
        """
        text = self.onnx_model.recognize(
            audio.get_np_float32(),
            sample_rate=audio.sample_rate,
            language=language or self.lang,
            target_language=language or self.lang
        )
        return text

if __name__ == "__main__":
    b = OnnxASR({"lang": "en", "model": "nemo-canary-1b-v2"})

    eu = "/home/miro/PycharmProjects/ovos-stt-plugin-vosk/jfk.wav"
    audio = AudioData.from_file(eu)

    a = b.execute(audio, language="en")
    print(a)
