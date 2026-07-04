import unittest
from unittest.mock import patch, MagicMock

import numpy as np


def _fake_adapter(asr_class_name, captured):
    """Build a stand-in onnx-asr adapter whose ``asr`` attribute is an instance
    of a class named ``asr_class_name`` and whose ``recognize`` records the
    keyword arguments it was called with."""
    adapter = MagicMock()
    adapter.asr = type(asr_class_name, (), {})()

    def recognize(waveform, sample_rate, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return "transcript"

    adapter.recognize = recognize
    return adapter


def _audio():
    audio = MagicMock()
    audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
    audio.sample_rate = 16000
    return audio


class TestLanguageKwargs(unittest.TestCase):
    def _run(self, asr_class_name):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter(asr_class_name, captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "dummy"})
            stt.execute(_audio(), language="es")
        return captured

    def test_whisper_gets_language_only(self):
        captured = self._run("WhisperOrt")
        self.assertEqual(captured, {"language": "es"})

    def test_canary_gets_language_and_target(self):
        captured = self._run("NemoConformerAED")
        self.assertEqual(captured, {"language": "es", "target_language": "es"})

    def test_nemo_ctc_gets_no_language_kwargs(self):
        captured = self._run("NemoConformerCtc")
        self.assertEqual(captured, {})

    def test_transducer_gets_no_language_kwargs(self):
        captured = self._run("NemoConformerRnnt")
        self.assertEqual(captured, {})

    def test_falls_back_to_instance_lang(self):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter("WhisperHf", captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "de", "model": "dummy"})
            stt.execute(_audio())
        self.assertEqual(captured, {"language": "de"})


if __name__ == "__main__":
    unittest.main()
