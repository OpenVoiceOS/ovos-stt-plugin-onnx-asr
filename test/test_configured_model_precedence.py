"""The model in the config wins over the built-in registry.

A configured ``model`` is an instruction from the operator. The registry is a
guess about a language the operator said nothing about. A guess that overrides an
instruction sends requests to a model nobody asked for, and leaves the model the
operator asked for loaded and unused.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ovos_stt_plugin_onnxasr.defaults import LANG_DEFAULTS, resolve_model


def _fake_loader(loaded):
    def load(model_id, **kwargs):
        model = MagicMock()
        model.asr = type("NemoConformerCtc", (), {})()
        model.recognize = MagicMock(return_value="transcript")
        loaded.append(model_id)
        return model
    return load


def _audio():
    audio = MagicMock()
    audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
    audio.sample_rate = 16000
    return audio


class TestConfiguredModelWins(unittest.TestCase):
    def test_execute_uses_the_configured_model(self):
        loaded = []
        with patch("onnx_asr.load_model", _fake_loader(loaded)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en",
                           "model": "OpenVoiceOS/moonshine-tiny-onnx"})
            self.assertEqual(stt.execute(_audio(), language="en"), "transcript")
        self.assertEqual(loaded, ["OpenVoiceOS/moonshine-tiny-onnx"])

    def test_configured_model_serves_every_language(self):
        loaded = []
        with patch("onnx_asr.load_model", _fake_loader(loaded)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "my-model"})
            stt.execute(_audio(), language="gl-ES")
            stt.execute(_audio(), language="pt")
        self.assertEqual(loaded, ["my-model"])

    def test_registry_serves_a_plugin_with_no_configured_model(self):
        loaded = []
        with patch("onnx_asr.load_model", _fake_loader(loaded)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en"})
            stt.execute(_audio(), language="gl")
        self.assertIn(LANG_DEFAULTS["gl"], loaded)


class TestResolutionOrder(unittest.TestCase):
    def test_configured_model_beats_registry(self):
        self.assertEqual(resolve_model("gl", {}, "fallback",
                                       configured_model="chosen"), "chosen")

    def test_lang2model_beats_configured_model(self):
        self.assertEqual(resolve_model("gl", {"gl": "per-lang"}, "fallback",
                                       configured_model="chosen"), "per-lang")

    def test_env_beats_configured_model(self):
        with patch.dict("os.environ", {"ONNX_ASR_DEFAULT_GL": "env-model"}):
            self.assertEqual(resolve_model("gl", {}, "fallback",
                                           configured_model="chosen"),
                             "env-model")

    def test_registry_serves_an_unconfigured_language(self):
        self.assertEqual(resolve_model("gl", {}, "fallback"),
                         LANG_DEFAULTS["gl"])


if __name__ == "__main__":
    unittest.main()
