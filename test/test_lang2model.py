import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np


def _adapter(asr_class_name="NemoConformerTDT"):
    adapter = MagicMock()
    adapter.asr = type(asr_class_name, (), {})()
    adapter.recognize = MagicMock(return_value="transcript")
    return adapter


def _audio():
    audio = MagicMock()
    audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
    audio.sample_rate = 16000
    return audio


CFG = {
    "lang": "en",
    "model": "default-model",
    "lang2model": {
        "ru": "gigaam-v2-rnnt",
        "PT-PT": "parakeet-pt",
    },
}


class TestLang2Model(unittest.TestCase):
    def _stt(self, load_model):
        with patch("onnx_asr.load_model", load_model):
            from ovos_stt_plugin_onnxasr import OnnxASR
            return OnnxASR(dict(CFG))

    def test_default_model_loads_eagerly(self):
        load = MagicMock(return_value=_adapter())
        self._stt(load)
        load.assert_called_once()
        self.assertEqual(load.call_args[0][0], "default-model")

    def test_lang_routes_to_mapped_model(self):
        models = {}
        def load(model_id, **kw):
            models[model_id] = _adapter()
            return models[model_id]
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR(dict(CFG))
            stt.execute(_audio(), language="ru-RU")
        self.assertIn("gigaam-v2-rnnt", models)
        models["gigaam-v2-rnnt"].recognize.assert_called_once()
        models["default-model"].recognize.assert_not_called()

    def test_lang2model_keys_normalized(self):
        stt = self._stt(MagicMock(return_value=_adapter()))
        self.assertEqual(stt.lang2model["pt"], "parakeet-pt")

    def test_unmapped_lang_falls_back_to_default(self):
        models = {}
        def load(model_id, **kw):
            models[model_id] = _adapter()
            return models[model_id]
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR(dict(CFG))
            stt.execute(_audio(), language="sw")
        self.assertEqual(list(models), ["default-model"])
        models["default-model"].recognize.assert_called_once()

    def test_models_cached_one_load_per_model(self):
        load = MagicMock(side_effect=lambda *a, **kw: _adapter())
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR(dict(CFG))
            stt.execute(_audio(), language="ru")
            stt.execute(_audio(), language="ru")
            stt.execute(_audio(), language="en")
        # default (eager) + gigaam = 2 loads total, en reuses default
        self.assertEqual(load.call_count, 2)

    def test_available_languages(self):
        stt = self._stt(MagicMock(return_value=_adapter()))
        self.assertEqual(stt.available_languages, {"ru", "pt"})
