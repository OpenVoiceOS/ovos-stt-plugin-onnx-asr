import unittest
from unittest.mock import MagicMock, patch, call

from ovos_stt_plugin_onnxasr.defaults import resolve_model
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
        "pt-PT": "parakeet-pt",
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

    def test_lang2model_full_tag_resolves_however_it_is_written(self):
        # The map is kept as the operator wrote it; matching standardizes both
        # sides, so the spelling of a key never decides whether it applies.
        stt = self._stt(MagicMock(return_value=_adapter()))
        for tag in ("pt-PT", "pt_PT", "PT-pt"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    resolve_model(tag, stt.lang2model, "default-model"),
                    "parakeet-pt")

    def test_unmapped_lang_falls_back_to_default(self):
        models = {}
        def load(model_id, **kw):
            models[model_id] = _adapter()
            return models[model_id]
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR(dict(CFG))
            stt.execute(_audio(), language="tlh")
        self.assertEqual(list(models), ["default-model"])
        models["default-model"].recognize.assert_called_once()

    def test_models_cached_one_load_per_model(self):
        load = MagicMock(side_effect=lambda *a, **kw: _adapter())
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR(dict(CFG))
            stt.execute(_audio(), language="ru")
            stt.execute(_audio(), language="ru")
            stt.execute(_audio(), language="tlh")
        # default (eager) + gigaam = 2 loads total; tlh reuses the default
        self.assertEqual(load.call_count, 2)

    def test_available_languages(self):
        stt = self._stt(MagicMock(return_value=_adapter()))
        langs = stt.available_languages
        # config keys plus the built-in registry
        self.assertIn("ru", langs)
        self.assertIn("pt-PT", langs)  # canonical BCP-47, not as written
        self.assertIn("gl", langs)  # from LANG_DEFAULTS
