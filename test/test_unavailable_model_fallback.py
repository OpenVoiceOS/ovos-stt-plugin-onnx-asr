import unittest
from unittest.mock import MagicMock

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
    "lang2model": {"ru": "broken-model"},
}


class TestUnavailableModelFallback(unittest.TestCase):
    def _stt(self, cfg, broken=("broken-model",)):
        def _load(model_id, *a, **kw):
            if model_id in broken:
                raise RuntimeError("download failed")
            return _adapter()

        load = MagicMock(side_effect=_load)
        import onnx_asr
        previous = onnx_asr.load_model
        onnx_asr.load_model = load
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        from ovos_stt_plugin_onnxasr import OnnxASR
        return OnnxASR(dict(cfg)), load

    def test_unavailable_language_model_falls_back_to_default(self):
        stt, _ = self._stt(CFG)
        # ru routes to a model that cannot be loaded; the language must still
        # be served rather than the request failing.
        self.assertEqual(stt.execute(_audio(), language="ru"), "transcript")
        self.assertIn("default-model", stt._models)
        self.assertNotIn("broken-model", stt._models)

    def test_failed_model_is_not_retried(self):
        stt, load = self._stt(CFG)
        stt.execute(_audio(), language="ru")
        calls_after_first = load.call_count
        stt.execute(_audio(), language="ru")
        # the broken model must not be attempted a second time
        self.assertEqual(load.call_count, calls_after_first)

    def test_default_model_failure_is_raised(self):
        stt, _ = self._stt(CFG, broken=("broken-model", "other"))
        stt._failed_models.add("default-model")
        with self.assertRaises(RuntimeError):
            stt.execute(_audio(), language="ru")


if __name__ == "__main__":
    unittest.main()
