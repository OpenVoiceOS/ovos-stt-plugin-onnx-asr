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
    "lang2model": {
        "ru": "ru-model",
        "es": "es-model",
    },
    "max_loaded_models": 2,
}


class TestMaxLoadedModels(unittest.TestCase):
    def _stt(self, cfg):
        load = MagicMock(side_effect=lambda *a, **kw: _adapter())
        import onnx_asr
        # Assign and restore rather than ``mock.patch``: when the module is a
        # stub, patch removes the attribute on exit instead of putting the
        # previous value back, and every later test that stubs the same
        # attribute then fails.
        previous = onnx_asr.load_model
        onnx_asr.load_model = load
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        from ovos_stt_plugin_onnxasr import OnnxASR
        return OnnxASR(dict(cfg)), load

    def test_lru_eviction_bounds_loaded_models(self):
        stt, load = self._stt(CFG)
        # default-model is already loaded eagerly (1/2)
        stt.execute(_audio(), language="ru")   # ru-model loaded (2/2)
        self.assertEqual(set(stt._models), {"default-model", "ru-model"})

        # es-model would be a 3rd resident model; default-model is the LRU
        # entry (untouched since startup) and must be evicted to make room.
        stt.execute(_audio(), language="es")
        self.assertEqual(set(stt._models), {"ru-model", "es-model"})
        self.assertNotIn("default-model", stt._models)

        # requesting the evicted default model again must reload it (a cache
        # miss), proving eviction actually dropped the model rather than
        # just bookkeeping around it.
        # "tlh" (Klingon) is not in lang2model or the built-in registry, so
        # it resolves back to the configured default model.
        stt.execute(_audio(), language="tlh")
        self.assertEqual(set(stt._models), {"es-model", "default-model"})
        self.assertNotIn("ru-model", stt._models)
        self.assertEqual(load.call_count, 4)  # default, ru, es, default-again

    def test_unset_max_loaded_models_keeps_unbounded_cache(self):
        cfg = dict(CFG)
        del cfg["max_loaded_models"]
        stt, _ = self._stt(cfg)
        stt.execute(_audio(), language="ru")
        stt.execute(_audio(), language="es")
        self.assertEqual(set(stt._models), {"default-model", "ru-model", "es-model"})


if __name__ == "__main__":
    unittest.main()
