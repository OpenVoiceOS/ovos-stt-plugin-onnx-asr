import threading
import unittest
from unittest.mock import MagicMock

import numpy as np


def _adapter():
    adapter = MagicMock()
    adapter.asr = type("NemoConformerTDT", (), {})()
    adapter.recognize = MagicMock(return_value="transcript")
    return adapter


def _audio():
    audio = MagicMock()
    audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
    audio.sample_rate = 16000
    return audio


LANGS = ["en", "ru", "es", "de", "fr", "it"]

CFG = {
    "lang": "en",
    "model": "default-model",
    "lang2model": {lang: f"{lang}-model" for lang in LANGS},
    "max_loaded_models": 2,
}


class TestConcurrentAccess(unittest.TestCase):
    def _stt(self, cfg):
        load = MagicMock(side_effect=lambda *a, **kw: _adapter())
        import onnx_asr
        previous = onnx_asr.load_model
        onnx_asr.load_model = load
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        from ovos_stt_plugin_onnxasr import OnnxASR
        return OnnxASR(dict(cfg))

    def test_concurrent_requests_do_not_crash(self):
        stt = self._stt(CFG)
        errors = []

        def worker(lang):
            try:
                for _ in range(200):
                    stt.execute(_audio(), language=lang)
            except Exception as err:  # noqa: BLE001 - recorded for the assert
                errors.append(err)

        threads = [threading.Thread(target=worker, args=(lang,))
                   for lang in LANGS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [], f"concurrent execute() raised: {errors[:3]}")

    def test_cache_never_exceeds_cap_during_a_load(self):
        # eviction must happen before the new model is loaded, otherwise
        # cap + 1 models are resident at the worst possible moment
        seen = []
        cfg = dict(CFG, max_loaded_models=1)

        stt = self._stt(cfg)

        import onnx_asr
        real_load = onnx_asr.load_model

        def _spy(*args, **kwargs):
            seen.append(len(stt._models))
            return real_load(*args, **kwargs)

        onnx_asr.load_model = _spy
        self.addCleanup(setattr, onnx_asr, "load_model", real_load)

        stt.execute(_audio(), language="ru")
        stt.execute(_audio(), language="es")
        self.assertTrue(all(count == 0 for count in seen),
                        f"models resident while loading another: {seen}")


class TestMaxLoadedModelsValidation(unittest.TestCase):
    def _stt(self, value):
        load = MagicMock(side_effect=lambda *a, **kw: _adapter())
        import onnx_asr
        previous = onnx_asr.load_model
        onnx_asr.load_model = load
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        from ovos_stt_plugin_onnxasr import OnnxASR
        return OnnxASR({"lang": "en", "model": "default-model",
                        "max_loaded_models": value})

    def test_string_value_is_coerced(self):
        self.assertEqual(self._stt("2").max_loaded_models, 2)

    def test_negative_value_does_not_crash(self):
        self.assertIsNone(self._stt(-1).max_loaded_models)

    def test_zero_does_not_crash(self):
        self.assertIsNone(self._stt(0).max_loaded_models)

    def test_garbage_value_does_not_crash(self):
        self.assertIsNone(self._stt("many").max_loaded_models)


if __name__ == "__main__":
    unittest.main()
