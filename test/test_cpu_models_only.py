"""``cpu_models_only`` restricts model selection to models a CPU-only
deployment can run.

Off (the default) changes nothing: every model in the catalogue stays
selectable, exactly as before the flag existed. On, a model whose id
advertises a parameter count the plugin treats as too large for CPU-only
inference (see ``defaults.CPU_MODEL_PARAM_LIMIT``) drops out of the
per-language registry, and naming one explicitly (``model`` or
``lang2model``) is refused rather than silently swapped for something else.
"""
import unittest
from unittest.mock import MagicMock, patch

from ovos_stt_plugin_onnxasr import defaults
from ovos_stt_plugin_onnxasr.defaults import LANG_DEFAULTS


def _adapter(asr_class_name="NemoConformerTDT"):
    adapter = MagicMock()
    adapter.asr = type(asr_class_name, (), {})()
    adapter.recognize = MagicMock(return_value="transcript")
    return adapter


CFG = {"lang": "en", "model": "default-model"}


class TestCpuModelsOnly(unittest.TestCase):
    def _stt(self, cfg, load_model=None):
        load_model = load_model or MagicMock(return_value=_adapter())
        with patch("onnx_asr.load_model", load_model):
            from ovos_stt_plugin_onnxasr import OnnxASR
            return OnnxASR(dict(cfg))

    def test_default_off_preserves_the_current_model_list_exactly(self):
        stt = self._stt(CFG)
        self.assertFalse(stt.cpu_models_only)
        self.assertEqual(stt._lang_registry, LANG_DEFAULTS)
        # available_languages is unfiltered: every LANG_DEFAULTS language is
        # still advertised.
        for lang in LANG_DEFAULTS:
            self.assertIn(defaults.standardize_lang(lang), stt.available_languages)

    def test_on_filters_the_registry(self):
        cfg = dict(CFG)
        cfg["cpu_models_only"] = True
        stt = self._stt(cfg)
        self.assertTrue(stt.cpu_models_only)
        self.assertLess(len(stt._lang_registry), len(LANG_DEFAULTS))
        for model_id in stt._lang_registry.values():
            self.assertTrue(defaults.is_cpu_friendly(model_id))
        # a language whose registry default is a 0.6B+ model drops out
        self.assertNotIn("en", stt._lang_registry)

    def test_on_explicit_excluded_model_raises_naming_the_flag(self):
        cfg = {"lang": "en", "model": "nemo-canary-1b-v2",
               "cpu_models_only": True}
        with self.assertRaises(ValueError) as ctx:
            self._stt(cfg)
        self.assertIn("cpu_models_only", str(ctx.exception))
        self.assertIn("nemo-canary-1b-v2", str(ctx.exception))

    def test_on_explicit_excluded_lang2model_entry_raises_naming_the_flag(self):
        cfg = {"lang": "en", "model": "moonshine-base",
               "cpu_models_only": True,
               "lang2model": {"en": "nemo-parakeet-tdt-0.6b-v2"}}
        with self.assertRaises(ValueError) as ctx:
            self._stt(cfg)
        self.assertIn("cpu_models_only", str(ctx.exception))
        self.assertIn("nemo-parakeet-tdt-0.6b-v2", str(ctx.exception))

    def test_garbage_config_value_does_not_crash(self):
        cfg = dict(CFG)
        cfg["cpu_models_only"] = {"not": "a bool"}
        stt = self._stt(cfg)
        self.assertFalse(stt.cpu_models_only)
        self.assertEqual(stt._lang_registry, LANG_DEFAULTS)

    def test_garbage_string_config_value_does_not_crash(self):
        cfg = dict(CFG)
        cfg["cpu_models_only"] = "banana"
        stt = self._stt(cfg)
        self.assertFalse(stt.cpu_models_only)


if __name__ == "__main__":
    unittest.main()
