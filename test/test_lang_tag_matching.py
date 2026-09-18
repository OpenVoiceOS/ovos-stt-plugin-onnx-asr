"""Language tag matching in :func:`resolve_model`.

Tag comparison is delegated to ``ovos-spec-tools``, which implements
OVOS-INTENT-2 §2. These tests pin the behaviour that matters to a caller: the
tag forms a speech client really sends, and the languages that must stay apart.
"""
import os
import unittest
from unittest.mock import patch

from ovos_stt_plugin_onnxasr import defaults
from ovos_stt_plugin_onnxasr.defaults import resolve_model

FALLBACK = "FALLBACK"


class TestTagForms(unittest.TestCase):
    """A tag reaches its entry whatever shape the caller writes it in."""

    def test_underscore_region_reaches_the_language(self):
        # "pt_PT" is the form a locale-derived config writes.
        self.assertEqual(resolve_model("pt_PT", {}, FALLBACK),
                         defaults.LANG_DEFAULTS["pt"])

    def test_english_with_region_reaches_english(self):
        for tag in ("en-US", "en_US", "EN-us", "en-GB"):
            with self.subTest(tag=tag):
                self.assertEqual(resolve_model(tag, {}, FALLBACK),
                                 defaults.LANG_DEFAULTS["en"])

    def test_lang2model_key_written_with_an_underscore(self):
        # The operator's spelling of the key must not decide whether it applies.
        self.assertEqual(resolve_model("pt-BR", {"pt_BR": "CUSTOM"}, FALLBACK),
                         "CUSTOM")

    def test_env_var_suffix_with_a_region(self):
        with patch.dict(os.environ, {"ONNX_ASR_DEFAULT_PT_BR": "ENVMODEL"}):
            self.assertEqual(resolve_model("pt-BR", {}, FALLBACK), "ENVMODEL")

    def test_traditional_chinese_reaches_the_chinese_model(self):
        for tag in ("zh-Hant", "zh-TW", "zh-HK", "zh-CN"):
            with self.subTest(tag=tag):
                self.assertEqual(resolve_model(tag, {}, FALLBACK),
                                 defaults.LANG_DEFAULTS["zh"])


class TestLanguagesStayApart(unittest.TestCase):
    """Nearest-tag matching must not hand a request to another language."""

    def test_irish_does_not_reach_the_ga_model(self):
        # "ga" is Irish; "gaa" is Ga, the Kwa language of Accra. Sharing two
        # letters is not sharing a language.
        ga_model = defaults.LANG_DEFAULTS["gaa"]
        for tag in ("ga", "ga-IE"):
            with self.subTest(tag=tag):
                self.assertNotEqual(resolve_model(tag, {}, FALLBACK), ga_model)

    def test_a_language_with_no_entry_falls_back(self):
        for tag in ("ga", "yue", "zz"):
            with self.subTest(tag=tag):
                self.assertEqual(resolve_model(tag, {}, FALLBACK), FALLBACK)


if __name__ == "__main__":
    unittest.main()
