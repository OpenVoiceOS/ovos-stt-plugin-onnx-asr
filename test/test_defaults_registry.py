import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sys

# stub the heavy backend before importing the package (same as test_providers)
sys.modules.setdefault("onnx_asr", MagicMock())

from ovos_stt_plugin_onnxasr.defaults import (LANG_DEFAULTS, env_lang_defaults,
                                              resolve_model)


class TestResolveModel(unittest.TestCase):
    def test_builtin_registry_primary_subtag(self):
        self.assertEqual(resolve_model("gl-ES", {}, "fallback"),
                         "OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx")

    def test_config_beats_registry(self):
        self.assertEqual(resolve_model("gl", {"gl": "custom"}, None), "custom")

    def test_full_tag_beats_primary(self):
        self.assertEqual(
            resolve_model("pt-BR", {"pt-br": "brazil-model", "pt": "portugal-model"}, None),
            "brazil-model")
        self.assertEqual(
            resolve_model("pt-PT", {"pt-br": "brazil-model", "pt": "portugal-model"}, None),
            "portugal-model")

    def test_unknown_lang_falls_back_to_default(self):
        self.assertEqual(resolve_model("tlh", {}, "fallback"), "fallback")

    def test_env_var_primary(self):
        with patch.dict("os.environ", {"ONNX_ASR_DEFAULT_PT": "env-model"}):
            self.assertEqual(resolve_model("pt", {}, None), "env-model")

    def test_env_var_full_bcp47(self):
        with patch.dict("os.environ", {"ONNX_ASR_DEFAULT_PT_BR": "env-brazil",
                                       "ONNX_ASR_DEFAULT_PT": "env-pt"}):
            self.assertEqual(resolve_model("pt-BR", {}, None), "env-brazil")
            self.assertEqual(resolve_model("pt-PT", {}, None), "env-pt")

    def test_env_beats_registry_config_beats_env(self):
        with patch.dict("os.environ", {"ONNX_ASR_DEFAULT_GL": "env-model"}):
            self.assertEqual(resolve_model("gl", {}, None), "env-model")
            self.assertEqual(resolve_model("gl", {"gl": "config-model"}, None),
                             "config-model")

    def test_env_parsing(self):
        with patch.dict("os.environ", {"ONNX_ASR_DEFAULT_ZH_HANT": "m1",
                                       "ONNX_ASR_DEFAULT_RU": "m2",
                                       "OTHER_VAR": "x"}):
            langs = env_lang_defaults()
            self.assertEqual(langs.get("zh-hant"), "m1")
            self.assertEqual(langs.get("ru"), "m2")
            self.assertNotIn("other-var", langs)

    def test_registry_sanity(self):
        # every entry is a non-empty string; key langs present
        for k, v in LANG_DEFAULTS.items():
            self.assertTrue(v and isinstance(v, str), k)
        for lang in ("en", "pt", "es", "gl", "eu", "ca", "ru", "hi", "ar", "zh"):
            self.assertIn(lang, LANG_DEFAULTS)


class TestPluginUsesRegistry(unittest.TestCase):
    def test_execute_routes_via_registry(self):
        models = {}
        def load(model_id, **kw):
            m = MagicMock()
            m.asr = type("NemoConformerCTC", (), {})()
            m.recognize = MagicMock(return_value="ok")
            models[model_id] = m
            return m
        audio = MagicMock()
        audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
        audio.sample_rate = 16000
        with patch("onnx_asr.load_model", load):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "default-model"})
            stt.execute(audio, language="gl-ES")
        self.assertIn("OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx", models)


class TestRegistryIntegrity(unittest.TestCase):
    """Guards against registry entries that cannot actually be loaded."""

    def test_no_known_empty_repos(self):
        # OpenVoiceOS/ai4bharat-indicconformer-as-onnx holds no ONNX weights;
        # Assamese must not point at it until one is published.
        self.assertNotIn("ai4bharat-indicconformer-as-onnx",
                         LANG_DEFAULTS.get("as", ""))

    def test_indic_langs_covered(self):
        for lang in ("hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or",
                     "as", "ur"):
            self.assertIn(lang, LANG_DEFAULTS)

    def test_hf_repo_ids_well_formed(self):
        for lang, model in LANG_DEFAULTS.items():
            if "/" in model:
                org, _, name = model.partition("/")
                self.assertTrue(org and name, f"{lang}: {model}")
                self.assertNotIn(" ", model, f"{lang}: {model}")
