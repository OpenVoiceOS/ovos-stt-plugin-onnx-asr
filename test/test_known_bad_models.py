"""A model known to return unusable text must not be a default.

A model that fails to load is loud: the log holds the error and the request
falls back. A model that loads and returns nonsense is silent, and every request
for that language gets nonsense. KNOWN_BAD_MODELS keeps such a model out of the
registry, and the registry is the only place that picks a model on its own.
"""
import unittest
from unittest.mock import patch

import ovos_stt_plugin_onnxasr.defaults as defaults


class TestKnownBadModels(unittest.TestCase):
    def test_no_language_points_at_a_known_bad_model(self):
        for lang, model in defaults.LANG_DEFAULTS.items():
            with self.subTest(lang=lang):
                self.assertNotIn(model, defaults.KNOWN_BAD_MODELS)

    def test_every_known_bad_model_carries_a_reason(self):
        for model, reason in defaults.KNOWN_BAD_MODELS.items():
            with self.subTest(model=model):
                self.assertTrue(reason.strip(),
                                f"{model} needs one line of evidence")

    def test_ga_serves_ga_the_kwa_language_and_not_irish(self):
        # "ga" in the model id names Ga, the Kwa language of Accra, whose tag is
        # "gaa". The two-letter tag "ga" is Irish, which this model does not
        # serve: 99.7% WER on Irish against 4.96% WER on Ga.
        self.assertEqual(defaults.LANG_DEFAULTS.get("gaa"),
                         "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx")
        self.assertNotEqual(defaults.LANG_DEFAULTS.get("ga"),
                            "OpenVoiceOS/misterkissi-w2v2-lg-xls-r-300m-ga-onnx")

    def test_a_new_known_bad_model_leaves_the_registry(self):
        # The list is the whole mechanism: nothing else has to change to drop a
        # model from every language it serves.
        served = defaults.LANG_DEFAULTS["gl"]
        table = {"gl": served, "en": "keep-me"}
        with patch.dict(defaults.KNOWN_BAD_MODELS, {served: "test"}):
            self.assertEqual(defaults.drop_known_bad(table), {"en": "keep-me"})
        self.assertEqual(defaults.drop_known_bad(table), table)


if __name__ == "__main__":
    unittest.main()
