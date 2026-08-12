import unittest
from unittest.mock import patch, MagicMock

import numpy as np


# Enough of a Whisper vocabulary for the plugin to read the language tokens
# out of. A real export carries the whole vocabulary and all ninety-nine
# languages; only the "<|xx|>" entries decide whether a hint can be selected,
# so the languages these tests ask for stand in for the full list.
_WHISPER_LANGS = ("en es de pt gl fr it nl pl ru zh ja ko ar he tr sv da no "
                  "fi is cy af mk sr bs sq id ms th sw mg ht").split()
WHISPER_TOKENS = {"<|startoftranscript|>": 0, "<|endoftext|>": 1, "hello": 2}
WHISPER_TOKENS.update({f"<|{code}|>": i for i, code in
                       enumerate(_WHISPER_LANGS, start=3)})


def _fake_adapter(asr_class_name, captured, tokens=None):
    """Build a stand-in onnx-asr adapter whose ``asr`` attribute is an instance
    of a class named ``asr_class_name`` and whose ``recognize`` records the
    keyword arguments it was called with."""
    adapter = MagicMock()
    adapter.asr = type(asr_class_name, (), {})()
    if "Whisper" in asr_class_name:
        adapter.asr._tokens = WHISPER_TOKENS if tokens is None else tokens

    def recognize(waveform, sample_rate, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return "transcript"

    adapter.recognize = recognize
    return adapter


def _audio():
    audio = MagicMock()
    audio.get_np_float32.return_value = np.zeros(16000, dtype=np.float32)
    audio.sample_rate = 16000
    return audio


class TestLanguageKwargs(unittest.TestCase):
    def _run(self, asr_class_name):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter(asr_class_name, captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "dummy"})
            stt.execute(_audio(), language="es")
        return captured

    def test_whisper_gets_language_only(self):
        captured = self._run("WhisperOrt")
        self.assertEqual(captured, {"language": "es"})

    def test_canary_gets_language_and_target(self):
        captured = self._run("NemoConformerAED")
        self.assertEqual(captured, {"language": "es", "target_language": "es"})

    def test_nemo_ctc_gets_no_language_kwargs(self):
        captured = self._run("NemoConformerCtc")
        self.assertEqual(captured, {})

    def test_transducer_gets_no_language_kwargs(self):
        captured = self._run("NemoConformerRnnt")
        self.assertEqual(captured, {})

    def test_falls_back_to_instance_lang(self):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter("WhisperHf", captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "de", "model": "dummy"})
            stt.execute(_audio())
        self.assertEqual(captured, {"language": "de"})


class TestWhisperWithoutLanguageToken(unittest.TestCase):
    """A fine-tune into a language Whisper never carried has no "<|xx|>" token.

    onnx-asr looks that token up unguarded, so passing the hint raised
    KeyError and the server answered 500. The hint has to be dropped, which
    makes onnx-asr detect the language instead.
    """

    def _run(self, lang, tokens=None):
        captured = {}
        with patch("onnx_asr.load_model",
                   return_value=_fake_adapter("WhisperOrt", captured, tokens)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "dummy"})
            stt.execute(_audio(), language=lang)
        return captured

    def test_unlisted_language_drops_the_hint(self):
        # "tn" is Setswana, which Whisper never carried.
        self.assertEqual(self._run("tn"), {})

    def test_listed_language_still_sends_the_hint(self):
        self.assertEqual(self._run("es"), {"language": "es"})

    def test_bcp47_tag_matches_on_the_primary_subtag(self):
        self.assertEqual(self._run("de-AT"), {"language": "de"})

    def test_vocabulary_without_any_language_token(self):
        self.assertEqual(self._run("en", tokens={"hello": 0}), {})


class TestLanguageNormalization(unittest.TestCase):
    def _run(self, asr_class_name, lang):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter(asr_class_name, captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "en", "model": "dummy"})
            stt.execute(_audio(), language=lang)
        return captured

    def test_bcp47_tag_reduced_to_primary_subtag(self):
        captured = self._run("WhisperOrt", "en-US")
        self.assertEqual(captured["language"], "en")

    def test_uppercase_tag_lowercased(self):
        captured = self._run("WhisperOrt", "PT-PT")
        self.assertEqual(captured["language"], "pt")

    def test_config_lang_normalized_when_no_override(self):
        captured = {}
        with patch("onnx_asr.load_model", return_value=_fake_adapter("WhisperOrt", captured)):
            from ovos_stt_plugin_onnxasr import OnnxASR
            stt = OnnxASR({"lang": "gl-ES", "model": "dummy"})
            stt.execute(_audio())
        self.assertEqual(captured["language"], "gl")
