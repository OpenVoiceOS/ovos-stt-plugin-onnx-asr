import unittest
from unittest.mock import MagicMock

import numpy as np


def _adapter():
    adapter = MagicMock()
    adapter.asr = type("NemoConformerTDT", (), {})()
    adapter.recognize = MagicMock(return_value="transcript")
    return adapter


def _audio(samples, sample_rate=16000):
    audio = MagicMock()
    audio.get_np_float32.return_value = np.full(samples, 0.5, dtype=np.float32)
    audio.sample_rate = sample_rate
    return audio


class TestPadToSeconds(unittest.TestCase):
    def _stt(self, **cfg):
        adapter = _adapter()
        import onnx_asr
        # Assign and restore rather than ``mock.patch``: when the module is a
        # stub, patch removes the attribute on exit instead of putting the
        # previous value back, and every later test that stubs the same
        # attribute then fails.
        previous = onnx_asr.load_model
        onnx_asr.load_model = MagicMock(return_value=adapter)
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        from ovos_stt_plugin_onnxasr import OnnxASR
        return OnnxASR({"lang": "en", "model": "some-model", **cfg}), adapter

    def _heard(self, adapter):
        """The waveform the model was given by the last request."""
        return adapter.recognize.call_args.args[0]

    def test_unset_passes_the_audio_on_as_it_arrived(self):
        stt, adapter = self._stt()
        self.assertIsNone(stt.pad_to_seconds)
        stt.execute(_audio(16000), language="en")
        self.assertEqual(len(self._heard(adapter)), 16000)

    def test_audio_is_padded_with_silence_to_the_next_multiple(self):
        stt, adapter = self._stt(pad_to_seconds=4)
        stt.execute(_audio(16000), language="en")
        heard = self._heard(adapter)
        self.assertEqual(len(heard), 64000)
        self.assertEqual(heard.dtype, np.float32)
        # the request's own audio comes first, untouched; the rest is silence
        self.assertTrue((heard[:16000] == 0.5).all())
        self.assertFalse(heard[16000:].any())

    def test_requests_of_different_lengths_reach_the_model_with_one_length(self):
        stt, adapter = self._stt(pad_to_seconds=4)
        lengths = set()
        for samples in (9000, 16000, 31999, 48000, 64000):
            stt.execute(_audio(samples), language="en")
            lengths.add(len(self._heard(adapter)))
        self.assertEqual(lengths, {64000})

    def test_audio_longer_than_the_length_goes_to_the_multiple_after(self):
        stt, adapter = self._stt(pad_to_seconds=4)
        stt.execute(_audio(64001), language="en")
        self.assertEqual(len(self._heard(adapter)), 128000)

    def test_a_whole_multiple_is_passed_on_without_a_copy(self):
        stt, adapter = self._stt(pad_to_seconds=4)
        audio = _audio(64000)
        stt.execute(audio, language="en")
        self.assertIs(self._heard(adapter), audio.get_np_float32.return_value)

    def test_the_length_is_counted_in_the_audio_s_own_sample_rate(self):
        stt, adapter = self._stt(pad_to_seconds=2.5)
        stt.execute(_audio(8000, sample_rate=8000), language="en")
        self.assertEqual(len(self._heard(adapter)), 20000)
        self.assertEqual(adapter.recognize.call_args.kwargs["sample_rate"], 8000)

    def test_a_value_from_the_environment_arrives_as_a_string(self):
        stt, adapter = self._stt(pad_to_seconds="4")
        self.assertEqual(stt.pad_to_seconds, 4.0)
        stt.execute(_audio(16000), language="en")
        self.assertEqual(len(self._heard(adapter)), 64000)

    def test_a_value_that_names_no_length_is_ignored(self):
        for bad in ("four", 0, -4, "", [], "nan", "inf", False, True):
            with self.subTest(value=bad):
                stt, adapter = self._stt(pad_to_seconds=bad)
                self.assertIsNone(stt.pad_to_seconds)
                stt.execute(_audio(16000), language="en")
                self.assertEqual(len(self._heard(adapter)), 16000)

    def test_a_length_under_one_sample_pads_nothing(self):
        stt, adapter = self._stt(pad_to_seconds=0.00001)
        stt.execute(_audio(16000), language="en")
        self.assertEqual(len(self._heard(adapter)), 16000)
