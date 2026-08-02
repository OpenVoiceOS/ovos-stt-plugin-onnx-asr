"""Tests for execution-provider (CPU/GPU) selection in OnnxASR.

``onnx_asr`` is stubbed so these run without the heavy onnxruntime dep; we
only exercise the plugin's config -> providers wiring and what gets passed to
``onnx_asr.load_model``.
"""
import sys
import unittest
from unittest.mock import MagicMock

# stub the heavy backend before importing the plugin; setdefault returns the
# already-installed stub when another test module got there first, so we always
# hold the same object the plugin module is bound to
_onnx_asr = sys.modules.setdefault("onnx_asr", MagicMock())

from ovos_stt_plugin_onnxasr import OnnxASR  # noqa: E402


def _make(config):
    _onnx_asr.load_model.reset_mock()
    plugin = OnnxASR(config=config)
    return plugin, _onnx_asr.load_model.call_args


class TestProviders(unittest.TestCase):
    def test_default_is_none(self):
        plugin, call = _make({"model": "m"})
        self.assertIsNone(plugin._get_providers())
        self.assertIsNone(call.kwargs.get("providers"))

    def test_use_cuda_selects_cuda_with_cpu_fallback(self):
        plugin, call = _make({"model": "m", "use_cuda": True})
        self.assertEqual(plugin._get_providers(),
                         ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(call.kwargs.get("providers"),
                         ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_explicit_providers_take_precedence(self):
        plugin, _ = _make({"model": "m", "use_cuda": True,
                           "providers": ["TensorrtExecutionProvider"]})
        self.assertEqual(plugin._get_providers(),
                         ["TensorrtExecutionProvider"])

    def test_quantization_and_model_forwarded(self):
        _, call = _make({"model": "nemo-parakeet-tdt-0.6b-v3",
                         "quantization": "int8"})
        self.assertEqual(call.args[0], "nemo-parakeet-tdt-0.6b-v3")
        self.assertEqual(call.kwargs.get("quantization"), "int8")


if __name__ == "__main__":
    unittest.main()
