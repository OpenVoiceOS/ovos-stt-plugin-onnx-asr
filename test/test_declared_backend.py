"""The installed distribution declares an inference runtime and a model hub.

`onnx-asr` keeps every runtime behind an extra: `onnxruntime` under `cpu`,
`onnxruntime-gpu` under `gpu`, `huggingface-hub` under `hub`. Its
unconditional requirements are `numpy` and `typing-extensions` alone. So a
plain `onnx_asr` requirement installs a package that cannot run, and this
plugin needs both extras: it imports `onnxruntime` directly in `_wav2vec2.py`,
and it resolves model ids from Hugging Face.

The rest of the suite stubs `onnx_asr` in `conftest.py` and never loads a real
model, so no other test can see a missing runtime. This one reads the
packaging metadata instead, which is where the defect lives.
"""
import unittest
from importlib.metadata import requires

DIST = "ovos-stt-plugin-onnx-asr"


def _onnx_asr_requirement():
    """The distribution's own `onnx_asr` requirement string."""
    for req in requires(DIST) or []:
        name = req.split()[0].split(">")[0].split("<")[0].split("=")[0]
        if name.replace("-", "_").startswith("onnx_asr"):
            return req
    return None


class TestDeclaredBackend(unittest.TestCase):
    def test_the_distribution_requires_onnx_asr(self):
        """The control. Without it the assertions below pass on a typo."""
        self.assertIsNotNone(_onnx_asr_requirement(),
                             f"{DIST} declares no onnx_asr requirement")

    def test_the_requirement_names_an_inference_runtime(self):
        """`onnx_asr` alone installs no runtime, so the plugin cannot import:
        `_wav2vec2.py` does `import onnxruntime`."""
        req = _onnx_asr_requirement()
        self.assertIn("cpu", req,
                      f"{req} names no runtime extra; onnxruntime is behind "
                      f"onnx-asr's 'cpu' extra")

    def test_the_requirement_names_the_model_hub(self):
        """Without it `onnx_asr.load_model` raises
        `ModuleNotFoundError: No module named 'huggingface_hub'` for any
        registry id."""
        req = _onnx_asr_requirement()
        self.assertIn("hub", req,
                      f"{req} names no hub extra; huggingface-hub is behind "
                      f"onnx-asr's 'hub' extra")

    def test_onnxruntime_is_importable_in_this_environment(self):
        """The metadata assertions above are about the declaration. This one
        is about the environment the suite runs in, so a green suite on a
        machine with no runtime cannot be mistaken for a working install."""
        import onnxruntime  # noqa: F401

    def test_huggingface_hub_is_importable_in_this_environment(self):
        import huggingface_hub  # noqa: F401


if __name__ == "__main__":
    unittest.main()
