"""Tests for the runtime ``wav2vec2-ctc`` registration shim (``_compat``).

onnx-asr is not installed here, so a minimal fake ``onnx_asr`` package is injected
into ``sys.modules`` reproducing the internals the shim touches: an
``onnx_asr.loader`` whose ``create_asr_resolver`` rebuilds a local ``model_types``
dict on every call and hands it to ``Resolver`` (exactly as the real one does), plus
the ``asr``/``onnx``/``utils`` names the vendored class imports. Absence of an
``onnx_asr.models.wav2vec2`` submodule mirrors an onnx-asr release without the type.
"""
import sys
import types
import unittest


def _install_fake_onnx_asr(with_native: bool = False):
    """Put a fake onnx_asr package in sys.modules; return its loader module."""
    for name in list(sys.modules):
        if name == "onnx_asr" or name.startswith("onnx_asr."):
            del sys.modules[name]

    # onnxruntime is a heavy transitive dep of onnx_asr; the vendored class imports
    # it at module load, so stub it for these dependency-free unit tests.
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = type("InferenceSession", (), {})
    sys.modules.setdefault("onnxruntime", ort)

    pkg = types.ModuleType("onnx_asr")
    pkg.__path__ = []  # mark as package so submodule imports are attempted
    sys.modules["onnx_asr"] = pkg

    asr = types.ModuleType("onnx_asr.asr")
    asr._AsrWithCtcDecoding = type("_AsrWithCtcDecoding", (), {})
    asr.Preprocessor = type("Preprocessor", (), {})
    onnx = types.ModuleType("onnx_asr.onnx")
    onnx.OnnxSessionOptions = dict
    utils = types.ModuleType("onnx_asr.utils")
    utils.is_float32_array = lambda a: True
    utils.is_int64_array = lambda a: True
    for mod in (asr, onnx, utils):
        sys.modules[mod.__name__] = mod
        setattr(pkg, mod.__name__.split(".")[-1], mod)

    loader = types.ModuleType("onnx_asr.loader")

    class Resolver:
        def __init__(self, model_types, model=None, local_dir=None, *, offline=None):
            self.model_types = model_types
            self.model = model
            if isinstance(model_types, type):
                self.model_type = model_types
            elif model in model_types:
                self.model_type = model_types[model]
            else:
                raise KeyError(f"model not supported: {model}")

    def create_asr_resolver(model=None, local_dir=None, *, offline=None):
        # rebuilt fresh on every call, just like the real onnx-asr; Resolver is
        # resolved from the module at call time (mirrors the real module-global
        # lookup, which is what lets the shim's spy capture the dict)
        model_types = {"whisper": object, "vosk": object}
        return loader.Resolver(model_types, model, local_dir, offline=offline)

    loader.Resolver = Resolver
    loader.create_asr_resolver = create_asr_resolver
    sys.modules["onnx_asr.loader"] = loader
    pkg.loader = loader

    if with_native:
        native = types.ModuleType("onnx_asr.models.wav2vec2")
        models = types.ModuleType("onnx_asr.models")
        models.__path__ = []
        sys.modules["onnx_asr.models"] = models
        sys.modules["onnx_asr.models.wav2vec2"] = native
        pkg.models = models

    return loader


class TestWav2Vec2Registration(unittest.TestCase):
    def setUp(self):
        # ensure a clean import of the shim against each freshly-faked onnx_asr
        for name in ("ovos_stt_plugin_onnxasr._compat",
                     "ovos_stt_plugin_onnxasr._wav2vec2"):
            sys.modules.pop(name, None)
        # The fake package replaces onnx_asr in sys.modules and has no
        # load_model. Restore whatever was there so tests that run afterwards
        # still see a module they can patch; test order is not fixed, so
        # leaking the fake makes unrelated tests fail on some runs only.
        saved = {name: mod for name, mod in sys.modules.items()
                 if name == "onnx_asr" or name.startswith("onnx_asr.")}
        self.addCleanup(self._restore_onnx_asr, saved)

    @staticmethod
    def _restore_onnx_asr(saved):
        for name in [n for n in sys.modules
                     if n == "onnx_asr" or n.startswith("onnx_asr.")]:
            del sys.modules[name]
        sys.modules.update(saved)

    def test_registers_wav2vec2_type(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc
        from ovos_stt_plugin_onnxasr._wav2vec2 import Wav2Vec2Ctc

        # before patching the type is unknown
        with self.assertRaises(KeyError):
            loader.create_asr_resolver("wav2vec2-ctc")

        ensure_wav2vec2_ctc()

        resolver = loader.create_asr_resolver("wav2vec2-ctc")
        self.assertIs(resolver.model_type, Wav2Vec2Ctc)

    def test_builtin_types_still_resolve(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc

        ensure_wav2vec2_ctc()
        resolver = loader.create_asr_resolver("whisper")
        self.assertIs(resolver.model_type, object)

    def test_idempotent(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc

        ensure_wav2vec2_ctc()
        patched = loader.create_asr_resolver
        ensure_wav2vec2_ctc()
        self.assertIs(loader.create_asr_resolver, patched)

    def test_entry_point_registers_wav2vec2(self):
        """``ensure_model_types`` is the single call site used by the plugin."""
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_model_types
        from ovos_stt_plugin_onnxasr._wav2vec2 import Wav2Vec2Ctc

        with self.assertRaises(KeyError):
            loader.create_asr_resolver("wav2vec2-ctc")

        ensure_model_types()

        resolver = loader.create_asr_resolver("wav2vec2-ctc")
        self.assertIs(resolver.model_type, Wav2Vec2Ctc)

    def test_native_support_is_noop(self):
        loader = _install_fake_onnx_asr(with_native=True)
        from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc

        original = loader.create_asr_resolver
        ensure_wav2vec2_ctc()
        self.assertIs(loader.create_asr_resolver, original)


if __name__ == "__main__":
    unittest.main()
