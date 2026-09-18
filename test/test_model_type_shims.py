"""Tests for the vendored model type shims (``_shims``).

onnx-asr is not installed here, so a fake ``onnx_asr`` package is injected into
``sys.modules``. It reproduces the internals the shims touch: a ``loader`` whose
``create_asr_resolver`` rebuilds a local ``model_types`` dict on every call and
hands it to ``Resolver`` (exactly as the real one does), the base classes and
helpers the vendored model modules import, and a ``preprocessors`` package.

The fake carries no model module of its own, which mirrors an onnx-asr release
without the families. A test that wants a family to be native adds the module.

The fake replaces the shared stub from ``conftest``. Every test restores what it
found, because test order is not fixed.
"""
import re
import onnxruntime  # noqa: F401  # keep the real module: a stub breaks the vendored imports
import sys
import types
import unittest
import unittest.mock

# Every family in _shims, with the model type names it must register.
FAMILIES = {
    "espnet": ["espnet-aed", "espnet-ctc"],
    "granite-nar": ["granite-nar"],
    "moonshine": ["moonshine", "moonshine-tiny", "moonshine-base"],
    "omnilingual": ["omnilingual-ctc"],
    "paraformer": ["paraformer"],
    "sensevoice": ["sensevoice"],
    "speech-llm": ["speech-llm"],
    "wav2vec2-adapters": ["wav2vec2-adapters"],
    "wav2vec2-ctc": ["wav2vec2-ctc"],
}

BUILTIN_TYPES = ("whisper", "vosk")


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install_fake_onnx_asr(native=()):
    """Put a fake onnx_asr package in sys.modules; return its loader module.

    Args:
        native: Family module base names the fake provides itself, for example
            ``("moonshine",)``.
    """
    for name in list(sys.modules):
        if name == "onnx_asr" or name.startswith("onnx_asr."):
            del sys.modules[name]

    pkg = types.ModuleType("onnx_asr")
    pkg.__path__ = []
    sys.modules["onnx_asr"] = pkg

    class BaseAsr:
        def __init__(self, *args, **kwargs):
            pass

    class _AsrWithDecoding(BaseAsr):
        pass

    class _AsrWithCtcDecoding(_AsrWithDecoding):
        pass

    asr = _module(
        "onnx_asr.asr",
        BaseAsr=BaseAsr,
        _AsrWithDecoding=_AsrWithDecoding,
        _AsrWithCtcDecoding=_AsrWithCtcDecoding,
        Preprocessor=type("Preprocessor", (), {}),
        TimestampedResult=type("TimestampedResult", (), {}),
    )
    onnx = _module(
        "onnx_asr.onnx",
        OnnxSessionOptions=dict,
        TensorRtOptions=type("TensorRtOptions", (), {}),
        get_onnx_device=lambda session: ("cpu", 0),
        get_onnx_providers=lambda options: [],
        update_onnx_providers=lambda options, **kwargs: options,
    )

    class ModelLoadingError(Exception):
        pass

    class ModelFileNotFoundError(ModelLoadingError, FileNotFoundError):
        def __init__(self, filename, path):
            super().__init__(f"{filename} not found in {path}")

    class MoreThanOneModelFileFoundError(ModelLoadingError, OSError):
        pass

    utils = _module(
        "onnx_asr.utils",
        is_float32_array=lambda a: True,
        is_int32_array=lambda a: True,
        is_int64_array=lambda a: True,
        log_softmax=lambda logits, axis=None: logits,
        ModelFileNotFoundError=ModelFileNotFoundError,
        MoreThanOneModelFileFoundError=MoreThanOneModelFileFoundError,
    )

    models = types.ModuleType("onnx_asr.models")
    models.__path__ = []
    whisper = _module("onnx_asr.models.whisper", bytes_to_unicode=lambda: {i: chr(i) for i in range(256)})
    sys.modules["onnx_asr.models"] = models
    sys.modules["onnx_asr.models.whisper"] = whisper
    models.whisper = whisper

    preprocessors = types.ModuleType("onnx_asr.preprocessors")
    preprocessors.__path__ = []

    class _NumpyPreprocessor:
        def __init__(self, name):
            raise KeyError(name)

    numpy_preprocessor = _module("onnx_asr.preprocessors.numpy_preprocessor", _NumpyPreprocessor=_NumpyPreprocessor)
    sys.modules["onnx_asr.preprocessors"] = preprocessors
    sys.modules["onnx_asr.preprocessors.numpy_preprocessor"] = numpy_preprocessor
    preprocessors.numpy_preprocessor = numpy_preprocessor

    class Resolver:
        def __init__(self, model_types, model=None, local_dir=None, *, offline=None):
            self.model_types = model_types
            self.model = model
            if model in model_types:
                self.model_type = model_types[model]
            else:
                raise KeyError(f"model not supported: {model}")

        @staticmethod
        def _download_model(quantization, *, local_files_only):
            raise NotImplementedError

        @staticmethod
        def _resolve_model_files(path, quantization):
            raise NotImplementedError

    resolver = _module("onnx_asr.resolver", Resolver=Resolver, model_repos={})

    loader = types.ModuleType("onnx_asr.loader")

    class Manager:
        default_onnx_config: dict = {}

        def _create_preprocessor(self, name):
            return ("builtin", name)

        def _create_asr_adapter(self, asr):
            return asr

        def create_asr(self, model=None, local_dir=None, *, quantization=None, offline=None, config=None):
            return loader.create_asr_resolver(model, local_dir, offline=offline)

    def create_asr_resolver(model=None, local_dir=None, *, offline=None):
        # rebuilt fresh on every call, just like the real onnx-asr
        model_types = {name: object for name in BUILTIN_TYPES}
        return loader.Resolver(model_types, model, local_dir, offline=offline)

    loader.Resolver = Resolver
    loader.Manager = Manager
    loader.create_asr_resolver = create_asr_resolver
    loader.update_onnx_providers = lambda options, **kwargs: options

    for mod in (asr, onnx, utils, resolver, loader):
        sys.modules[mod.__name__] = mod
        setattr(pkg, mod.__name__.split(".")[-1], mod)
    pkg.models = models
    pkg.preprocessors = preprocessors

    for base in native:
        name = f"onnx_asr.models.{base}"
        sys.modules[name] = types.ModuleType(name)

    return loader


class _FakeOnnxAsrTestCase(unittest.TestCase):
    """Installs the fake onnx_asr and always puts back what it replaced."""

    def setUp(self):
        for name in list(sys.modules):
            if name.startswith("ovos_stt_plugin_onnxasr."):
                del sys.modules[name]
        saved = {name: mod for name, mod in sys.modules.items()
                 if name == "onnx_asr" or name.startswith("onnx_asr.")}
        self.addCleanup(self._restore_onnx_asr, saved)

    @staticmethod
    def _restore_onnx_asr(saved):
        for name in [n for n in sys.modules if n == "onnx_asr" or n.startswith("onnx_asr.")]:
            del sys.modules[name]
        sys.modules.update(saved)


class TestFamilyRegistration(_FakeOnnxAsrTestCase):
    def test_unregistered_types_do_not_resolve(self):
        """RED: without the shim every extra model type is unknown."""
        loader = _install_fake_onnx_asr()
        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    with self.assertRaises(KeyError):
                        loader.create_asr_resolver(type_name)

    def test_every_family_resolves_after_the_shim(self):
        """GREEN: after the shim every extra model type resolves to a class."""
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    resolver = loader.create_asr_resolver(type_name)
                    self.assertTrue(isinstance(resolver.model_type, type))
                    self.assertNotIn(resolver.model_type, (object,))

    def test_one_class_per_family(self):
        """Each family resolves to a class that comes from its own module."""
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        expected = {
            "espnet-aed": "espnet",
            "espnet-ctc": "espnet",
            "granite-nar": "granite_nar",
            "moonshine": "moonshine",
            "omnilingual-ctc": "omnilingual",
            "paraformer": "paraformer",
            "sensevoice": "sensevoice",
            "speech-llm": "speech_llm",
            "wav2vec2-adapters": "wav2vec2_adapters",
            "wav2vec2-ctc": "_wav2vec2",
        }
        for type_name, module_suffix in expected.items():
            with self.subTest(model_type=type_name):
                model_type = loader.create_asr_resolver(type_name).model_type
                self.assertTrue(model_type.__module__.endswith(module_suffix), model_type.__module__)

    def test_builtin_types_still_resolve(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()
        for type_name in BUILTIN_TYPES:
            with self.subTest(model_type=type_name):
                self.assertIs(loader.create_asr_resolver(type_name).model_type, object)

    def test_idempotent(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()
        patched = loader.create_asr_resolver
        ensure_model_types()
        self.assertIs(loader.create_asr_resolver, patched)

    def test_only_selects_one_family(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types(only={"moonshine"})

        self.assertTrue(isinstance(loader.create_asr_resolver("moonshine").model_type, type))
        with self.assertRaises(KeyError):
            loader.create_asr_resolver("sensevoice")


class TestUpstreamWins(_FakeOnnxAsrTestCase):
    def test_native_family_is_not_overridden(self):
        """A family the installed onnx-asr provides is left alone."""
        loader = _install_fake_onnx_asr(native=("moonshine",))
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        with self.assertRaises(KeyError):
            loader.create_asr_resolver("moonshine")
        # the other families are unaffected
        self.assertTrue(isinstance(loader.create_asr_resolver("sensevoice").model_type, type))

    def test_native_model_type_name_is_not_overridden(self):
        """A model type name already in the resolver mapping keeps its class."""
        loader = _install_fake_onnx_asr()
        original_create = loader.create_asr_resolver
        sentinel = type("NativeSenseVoice", (), {})

        def create_asr_resolver(model=None, local_dir=None, *, offline=None):
            model_types = {name: object for name in BUILTIN_TYPES}
            model_types["sensevoice"] = sentinel
            return loader.Resolver(model_types, model, local_dir, offline=offline)

        loader.create_asr_resolver = create_asr_resolver
        self.addCleanup(setattr, loader, "create_asr_resolver", original_create)

        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        self.assertIs(loader.create_asr_resolver("sensevoice").model_type, sentinel)

    def test_all_native_leaves_onnx_asr_untouched(self):
        loader = _install_fake_onnx_asr(native=(
            "espnet", "granite_nar", "moonshine", "omnilingual", "paraformer",
            "sensevoice", "speech_llm", "wav2vec2_adapters", "wav2vec2",
        ))
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        original = loader.create_asr_resolver
        ensure_model_types()
        self.assertIs(loader.create_asr_resolver, original)


class TestIsolation(_FakeOnnxAsrTestCase):
    def test_a_broken_family_does_not_stop_the_others(self):
        loader = _install_fake_onnx_asr()
        import ovos_stt_plugin_onnxasr._shims as shims

        original_families = shims._families
        self.addCleanup(setattr, shims, "_families", original_families)

        def broken():
            raise RuntimeError("this family is broken on purpose")

        shims._families = lambda: [
            shims._family("broken", "onnx_asr.models.broken", broken),
            *original_families(),
        ]

        shims.ensure_model_types()

        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    self.assertTrue(isinstance(loader.create_asr_resolver(type_name).model_type, type))

    def test_a_broken_core_patch_skips_only_its_family(self):
        loader = _install_fake_onnx_asr()
        import ovos_stt_plugin_onnxasr._shims as shims
        from ovos_stt_plugin_onnxasr._shims import _core

        def broken_patch():
            raise RuntimeError("core patch failed on purpose")

        original_patch = _core.patch_fetcher_support
        _core.patch_fetcher_support = broken_patch
        self.addCleanup(setattr, _core, "patch_fetcher_support", original_patch)

        shims.ensure_model_types()

        with self.assertRaises(KeyError):
            loader.create_asr_resolver("wav2vec2-adapters")
        self.assertTrue(isinstance(loader.create_asr_resolver("sensevoice").model_type, type))


    def test_a_native_module_that_does_not_import_falls_back(self):
        """A native module that raises on import must not remove the family.

        The user gets the vendored class. Without this, the family counts as native,
        the plugin stays out of the way, and the model type resolves nowhere.
        """
        import importlib

        loader = _install_fake_onnx_asr()
        real_import_module = importlib.import_module

        def import_module(name, *args, **kwargs):
            if name == "onnx_asr.models.sensevoice":
                raise AttributeError("this native module is broken on purpose")
            return real_import_module(name, *args, **kwargs)

        patcher = unittest.mock.patch("importlib.import_module", import_module)
        patcher.start()
        self.addCleanup(patcher.stop)

        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        model_type = loader.create_asr_resolver("sensevoice").model_type
        self.assertTrue(model_type.__module__.endswith("sensevoice"), model_type.__module__)


class TestAdditiveRegistration(_FakeOnnxAsrTestCase):
    """A call that names some families must not lock the others out."""

    def test_a_later_call_adds_the_families_the_first_call_left_out(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types(only={"moonshine"})
        ensure_model_types()

        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    self.assertTrue(isinstance(loader.create_asr_resolver(type_name).model_type, type))

    def test_ensure_wav2vec2_ctc_does_not_lock_out_the_other_families(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_model_types, ensure_wav2vec2_ctc

        ensure_wav2vec2_ctc()
        ensure_model_types()

        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    self.assertTrue(isinstance(loader.create_asr_resolver(type_name).model_type, type))

    def test_a_second_call_keeps_what_the_first_call_added(self):
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()
        ensure_model_types(only={"moonshine"})

        for family, type_names in FAMILIES.items():
            for type_name in type_names:
                with self.subTest(family=family, model_type=type_name):
                    self.assertTrue(isinstance(loader.create_asr_resolver(type_name).model_type, type))


class TestResolverApiDrift(_FakeOnnxAsrTestCase):
    def test_capture_failure_leaves_onnx_asr_untouched(self):
        loader = _install_fake_onnx_asr()

        def create_asr_resolver(model=None, local_dir=None, *, offline=None):
            raise RuntimeError("internals changed")

        loader.create_asr_resolver = create_asr_resolver
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        ensure_model_types()

        self.assertIs(loader.create_asr_resolver, create_asr_resolver)


class TestW2vBertPreprocessor(_FakeOnnxAsrTestCase):
    def test_manager_builds_the_w2vbert_preprocessor(self):
        """ESPnet needs a w2v-BERT preprocessor that a release does not ship."""
        loader = _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        manager = loader.Manager()
        self.assertEqual(manager._create_preprocessor("w2vbert"), ("builtin", "w2vbert"))

        ensure_model_types()

        preprocessor = loader.Manager()._create_preprocessor("w2vbert")
        self.assertEqual(preprocessor._melscale_fbanks.shape, (257, 80))
        # every other preprocessor name still goes to onnx-asr
        self.assertEqual(loader.Manager()._create_preprocessor("nemo80"), ("builtin", "nemo80"))

    def test_filterbank_matches_the_kaldi_mel_scale(self):
        """The matrix must equal a Kaldi mel filterbank computed on its own.

        The reference below is written from the definition, not from the vendored
        code: 80 triangular bins, evenly spaced on the Kaldi mel scale
        ``1127 * ln(1 + f / 700)`` between 20 Hz and 8 kHz, over the 257 frequency
        points of a 512-point FFT at 16 kHz.
        """
        import numpy as np

        _install_fake_onnx_asr()
        from ovos_stt_plugin_onnxasr._shims._core import _w2vbert_melscale_fbanks

        def to_mel(hz):
            return 1127.0 * np.log(1.0 + hz / 700.0)

        n_freqs, n_mels = 257, 80
        freqs = [8000.0 * i / (n_freqs - 1) for i in range(n_freqs)]
        low, high = to_mel(20.0), to_mel(8000.0)
        edges = [low + (high - low) * i / (n_mels + 1) for i in range(n_mels + 2)]

        reference = np.zeros((n_freqs, n_mels))
        for row, freq in enumerate(freqs):
            mel = to_mel(freq)
            for col in range(n_mels):
                left, centre, right = edges[col], edges[col + 1], edges[col + 2]
                up = (mel - left) / (centre - left)
                down = (right - mel) / (right - centre)
                reference[row, col] = max(0.0, min(up, down))

        fbanks = _w2vbert_melscale_fbanks()
        self.assertEqual(fbanks.shape, (n_freqs, n_mels))
        self.assertLess(float(np.abs(fbanks - reference).max()), 1e-6)


class TestVendoredProvenance(unittest.TestCase):
    def test_every_vendored_file_names_its_source_commit(self):
        from pathlib import Path

        import ovos_stt_plugin_onnxasr

        vendored = Path(ovos_stt_plugin_onnxasr.__file__).parent / "_shims" / "_vendored"
        files = sorted(p for p in vendored.glob("*.py") if p.name != "__init__.py")
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=path.name):
                header = path.read_text()[:2000]
                self.assertIn("MIT License", header)
                self.assertRegex(header, r"Source commit:\s+[0-9a-f]{40}")
                self.assertRegex(header, r"Upstream project:\s+https://github\.com/istupakov/onnx-asr")

    def test_source_paths_say_what_they_are_relative_to(self):
        """A re-syncer must not have to guess where a source path starts.

        The fork holds a ``preprocessors`` directory at its root as well as
        ``src/onnx_asr/preprocessors``, so a bare ``preprocessors/...`` path is
        ambiguous unless the package says which root the paths use.
        """
        from pathlib import Path

        import ovos_stt_plugin_onnxasr
        from ovos_stt_plugin_onnxasr._shims import _vendored

        self.assertIn("from the root of the source fork", _vendored.__doc__)

        path = Path(ovos_stt_plugin_onnxasr.__file__).parent / "_shims" / "_vendored" / "fbanks.py"
        header = path.read_text()[:2000]
        self.assertIn("from the root of the fork", header)

    def test_all_vendored_files_share_one_source_commit(self):
        from pathlib import Path

        import ovos_stt_plugin_onnxasr

        vendored = Path(ovos_stt_plugin_onnxasr.__file__).parent / "_shims" / "_vendored"
        commits = {
            re.search(r"Source commit:\s+([0-9a-f]{40})", path.read_text()[:2000]).group(1)
            for path in vendored.glob("*.py")
            if "Source commit" in path.read_text()[:2000]
        }
        self.assertEqual(len(commits), 1, commits)


if __name__ == "__main__":
    unittest.main()
