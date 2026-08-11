"""Tests for the model type shims against the installed onnx-asr.

The rest of the suite drives the shims through a fake ``onnx_asr``. A fake cannot
show what the shims do to the real resolver, the real model manager and the real
model repository table. These tests use the installed onnx-asr instead.

onnx-asr is a dependency of the plugin, so it is always installed. No test here
downloads a model: a resolver reads the repository table and the model type table
only, and every test stops the load before a file is fetched.

Each test gets an onnx-asr of its own: the modules are removed from
``sys.modules`` and imported again, so a test starts from an unpatched onnx-asr and
gives back what it found. The plugin modules go the same way, because the vendored
model classes bind onnx-asr base classes at import time.
"""
import importlib
import sys
import threading
import unittest

# Short names that must load a model, whatever else the table grows to hold.
VERIFIED_REPOS = {
    "espnet-ctc": "OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx",
    "espnet-aed": "OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx",
    "omnilingual-ctc": "OpenVoiceOS/omnilingual-asr-ctc-1b-onnx",
}

# A built-in model type with a repository of its own, so a resolver for it is built
# without a name lookup failure and without any download.
BUILTIN_MODEL = "whisper-base"


class _Stop(Exception):
    """Ends a model load after the step under test."""


def _module_names():
    return [
        name
        for name in sys.modules
        if name in ("onnx_asr", "ovos_stt_plugin_onnxasr")
        or name.startswith("onnx_asr.")
        or name.startswith("ovos_stt_plugin_onnxasr.")
    ]


class RealOnnxAsrTestCase(unittest.TestCase):
    """Gives every test an onnx-asr of its own, and puts back what it found."""

    def setUp(self):
        saved = {name: sys.modules[name] for name in _module_names()}
        self.addCleanup(self._restore, saved)

    @staticmethod
    def _restore(saved):
        for name in _module_names():
            del sys.modules[name]
        sys.modules.update(saved)

    def fresh_onnx_asr(self):
        """Import an unpatched onnx-asr and return its loader and resolver modules."""
        for name in _module_names():
            del sys.modules[name]
        loader = importlib.import_module("onnx_asr.loader")
        resolver = importlib.import_module("onnx_asr.resolver")
        # This must be the installed onnx-asr, never the stub the suite shares.
        self.assertTrue(callable(loader.create_asr_resolver), loader)
        self.assertIsInstance(resolver.model_repos, dict)
        self.assertTrue(hasattr(resolver.Resolver, "resolve_model"), resolver)
        return loader, resolver

    @staticmethod
    def ensure_model_types(*args, **kwargs):
        from ovos_stt_plugin_onnxasr._shims import ensure_model_types

        return ensure_model_types(*args, **kwargs)

    @staticmethod
    def model_types(loader) -> dict:
        """Return the model type mapping the resolver factory hands to Resolver."""
        captured: dict = {}
        real_resolver = loader.Resolver

        class _Spy:
            def __init__(self, model_types, *args, **kwargs):
                captured.update(model_types)

        loader.Resolver = _Spy
        try:
            loader.create_asr_resolver(BUILTIN_MODEL)
        except Exception:
            pass
        finally:
            loader.Resolver = real_resolver
        return captured


class TestResolverSpyNeverEscapes(RealOnnxAsrTestCase):
    def test_concurrent_callers_always_get_a_resolver(self):
        """No caller may ever receive anything but a Resolver.

        A registration that swaps ``loader.Resolver`` while another thread is inside
        the resolver factory hands that thread the replacement object. The caller
        then fails with an AttributeError somewhere else entirely.
        """
        loader, resolver_mod = self.fresh_onnx_asr()
        self.ensure_model_types()

        resolver_cls = resolver_mod.Resolver
        wrong_types = []
        errors = []

        def worker():
            for _ in range(400):
                try:
                    resolver = loader.create_asr_resolver(BUILTIN_MODEL)
                except Exception as err:  # noqa: BLE001 - the test reports it
                    errors.append(repr(err))
                    return
                if not isinstance(resolver, resolver_cls):
                    wrong_types.append(type(resolver).__name__)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(wrong_types, [], f"{len(wrong_types)} callers got a non-Resolver object")
        self.assertEqual(errors, [])

    def test_registration_while_the_factory_runs_is_safe(self):
        """Registration must not disturb a resolver build that is already running."""
        loader, resolver_mod = self.fresh_onnx_asr()

        resolver_cls = resolver_mod.Resolver
        wrong_types = []
        errors = []
        stop = threading.Event()

        def register():
            while not stop.is_set():
                self.ensure_model_types()

        def build():
            for _ in range(400):
                try:
                    resolver = loader.create_asr_resolver(BUILTIN_MODEL)
                except Exception as err:  # noqa: BLE001 - the test reports it
                    errors.append(repr(err))
                    return
                if not isinstance(resolver, resolver_cls):
                    wrong_types.append(type(resolver).__name__)

        registrars = [threading.Thread(target=register) for _ in range(2)]
        builders = [threading.Thread(target=build) for _ in range(6)]
        for thread in registrars + builders:
            thread.start()
        for thread in builders:
            thread.join()
        stop.set()
        for thread in registrars:
            thread.join()

        self.assertEqual(wrong_types, [], f"{len(wrong_types)} callers got a non-Resolver object")
        self.assertEqual(errors, [])


class TestSingleResolution(RealOnnxAsrTestCase):
    def test_create_asr_builds_one_resolver(self):
        """A model load resolves the model once, as onnx-asr does."""
        loader, resolver_mod = self.fresh_onnx_asr()
        self.ensure_model_types()

        built = []
        real_resolver = resolver_mod.Resolver

        class _Counting(real_resolver):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                built.append(1)

            def resolve_model(self, **kwargs):
                raise _Stop

        loader.Resolver = _Counting
        resolver_mod.Resolver = _Counting
        self.addCleanup(setattr, resolver_mod, "Resolver", real_resolver)
        self.addCleanup(setattr, loader, "Resolver", real_resolver)

        with self.assertRaises(_Stop):
            loader.Manager().create_asr(BUILTIN_MODEL)

        self.assertEqual(len(built), 1, f"{len(built)} resolvers built for one model load")


class TestAdditiveRegistration(RealOnnxAsrTestCase):
    def test_a_later_call_adds_the_families_the_first_call_left_out(self):
        loader, _ = self.fresh_onnx_asr()
        self.ensure_model_types()
        complete = set(self.model_types(loader))

        loader, _ = self.fresh_onnx_asr()
        self.ensure_model_types(only={"moonshine"})
        self.ensure_model_types()
        after_two_calls = set(self.model_types(loader))

        self.assertEqual(complete - after_two_calls, set())

    def test_ensure_wav2vec2_ctc_does_not_lock_out_the_other_families(self):
        loader, _ = self.fresh_onnx_asr()
        self.ensure_model_types()
        complete = set(self.model_types(loader))

        loader, _ = self.fresh_onnx_asr()
        from ovos_stt_plugin_onnxasr._compat import ensure_wav2vec2_ctc

        ensure_wav2vec2_ctc()
        self.ensure_model_types()

        self.assertEqual(complete - set(self.model_types(loader)), set())


class TestCorePatchesFollowTheFamilies(RealOnnxAsrTestCase):
    def test_no_core_patch_when_no_model_type_is_added(self):
        """onnx-asr keeps its own core behaviour when the plugin adds nothing.

        The resolver factory here already knows every model type the plugin
        carries, which is what a later onnx-asr release looks like. Nothing is
        registered, so nothing may be patched.
        """
        loader, resolver_mod = self.fresh_onnx_asr()

        complete = dict(self.model_types(loader))
        for name in self.model_types_of_every_family():
            complete.setdefault(name, object)

        def create_asr_resolver(model=None, local_dir=None, *, offline=None):
            return loader.Resolver(dict(complete), model, local_dir, offline=offline)

        loader.create_asr_resolver = create_asr_resolver

        original_create_asr = loader.Manager.create_asr
        original_preprocessor = loader.Manager._create_preprocessor
        original_download = resolver_mod.Resolver._download_model
        original_resolve_files = resolver_mod.Resolver._resolve_model_files
        had_fetch = hasattr(resolver_mod.Resolver, "fetch")

        self.ensure_model_types()

        self.assertIs(loader.Manager.create_asr, original_create_asr)
        self.assertIs(loader.Manager._create_preprocessor, original_preprocessor)
        self.assertIs(resolver_mod.Resolver._download_model, original_download)
        self.assertIs(resolver_mod.Resolver._resolve_model_files, original_resolve_files)
        self.assertEqual(hasattr(resolver_mod.Resolver, "fetch"), had_fetch)

    def test_core_patches_arrive_with_the_family_that_needs_them(self):
        """The fetcher patch belongs to wav2vec2-adapters only."""
        loader, resolver_mod = self.fresh_onnx_asr()
        original_create_asr = loader.Manager.create_asr

        self.ensure_model_types(only={"sensevoice"})
        self.assertIs(loader.Manager.create_asr, original_create_asr)

        self.ensure_model_types(only={"wav2vec2-adapters"})
        from ovos_stt_plugin_onnxasr._shims import _is_native

        if _is_native("onnx_asr.models.wav2vec2_adapters"):
            # onnx-asr carries the family; the plugin must stay out of the way
            self.assertIs(loader.Manager.create_asr, original_create_asr)
        else:
            self.assertIsNot(loader.Manager.create_asr, original_create_asr)

    @staticmethod
    def model_types_of_every_family() -> list:
        from ovos_stt_plugin_onnxasr._shims import _families

        names = []
        for family in _families():
            names.extend(family.build())
        return names


class TestModelRepositories(RealOnnxAsrTestCase):
    def test_every_short_name_resolves_to_its_repository(self):
        """A model type name alone must be enough to load a model."""
        loader, resolver_mod = self.fresh_onnx_asr()
        self.ensure_model_types()

        from ovos_stt_plugin_onnxasr._shims._model_repos import MODEL_REPOS

        self.assertTrue(MODEL_REPOS)
        for name, repo in MODEL_REPOS.items():
            with self.subTest(model_type=name):
                self.assertEqual(resolver_mod.model_repos.get(name), repo)
                # a repository id, and no download: the name alone loads a model
                self.assertEqual(loader.create_asr_resolver(name).repo_id, repo)

    def test_the_verified_short_names_keep_their_repositories(self):
        loader, resolver_mod = self.fresh_onnx_asr()
        self.ensure_model_types()

        for name, repo in VERIFIED_REPOS.items():
            with self.subTest(model_type=name):
                self.assertEqual(resolver_mod.model_repos.get(name), repo)
                self.assertEqual(loader.create_asr_resolver(name).repo_id, repo)

    def test_a_repository_onnx_asr_knows_is_never_replaced(self):
        _, resolver_mod = self.fresh_onnx_asr()
        resolver_mod.model_repos["sensevoice"] = "someone/else-onnx"

        self.ensure_model_types()

        self.assertEqual(resolver_mod.model_repos["sensevoice"], "someone/else-onnx")


if __name__ == "__main__":
    unittest.main()
