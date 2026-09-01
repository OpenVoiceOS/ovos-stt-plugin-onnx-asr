"""Every model in the registry must be loadable.

The registry picks the model for a language nobody configured. An entry that
points at a repository without the files the loader asks for takes that language
away: the load raises, and the request falls back to a model for another
language. The user hears a wrong transcript and no error.

These tests read the Hugging Face API only — the repository file list and
``config.json``, a few kilobytes per repository. No weights are downloaded.

For each entry they check that:

* the repository exists and holds a ``config.json``;
* ``config.json`` declares a ``model_type`` that this plugin serves, counting the
  model types the plugin registers itself;
* the repository holds every file the model type asks for, for the quantization
  the plugin loads it with.

They also keep :data:`FP32_ONLY_MODELS` equal to the repositories that really
hold no quantized weights, in both directions, so the fp32 fallback covers every
model that needs it and no model that does not.
"""
import fnmatch
import json
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from test_real_onnx_asr_shims import RealOnnxAsrTestCase

from ovos_stt_plugin_onnxasr.defaults import (FP32_ONLY_MODELS, LANG_DEFAULTS,
                                              quantization_for)

_API = "https://huggingface.co/api/models/"
_FILES = "https://huggingface.co/{repo}/resolve/main/{name}"
_ATTEMPTS = 3
_TIMEOUT = 30


def _get_json(url: str):
    last = None
    for _ in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            if err.code < 500:
                raise
            last = err
        except Exception as err:  # transport hiccup
            last = err
    raise AssertionError(f"the Hugging Face API did not answer for {url}: {last}")


def _repo_facts(repo: str):
    """Return (file list, config) for a Hugging Face repository."""
    info = _get_json(_API + repo)
    files = [sibling["rfilename"] for sibling in info.get("siblings", [])]
    config = None
    if "config.json" in files:
        config = _get_json(_FILES.format(repo=repo, name="config.json"))
    return files, config


def _holds(files, pattern: str) -> bool:
    """Say whether ``files`` holds a file the loader's pattern matches.

    The loader globs a downloaded directory, so ``**/`` means any depth and a
    bare name means the top level. A repository path is matched both whole and
    by its last part.
    """
    bare = pattern.removeprefix("**/")
    return any(fnmatch.fnmatch(name, pattern)
               or fnmatch.fnmatch(name, bare)
               or fnmatch.fnmatch(name.rsplit("/", 1)[-1], bare)
               for name in files)


# Registry models that are named, not repository ids. onnx-asr owns these and
# resolves them itself, so there is no repository to read here.
_SHORT_NAMES = {model for model in LANG_DEFAULTS.values() if "/" not in model}

_REPOS = sorted({model for model in LANG_DEFAULTS.values()} - _SHORT_NAMES)


def _fetch_all():
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(_REPOS, pool.map(_repo_facts, _REPOS)))


class TestRegistryModelsResolvable(RealOnnxAsrTestCase):
    """Reads the model type table of the real onnx-asr, plus the plugin's own.

    The suite shares a stub for onnx-asr, and a stub has no model files to ask
    about. The base class swaps the stub for the installed onnx-asr for the
    length of a test and puts the stub back afterwards.
    """

    facts = None

    def setUp(self):
        super().setUp()
        loader, _ = self.fresh_onnx_asr()
        self.ensure_model_types()
        self.types = self.model_types(loader)
        if TestRegistryModelsResolvable.facts is None:
            TestRegistryModelsResolvable.facts = _fetch_all()

    def _langs_of(self, model: str):
        return sorted(lang for lang, m in LANG_DEFAULTS.items() if m == model)

    def test_every_repo_declares_a_model_type_the_plugin_serves(self):
        for repo in _REPOS:
            with self.subTest(repo=repo, langs=self._langs_of(repo)):
                _, config = self.facts[repo]
                self.assertIsNotNone(config, f"{repo} holds no config.json")
                model_type = config.get("model_type")
                self.assertIn(model_type, self.types,
                              f"{repo} declares model_type {model_type!r}, "
                              f"which this plugin does not serve")

    def test_every_repo_holds_the_files_the_loader_asks_for(self):
        for repo in _REPOS:
            files, config = self.facts[repo]
            model_class = self.types[config["model_type"]]
            quantization = quantization_for(repo, "int8")
            with self.subTest(repo=repo, langs=self._langs_of(repo),
                              quantization=quantization):
                wanted = model_class._get_model_files(quantization).values()
                missing = [name for name in wanted if not _holds(files, name)]
                self.assertEqual(missing, [], f"{repo} holds no {missing}")

    def test_every_repo_holds_fp32_weights(self):
        for repo in _REPOS:
            files, config = self.facts[repo]
            model_class = self.types[config["model_type"]]
            with self.subTest(repo=repo, langs=self._langs_of(repo)):
                wanted = model_class._get_model_files(None).values()
                missing = [name for name in wanted if not _holds(files, name)]
                self.assertEqual(missing, [], f"{repo} holds no {missing}")

    def test_fp32_only_set_matches_the_repositories(self):
        measured = set()
        for repo in _REPOS:
            files, config = self.facts[repo]
            model_class = self.types[config["model_type"]]
            wanted = model_class._get_model_files("int8").values()
            if any(not _holds(files, name) for name in wanted):
                measured.add(repo)
        self.assertEqual(
            measured, set(FP32_ONLY_MODELS),
            "FP32_ONLY_MODELS no longer says what the repositories hold; "
            f"add {sorted(measured - set(FP32_ONLY_MODELS))} and "
            f"drop {sorted(set(FP32_ONLY_MODELS) - measured)}")


if __name__ == "__main__":
    unittest.main()
