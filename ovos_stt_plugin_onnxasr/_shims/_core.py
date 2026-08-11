"""Runtime patches for onnx-asr core behaviours the vendored model types need.

The vendored model classes in :mod:`._vendored` come from a fork of onnx-asr. Some
of them use core behaviours that a released onnx-asr does not have yet. This module
adds those behaviours to the installed onnx-asr at runtime.

Every patch here is:

* **Self-disabling** — it checks for the behaviour first and does nothing when the
  installed onnx-asr already has it. The installed onnx-asr always wins.
* **Idempotent** — a second call does nothing.
* **Additive** — it adds a new branch and keeps the behaviour of every other case,
  so built-in model types work as before.

Each patch raises on failure. The caller registers one model family at a time and
skips the family whose patch failed. A patch runs only when the family that needs
it really is registered.

:func:`patch_fetcher_support` is the one patch that reaches past its own family: it
replaces ``Manager.create_asr``, which builds every ASR model. It keeps the
behaviour of every model type without fetcher support, and it resolves the model
once, as ``Manager.create_asr`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ovos_utils.log import LOG

_SENTINEL = "_ovos_shim_patched"


class FileFetcher(Protocol):
    """Fetches one model file on demand and returns its local path."""

    def __call__(self, filename: str) -> Path:
        """Download `filename` from the model repository if it is not local yet."""
        ...


# --------------------------------------------------------------------------------
# w2v-BERT preprocessor
# --------------------------------------------------------------------------------

def _w2vbert_melscale_fbanks() -> np.ndarray:
    """Build the w2v-BERT 2.0 mel filterbank matrix.

    The released onnx-asr ships a ``fbanks.npz`` without a ``w2vbert`` entry. The
    matrix is a pure function of the feature extractor settings, so it is computed
    here with the same code and the same settings that build the shipped file.
    """
    from ._vendored.fbanks import melscale_fbanks

    # Settings of preprocessors/w2vbert.py: 80 Kaldi mel bins, n_fft 512, 16 kHz.
    return melscale_fbanks(512 // 2 + 1, 20, 0, 80, 16_000, mel_scale="kaldi").astype(np.float32)


def _numpy_preprocessor_base() -> type:
    """Return a ``_NumpyPreprocessor`` that also knows the ``w2vbert`` filterbank."""
    from onnx_asr.preprocessors.numpy_preprocessor import _NumpyPreprocessor as _Upstream

    class _NumpyPreprocessorWithW2vBert(_Upstream):  # type: ignore[misc, valid-type]
        def __init__(self, name: str):
            try:
                super().__init__(name)
            except KeyError:
                if name != "w2vbert":
                    raise
                self._melscale_fbanks = _w2vbert_melscale_fbanks()

    return _NumpyPreprocessorWithW2vBert


def __getattr__(name: str):
    # `_vendored.w2vbert_preprocessor` subclasses `_NumpyPreprocessor` at import time.
    if name == "_NumpyPreprocessor":
        return _numpy_preprocessor_base()
    raise AttributeError(name)


def patch_w2vbert_preprocessor() -> None:
    """Make ``Manager._create_preprocessor`` build the ``w2vbert`` preprocessor.

    Does nothing when the installed onnx-asr already builds it.
    """
    import onnx_asr.loader as loader

    manager = loader.Manager
    original = manager._create_preprocessor
    if getattr(original, _SENTINEL, False):
        return
    try:
        from onnx_asr.preprocessors.numpy_preprocessor import W2vBertPreprocessorNumpy  # noqa: F401

        return  # installed onnx-asr provides it -> leave it alone
    except ImportError:
        pass

    from ._vendored.w2vbert_preprocessor import W2vBertPreprocessorNumpy

    def _create_preprocessor(self, name: str):
        if name == "w2vbert":
            # There is no ONNX preprocessor graph for w2v-BERT features, always use NumPy.
            return W2vBertPreprocessorNumpy(name)
        return original(self, name)

    setattr(_create_preprocessor, _SENTINEL, True)
    manager._create_preprocessor = _create_preprocessor
    LOG.debug("onnx-asr: added w2vbert preprocessor")


# --------------------------------------------------------------------------------
# On-demand file fetching (lazily fetched asset directories)
# --------------------------------------------------------------------------------

LAZY_DIR_SUFFIX = "/*"
"""A `_get_model_files` value that ends with this names a directory whose members are fetched on demand."""


def _is_lazy_dir(filename: str) -> bool:
    return filename.endswith(LAZY_DIR_SUFFIX)


def patch_fetcher_support() -> None:
    """Add on-demand file fetching to the resolver and the model manager.

    A model type with ``_supports_fetcher = True`` names an asset directory whose
    members are downloaded one by one. Does nothing when the installed onnx-asr
    already supports the fetcher.

    ``Manager.create_asr`` builds every ASR model, so the replacement here is on the
    path of every model type. It keeps the behaviour of a model type that has no
    fetcher support.
    """
    import onnx_asr.loader as loader
    import onnx_asr.resolver as resolver_mod
    from onnx_asr.asr import BaseAsr
    from onnx_asr.utils import ModelFileNotFoundError, MoreThanOneModelFileFoundError

    if hasattr(resolver_mod.Resolver, "fetch"):
        return  # installed onnx-asr provides it -> leave it alone
    if getattr(loader.Manager.create_asr, _SENTINEL, False):
        return

    if not hasattr(BaseAsr, "_supports_fetcher"):
        BaseAsr._supports_fetcher = False

    resolver_cls = resolver_mod.Resolver
    original_download = resolver_cls._download_model
    original_create_asr = loader.Manager.create_asr

    def _find_dir(path: Path, filename: str) -> Path:
        """Resolve a directory of model assets."""
        if _is_lazy_dir(filename):
            # Members arrive one by one through `fetch`, so an empty or missing
            # directory is normal.
            return Path(path, filename.removesuffix(LAZY_DIR_SUFFIX))
        directory = Path(path, filename)
        if not directory.is_dir():
            raise ModelFileNotFoundError(filename, path)
        return directory

    def _download_model(self, quantization, *, local_files_only):
        if not any(_is_lazy_dir(file) for file in self.model_type._get_model_files(quantization).values()):
            return original_download(self, quantization, local_files_only=local_files_only)

        from huggingface_hub import snapshot_download

        # A lazily fetched directory is not part of the snapshot: drop it from the
        # allow list. Its members arrive one by one through `fetch`.
        files = [file for file in self.model_type._get_model_files(quantization).values() if not _is_lazy_dir(file)]
        files = [
            *files,
            *(file.removeprefix("**/") for file in files if file.startswith("**/")),
        ]
        # a value that ends with "/" names a directory of model assets
        files = [file + "**" if file.endswith("/") else file for file in files]
        files = [
            "config.json",
            "config.yaml",
            *files,
            *(str(path.with_suffix(".onnx?data")) for file in files if (path := Path(file)).suffix == ".onnx"),
        ]

        assert self.repo_id is not None
        return Path(
            snapshot_download(
                self.repo_id, local_dir=self.local_dir, local_files_only=local_files_only, allow_patterns=files
            )
        )

    def _resolve_model_files(self, path: Path, quantization):
        files = self.model_type._get_model_files(quantization)
        if Path(path, "config.json").exists():
            files |= {"config": "config.json"}

        def find(filename: str) -> Path:
            if _is_lazy_dir(filename) or filename.endswith("/"):
                return _find_dir(path, filename)

            found = list(path.glob(filename))
            if len(found) > 1:
                raise MoreThanOneModelFileFoundError(filename, path)
            if len(found) == 0:
                orig_path = Path(filename)
                if orig_path.suffix == ".onnx":
                    found = list(path.glob(str(orig_path.with_suffix(".ort"))))
                    if len(found) == 1 and found[0].is_file():
                        return found[0]
                raise ModelFileNotFoundError(filename, path)
            if not found[0].is_file():
                raise ModelFileNotFoundError(filename, path)
            return found[0]

        return {key: find(filename) for key, filename in files.items()}

    def fetch(self, filename: str) -> Path:
        """Resolve one model file, downloading it if the model repository is remote."""
        if self.local_dir is not None and (local_path := Path(self.local_dir, filename)).is_file():
            return local_path

        if self.repo_id is None:
            assert self.local_dir is not None
            raise ModelFileNotFoundError(filename, self.local_dir)

        from huggingface_hub import hf_hub_download

        try:
            return Path(hf_hub_download(self.repo_id, filename, local_dir=self.local_dir, local_files_only=True))
        except FileNotFoundError:
            if self.offline:
                raise
            return Path(hf_hub_download(self.repo_id, filename, local_dir=self.local_dir))

    known_arguments = {"quantization", "offline", "config"}

    def create_asr(self, model=None, local_dir=None, **kwargs):
        """Build an ASR model, and give a fetcher to a model type that takes one.

        The model is resolved once, for every model type. A model type without
        fetcher support gets the behaviour it had.
        """
        if not known_arguments.issuperset(kwargs):
            # An onnx-asr release added an argument. Give the whole call to
            # onnx-asr, which knows what the argument means.
            LOG.warning("onnx-asr create_asr has an unknown argument; on-demand file fetching is not available")
            return original_create_asr(self, model, local_dir, **kwargs)

        quantization = kwargs.get("quantization")
        config = kwargs.get("config")
        resolver = loader.create_asr_resolver(model, local_dir, offline=kwargs.get("offline"))
        if config is None:
            config = loader.update_onnx_providers(
                self.default_onnx_config, excluded_providers=resolver.model_type._get_excluded_providers()
            )
        extra = {"fetcher": resolver.fetch} if getattr(resolver.model_type, "_supports_fetcher", False) else {}
        return self._create_asr_adapter(
            resolver.model_type(
                resolver.resolve_model(quantization=quantization),
                self._create_preprocessor,
                config,
                **extra,
            )
        )

    setattr(create_asr, _SENTINEL, True)
    resolver_cls.fetch = fetch
    resolver_cls._download_model = _download_model
    resolver_cls._resolve_model_files = _resolve_model_files
    loader.Manager.create_asr = create_asr
    LOG.debug("onnx-asr: added on-demand model file fetching")


# --------------------------------------------------------------------------------
# Model repositories
# --------------------------------------------------------------------------------

def add_model_repos(repos: dict) -> None:
    """Map short model names to Hugging Face repositories.

    Never replaces a mapping the installed onnx-asr already has.
    """
    import onnx_asr.resolver as resolver_mod

    for name, repo in repos.items():
        resolver_mod.model_repos.setdefault(name, repo)
