# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/wav2vec2_adapters.py
#
# MIT License. Copyright (c) 2025 Ilya Stupakov and onnx-asr contributors.
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Only import lines are adjusted. Do not edit the code below; re-sync it instead.
"""Wav2Vec2 CTC model with per-language adapter packs.

One shared base graph holds every language-independent weight. The adapter layers
and the CTC head are graph *inputs*, not initializers, so a single session serves
all languages. Each language adds only a small `.npz` pack and a vocabulary file.
This is how `facebook/mms-1b-all` ships more than a thousand languages without a
full 3.6 GB export per language.

The active language is per call (`recognize(..., language="lg")`). Like the other
model classes here, one instance is not safe to share between threads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, TimestampedResult, _AsrWithCtcDecoding

from .._core import FileFetcher
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import ModelFileNotFoundError, is_float32_array, is_int64_array

BLANK_TOKEN = "<blk>"
VOCAB_SIZE_KEY = "vocab_size"
MAX_LANGUAGES_IN_MESSAGE = 20


def _language_list(languages: list[str]) -> str:
    """Join the language codes, short enough to read when the model has more than a thousand."""
    if len(languages) > MAX_LANGUAGES_IN_MESSAGE:
        head = ", ".join(languages[:MAX_LANGUAGES_IN_MESSAGE])
        return f"{head} and {len(languages) - MAX_LANGUAGES_IN_MESSAGE} more"
    return ", ".join(languages)


class LanguageNotFoundError(ValueError):
    """Raised when no adapter pack matches the requested language."""

    def __init__(self, language: str, available: list[str]):
        """Create error with the list of available languages."""
        super().__init__(
            f"Language '{language}' has no adapter pack. Available languages: {_language_list(available)}."
        )


class LanguageNotSpecifiedError(ValueError):
    """Raised when no language was given and the model has no default language."""

    def __init__(self, available: list[str]):
        """Create error with the list of available languages."""
        super().__init__(
            "This model needs a language. Pass language=<code> to recognize, or set 'default_language' "
            f"in config.json. Available languages: {_language_list(available)}."
        )


class MissingAdapterInputsError(ValueError):
    """Raised when an adapter pack does not cover every adapter input of the graph."""

    def __init__(self, language: str, missing: list[str]):
        """Create error with the list of missing graph inputs."""
        super().__init__(f"Adapter pack '{language}' does not contain: {', '.join(missing)}.")


@dataclass
class _LanguagePack:
    """Everything that makes the shared base graph speak one language."""

    language: str
    tensors: dict[str, npt.NDArray[np.float32]]
    vocab: dict[int, str]
    blank_idx: int
    vocab_size: int


class Wav2Vec2Adapters(_AsrWithCtcDecoding):
    """Wav2Vec2 CTC model with a shared base graph and per-language adapter packs."""

    _supports_fetcher = True

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
        fetcher: FileFetcher | None = None,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

        audio_inputs = {"input_values", "input_lengths"}
        self._adapter_inputs = [i.name for i in self._model.get_inputs() if i.name not in audio_inputs]

        self._fetcher = fetcher
        self._adapters_dir = Path(model_files["adapters"])
        self._vocabs_dir = Path(model_files["vocabs"])
        self._adapter_files = {path.stem: path for path in sorted(self._adapters_dir.glob("*.npz"))}
        self._vocab_files = {path.stem.removesuffix("_vocab"): path for path in sorted(self._vocabs_dir.glob("*.txt"))}
        # With a lazily fetched repository nothing is on disk yet, so config.json names the packs.
        self._languages = sorted(set(self.config.get("languages", [])) | set(self._adapter_files))
        self._aliases = self.config.get("language_aliases", {})
        self._packs: dict[str, _LanguagePack] = {}
        self._default: _LanguagePack | None = None

        default_language = self.config.get("default_language")
        if default_language is None and len(self._languages) == 1:
            default_language = self._languages[0]
        if default_language is not None:
            self._default = self._load_pack(self._resolve_language(default_language))
        self._pack = self._default

        self.preload(*self.config.get("preload_languages", []))

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        # The pack directories are fetched per language, not as a whole: the repository holds
        # more than a thousand of them.
        return {"model": f"model{suffix}.onnx", "adapters": "adapters/*", "vocabs": "vocabs/*"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 320))

    @property
    def languages(self) -> list[str]:
        """Languages with an adapter pack, as ISO codes."""
        return list(self._languages)

    def preload(self, *languages: str) -> None:
        """Load the packs of these languages now, so the first call for them does not wait."""
        for language in languages:
            self._load_pack(self._resolve_language(language))

    def _resolve_language(self, language: str) -> str:
        """Map a language tag to an adapter pack name.

        Accepts the pack name itself, a BCP-47 tag whose primary subtag is a pack
        name (`lg-UG`), or an alias from `language_aliases` in config.json (`pt` for
        `por`), also with a region subtag.
        """
        known = set(self._languages)
        for candidate in (language, language.replace("_", "-").split("-")[0]):
            if candidate in known:
                return candidate
            alias = self._aliases.get(candidate.lower())
            if alias in known:
                assert alias is not None
                return alias
        raise LanguageNotFoundError(language, self.languages)

    def _adapter_file(self, language: str) -> Path:
        if language not in self._adapter_files:
            self._adapter_files[language] = self._fetch(f"{self._adapters_dir.name}/{language}.npz")
        return self._adapter_files[language]

    def _vocab_file(self, language: str) -> Path:
        if language not in self._vocab_files:
            self._vocab_files[language] = self._fetch(f"{self._vocabs_dir.name}/{language}.txt")
        return self._vocab_files[language]

    def _fetch(self, filename: str) -> Path:
        if self._fetcher is None:
            raise ModelFileNotFoundError(filename, self._adapters_dir.parent)
        return self._fetcher(filename)

    def _load_pack(self, language: str) -> _LanguagePack:
        if language in self._packs:
            return self._packs[language]

        with np.load(self._adapter_file(language)) as data:
            tensors = {key: data[key].astype(np.float32) for key in data.files if key != VOCAB_SIZE_KEY}
            vocab_size = int(data[VOCAB_SIZE_KEY]) if VOCAB_SIZE_KEY in data.files else 0

        if missing := [name for name in self._adapter_inputs if name not in tensors]:
            raise MissingAdapterInputsError(language, missing)
        tensors = {name: tensors[name] for name in self._adapter_inputs}

        with self._vocab_file(language).open("rt", encoding="utf-8") as f:
            vocab = {int(id): token.replace("▁", " ") for token, id in (line.strip("\n").split(" ") for line in f)}

        blank_idx = next(id for id, token in vocab.items() if token == BLANK_TOKEN)
        pack = _LanguagePack(language, tensors, vocab, blank_idx, vocab_size or len(vocab))
        self._packs[language] = pack
        return pack

    def _active_pack(self) -> _LanguagePack:
        if self._pack is None:
            raise LanguageNotSpecifiedError(self.languages)
        return self._pack

    @property
    def _vocab(self) -> dict[int, str]:  # type: ignore[override]
        return self._active_pack().vocab

    @property
    def _vocab_size(self) -> int:  # type: ignore[override]
        return self._active_pack().vocab_size

    @property
    def _blank_idx(self) -> int:  # type: ignore[override]
        return self._active_pack().blank_idx

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        pack = self._active_pack()
        inputs: dict[str, npt.NDArray[np.float32] | npt.NDArray[np.int64]] = {
            "input_values": waveforms,
            "input_lengths": waveforms_len.astype(np.int64),
        }
        inputs |= pack.tensors

        (logprobs,) = self._model.run(["logprobs"], inputs)
        assert is_float32_array(logprobs)
        if logprobs.shape[-1] > pack.vocab_size:  # base graph padded to the largest vocabulary
            logprobs = np.ascontiguousarray(logprobs[..., : pack.vocab_size])

        out_lens = waveforms_len // self._subsampling_factor + 1
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize a batch of waveforms in the language given by `language`.

        Without `language`, the model falls back to `default_language`. The choice
        never carries over to the next call.
        """
        language = kwargs.pop("language", None)
        if language is None:
            self._pack = self._default
        else:
            assert isinstance(language, str)
            self._pack = self._load_pack(self._resolve_language(language))
        self._active_pack()
        return super().recognize_batch(waveforms, waveforms_len, **kwargs)
