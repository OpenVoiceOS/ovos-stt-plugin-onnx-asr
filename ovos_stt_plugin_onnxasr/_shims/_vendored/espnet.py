# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/espnet.py
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
"""ESPnet E-Branchformer model implementations."""

import typing
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, _AsrWithCtcDecoding, _AsrWithDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array

LANGUAGE_PLACEHOLDER = "{language}"
REGION_PLACEHOLDER = "{region}"


class UnknownVocabTokenError(ValueError):
    """Raised when config.json names a token that the vocabulary does not have."""

    def __init__(self, token: str):
        """Create error for one missing token."""
        super().__init__(f"Token '{token}' is not in vocab.txt.")


class LanguageNotFoundError(ValueError):
    """Raised when the requested language tag matches no language of the model."""

    def __init__(self, language: str, available: list[str]):
        """Create error with the list of known language tags."""
        super().__init__(f"Language '{language}' is not supported. Available languages: {', '.join(available)}.")


class _Espnet(_AsrWithDecoding):
    """Common parts of the ESPnet ASR model implementations."""

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", "w2vbert"))

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", 8))


class EspnetCtc(_Espnet, _AsrWithCtcDecoding):
    """ESPnet E-Branchformer CTC model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt", "config": "config.json"}

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        logprobs, logprobs_lens = self._model.run(
            ["logprobs", "logprobs_lens"], {"features": features, "features_lens": features_lens.astype(np.int64)}
        )
        assert is_float32_array(logprobs)
        assert is_int64_array(logprobs_lens)
        return logprobs, np.minimum(logprobs_lens, logprobs.shape[1])


class EspnetAED(_Espnet):
    """ESPnet E-Branchformer attention decoder model implementation.

    The encoder graph and the transformer decoder graph are separate files. Decoding is
    greedy and recomputes the decoder over the whole prefix at every step, so the decoder
    graph needs no key-value cache inputs.

    Decoding starts from `sos_token` (`<sos/eos>` unless config.json says otherwise) and
    stops at `eos_token`. Models of the OWSM family put a task prompt between the two:
    `prompt_tokens` in config.json lists that prompt, one entry per decode step. An entry
    is a vocabulary token that is forced, or one of the placeholders `{language}` and
    `{region}`, which the `language` argument of `recognize` fills in. A placeholder with
    nothing to fill it in is left to the model, which then predicts the slot itself.
    Without `prompt_tokens` the prompt is empty and decoding is plain `<sos/eos>` to
    `<sos/eos>`.
    """

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)
        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        self._token_ids = {token: id for id, token in self._vocab.items()}
        sos_token = str(config.get("sos_token", "<sos/eos>"))
        self._sos_token_id = self._token_id(sos_token)
        self._eos_token_id = self._token_id(str(config.get("eos_token", sos_token)))

        self._prompt_tokens: list[str] = list(config.get("prompt_tokens", []))
        for token in self._prompt_tokens:
            if token not in (LANGUAGE_PLACEHOLDER, REGION_PLACEHOLDER):
                self._token_id(token)
        self._languages: list[str] = list(config.get("languages", []))
        self._language_aliases: dict[str, str] = dict(config.get("language_aliases", {}))
        self._default_language: str | None = config.get("default_language")

    def _token_id(self, token: str) -> int:
        if token not in self._token_ids:
            raise UnknownVocabTokenError(token)
        return self._token_ids[token]

    @property
    def languages(self) -> list[str]:
        """Language tags the model was published with."""
        return self._languages

    def _resolve_language(self, language: str) -> str:
        """Map a language argument to one of the language tags of the model.

        Accepts a full tag (`zh-CN`), the same tag with an underscore (`zh_CN`), or a
        bare language (`zh`), which `language_aliases` maps to a full tag.
        """
        tag = language.replace("_", "-")
        for candidate in (tag, tag.split("-")[0]):
            if candidate in self._languages:
                return candidate
            if candidate in self._language_aliases:
                return self._language_aliases[candidate]
        raise LanguageNotFoundError(language, self._languages)

    def _prompt(self, language: str | None) -> list[int | None]:
        """Build the decode prompt: a forced token id per step, or None to let the model predict."""
        if not self._prompt_tokens:
            return []

        requested = language or self._default_language
        parts = self._resolve_language(requested).split("-", 1) if requested else []
        fills = {
            LANGUAGE_PLACEHOLDER: parts[0] if parts else None,
            REGION_PLACEHOLDER: parts[1] if len(parts) > 1 else None,
        }

        prompt: list[int | None] = []
        for token in self._prompt_tokens:
            if token in fills:
                value = fills[token]
                prompt.append(self._token_id(f"<{value}>") if value else None)
            else:
                prompt.append(self._token_id(token))
        return prompt

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"encoder{suffix}.onnx",
            "decoder": f"decoder{suffix}.onnx",
            "vocab": "vocab.txt",
            "config": "config.json",
        }

    @property
    def _max_sequence_length(self) -> int:
        return int(self.config.get("max_sequence_length", 200))

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = self._encoder.run(
            ["encoder_out", "encoder_out_lens"], {"features": features, "features_lens": features_lens.astype(np.int64)}
        )
        assert is_float32_array(encoder_out)
        assert is_int64_array(encoder_out_lens)
        return encoder_out, np.minimum(encoder_out_lens, encoder_out.shape[1])

    def _decode(
        self,
        tokens: npt.NDArray[np.int64],
        encoder_out: npt.NDArray[np.float32],
        encoder_out_lens: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        (logprobs,) = self._decoder.run(
            ["logprobs"],
            {"tokens": tokens, "encoder_out": encoder_out, "encoder_out_lens": encoder_out_lens},
        )
        assert is_float32_array(logprobs)
        return logprobs

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], None, Iterable[float]]]:
        language = kwargs.get("language")
        prompt = self._prompt(str(language) if language is not None else None)

        batch_size = encoder_out.shape[0]
        batch_tokens = np.full((batch_size, 1), self._sos_token_id, dtype=np.int64)
        batch_logprobs = np.zeros((batch_size, 0), dtype=np.float32)
        finished = np.zeros(batch_size, dtype=bool)
        max_length = min(self._max_sequence_length, int(max(encoder_out_lens))) + len(prompt)

        while batch_tokens.shape[1] <= max_length:
            logprobs = self._decode(batch_tokens, encoder_out, encoder_out_lens)

            step = batch_tokens.shape[1] - 1
            forced = prompt[step] if step < len(prompt) else None
            if forced is not None:
                next_tokens = np.full(batch_size, forced, dtype=np.int64)
            else:
                next_tokens = np.argmax(logprobs[:, -1], axis=-1)

            if step >= len(prompt):
                next_tokens[finished] = self._eos_token_id
                finished |= next_tokens == self._eos_token_id
                if finished.all():
                    break

            next_logprobs = np.take_along_axis(logprobs[:, -1], next_tokens[:, None], axis=-1).squeeze(axis=-1)
            batch_tokens = np.concatenate((batch_tokens, next_tokens[:, None]), axis=-1)
            batch_logprobs = np.concatenate((batch_logprobs, next_logprobs[:, None]), axis=-1)

        # Drop the start token and the prompt: they are not part of the transcript.
        for tokens, logprobs in zip(batch_tokens[:, 1 + len(prompt) :], batch_logprobs[:, len(prompt) :], strict=True):
            mask = tokens != self._eos_token_id
            yield tokens[mask], None, logprobs[mask]
