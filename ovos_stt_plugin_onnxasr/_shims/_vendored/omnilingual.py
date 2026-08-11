# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/omnilingual.py
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
"""Omnilingual ASR CTC model (Meta, 1600+ languages).

The graph is a wav2vec2 encoder with a CTC head. It takes the raw 16 kHz waveform,
so there is no separate feature frontend: the convolutional feature extractor is
part of the graph. One shared vocabulary covers every language, and nothing selects
a language, so the model transcribes whatever it hears.

Reference: https://github.com/facebookresearch/omnilingual-asr (Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, TimestampedResult, _AsrWithCtcDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int64_array

BLANK_IDX = 0
"""Index of the CTC blank. The exported vocabulary puts `<s>` there."""

SUBSAMPLING_FACTOR = 320
"""The wav2vec2 feature extractor emits one frame per 320 input samples (20 ms)."""


class OmnilingualCtc(_AsrWithCtcDecoding):
    """Omnilingual ASR CTC model with a single vocabulary for 1600+ languages."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        self._vocab = self._read_tokens(model_files["tokens"])
        self._vocab_size = len(self._vocab)
        self._blank_idx = BLANK_IDX
        self._model = rt.InferenceSession(model_files["model"], **onnx_options)

    @staticmethod
    def _read_tokens(path: Path) -> dict[int, str]:
        """Read a `token id` file where the token may itself be a space."""
        vocab = {}
        with Path(path).open("rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                token, index = line.rsplit(" ", 1)
                vocab[int(index)] = token
        return vocab

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "tokens": "tokens.txt"}

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    @property
    def _subsampling_factor(self) -> int:
        return int(self.config.get("subsampling_factor", SUBSAMPLING_FACTOR))

    def _decode_tokens(
        self,
        ids: Iterable[int],
        indices: Iterable[int] | None,
        logprobs: Iterable[float] | None,
    ) -> TimestampedResult:
        """Join the tokens as they are.

        The vocabulary already holds real spaces, so the word separator needs no
        substitution. Only the leading and trailing spaces of a segment go away.
        """
        tokens = [self._vocab[i] for i in ids]
        timestamps = (
            None if indices is None else (self.window_step * self._subsampling_factor * np.asarray(indices)).tolist()
        )
        return TimestampedResult(
            "".join(tokens).strip(),
            timestamps,
            tokens,
            None if logprobs is None else np.asarray(logprobs).tolist(),
        )

    def _encode(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        (logits,) = self._model.run(["logits"], {"x": waveforms})
        assert is_float32_array(logits)

        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        assert is_float32_array(logprobs)

        out_lens = waveforms_len // self._subsampling_factor
        out_lens = np.minimum(out_lens, logprobs.shape[1]).astype(np.int64)
        assert is_int64_array(out_lens)
        return logprobs, out_lens
