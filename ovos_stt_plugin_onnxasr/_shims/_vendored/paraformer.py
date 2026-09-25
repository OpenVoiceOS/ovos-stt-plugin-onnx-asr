# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/paraformer.py
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
"""Paraformer model implementation (FunASR offline non-autoregressive ASR).

The family covers the `csukuangfj/sherpa-onnx-paraformer-*` exports of the Alibaba
FunASR Paraformer models: a SAN-M encoder, a CIF predictor that decides how many
tokens the utterance has, and a single pass non-autoregressive decoder. There is no
decoding loop and no blank symbol, so this is not a CTC model: the graph emits one
logit row per output token and decoding is one argmax per row.

`model.onnx`
    FunASR 560 dim features + lengths -> logits and the CIF token count.

The graph expects the FunASR frontend output, which is a kaldi fbank of an int16
scaled waveform, then a low frame rate stack (m=7, n=6), then the `am.mvn` mean
variance statistics. Only the fbank is a preprocessor step, and the `wespeaker`
preprocessor already computes it. The LFR stack and the CMVN stay in this class,
with the statistics in `config.json`, because the graphs are copied byte for byte
from the sherpa-onnx repositories and are not re-exported.

The last decoded token is `</s>`, so decoding stops at the end of sequence id and
also at the token count that the CIF predictor returned, whichever comes first.
"""

import typing
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import Preprocessor, TimestampedResult, _AsrWithDecoding
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array, is_int32_array, log_softmax

EOS_TOKEN = "</s>"
SUBWORD_SUFFIX = "@@"


class Paraformer(_AsrWithDecoding):
    """Paraformer model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)

        self._model = rt.InferenceSession(model_files["model"], **onnx_options)
        # The FunASR frontend runs the fbank on an int16 scaled waveform. The scale
        # matters because the log floor and the CMVN both depend on it.
        self._waveform_scale = float(config.get("waveform_scale", 1 << 15))
        self._lfr_m = int(config.get("lfr_window_size", 7))
        self._lfr_n = int(config.get("lfr_window_shift", 6))
        self._cmvn_add = np.array(config["neg_mean"], dtype=np.float32)
        self._cmvn_mul = np.array(config["inv_stddev"], dtype=np.float32)
        self._eos_idx = next(id for id, token in self._vocab.items() if token == EOS_TOKEN)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {"model": f"model{suffix}.onnx", "vocab": "vocab.txt", "config": "config.json"}

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", "wespeaker"))

    @property
    def _subsampling_factor(self) -> int:
        return self._lfr_n

    def _features(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        features, features_lens = self._preprocessor(
            (waveforms * self._waveform_scale).astype(np.float32), waveforms_len
        )
        return self._lfr_cmvn(features, features_lens)

    def _lfr_cmvn(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Stack `lfr_m` frames with a hop of `lfr_n`, then apply the `am.mvn` statistics."""
        batch, time, dim = features.shape
        m, n = self._lfr_m, self._lfr_n

        # Replace batch padding with the last valid frame of each item, which is what
        # the FunASR right padding does, so a batched run and a single clip run agree.
        last = features[np.arange(batch), np.maximum(features_lens - 1, 0)][:, None]
        valid = np.arange(time)[None, :] < features_lens[:, None]
        features = np.where(valid[:, :, None], features, last)

        time_lfr = -(-time // n)
        left = np.repeat(features[:, :1], (m - 1) // 2, axis=1)
        right = np.repeat(last, max(n * time_lfr + 1 - (m - 1) // 2 - time, 0), axis=1)
        padded = np.concatenate([left, features, right], axis=1)
        stacked = np.concatenate([padded[:, k : k + n * time_lfr : n] for k in range(m)], axis=2)

        speech = (stacked.reshape(batch, time_lfr, m * dim) + self._cmvn_add) * self._cmvn_mul
        return speech.astype(np.float32), -(-features_lens // n)

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        logits, token_num = self._model.run(
            ["logits", "token_num"], {"speech": features, "speech_lengths": features_lens.astype(np.int32)}
        )
        assert is_float32_array(logits)
        assert is_int32_array(token_num)
        return logits, np.minimum(token_num.astype(np.int64), logits.shape[1])

    def _decoding(
        self, encoder_out: npt.NDArray[np.float32], encoder_out_lens: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], None, Iterable[float] | None]]:
        need_logprobs = kwargs.get("need_logprobs")
        for logits, token_num in zip(encoder_out, encoder_out_lens, strict=True):
            tokens = logits[:token_num].argmax(axis=-1)
            eos = np.flatnonzero(tokens == self._eos_idx)
            if eos.size:
                tokens = tokens[: eos[0]]
            logprobs = (
                [float(log_softmax(row)[token]) for row, token in zip(logits, tokens, strict=False)]
                if need_logprobs
                else None
            )
            # Paraformer has no per token frame index, so there are no timestamps.
            yield tokens.tolist(), None, logprobs

    def _decode_tokens(
        self, ids: Iterable[int], indices: Iterable[int] | None, logprobs: Iterable[float] | None
    ) -> TimestampedResult:
        result = super()._decode_tokens(ids, indices, logprobs)
        assert result.tokens is not None
        result.text = self._join(result.tokens)
        return result

    @staticmethod
    def _join(tokens: list[str]) -> str:
        """Join FunASR tokens: `@@` marks a subword, and CJK tokens carry no spaces."""
        parts: list[str] = []
        mergeable = False
        for i, token in enumerate(tokens):
            if token.endswith(SUBWORD_SUFFIX):
                word = token.removesuffix(SUBWORD_SUFFIX)
                parts.append(word if mergeable else " " + word)
                mergeable = True
            elif token.isascii():
                parts.append(token if mergeable else " " + token)
                mergeable = False
            else:
                if i > 0 and tokens[i - 1].isascii():
                    parts.append(" ")
                parts.append(token)
                mergeable = False
        return "".join(parts).strip()

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch.

        Args:
            waveforms: Waveforms batch.
            waveforms_len: Waveform lengths.
            **kwargs: Passed on to the decoding.

        """
        logits, token_num = self._encode(*self._features(waveforms, waveforms_len))
        return map(self._decode_tokens, *zip(*self._decoding(logits, token_num, **kwargs), strict=False))
