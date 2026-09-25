# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/granite_nar.py
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
"""Granite Speech NAR model implementation (non-autoregressive transcript editing).

The family covers `ibm-granite/granite-speech-4.1-*-nar`: a conformer encoder with
a BPE CTC head writes a first-pass hypothesis, and a bidirectional language model
rewrites that hypothesis in a single forward pass. There is no decoding loop and
no KV cache, so the ONNX export is two graphs:

`encoder.onnx`
    Raw waveform -> audio embeddings in the editor embedding space, plus the BPE
    CTC logits and the length of both.
`editor.onnx`
    Audio embeddings + hypothesis token ids -> logits over the hypothesis slots.

The runtime collapses the CTC hypothesis, puts a blank editing slot before, after
and between every surviving token, and runs the editor once. A second CTC collapse
over the editor output gives the transcript, so the editor can delete a token (by
predicting blank), keep it, or fill a slot with a new one.

The token embedding table is tied to the editor output head, so it stays inside
`editor.onnx` and the runtime passes token ids rather than embeddings.
"""

import json
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from .speech_llm import SPECIAL_TOKEN_PATTERN
from onnx_asr.models.whisper import bytes_to_unicode
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array


def _ctc_collapse(token_ids: npt.NDArray[np.int64], blank_id: int) -> npt.NDArray[np.int64]:
    """Merge repeated ids, then drop the blanks."""
    if token_ids.size == 0:
        return token_ids
    kept = np.concatenate([[True], token_ids[1:] != token_ids[:-1]])
    collapsed = token_ids[kept]
    return collapsed[collapsed != blank_id]


def _insertion_slots(token_ids: npt.NDArray[np.int64], blank_id: int, min_length: int) -> npt.NDArray[np.int64]:
    """Surround every hypothesis token with a blank slot the editor may fill."""
    length = max(2 * token_ids.size + 1, min_length)
    slots = np.full(length, blank_id, dtype=np.int64)
    slots[1 : 2 * token_ids.size : 2] = token_ids
    return slots


class GraniteNar(BaseAsr):
    """Granite Speech NAR model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)

        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._editor = rt.InferenceSession(model_files["editor"], **onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            tokens: dict[str, int] = json.load(f)
        self._vocab = {id: token for token, id in tokens.items()}
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        self._blank_token_id = config["blank_token_id"]
        self._min_edit_sequence_length = config.get("min_edit_sequence_length", 8)

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder{suffix}.onnx",
            "editor": f"**/editor{suffix}.onnx",
            "vocab": "vocab.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", "identity"))

    def _encode(self, waveform: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        audio_embeds, ctc_logits = self._encoder.run(["audio_embeds", "ctc_logits"], {"input_features": waveform})
        assert is_float32_array(audio_embeds)
        assert is_float32_array(ctc_logits)
        return audio_embeds, ctc_logits

    def _edit(self, audio_embeds: npt.NDArray[np.float32], text_ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        (logits,) = self._editor.run(["logits"], {"audio_embeds": audio_embeds, "text_ids": text_ids[None]})
        assert is_float32_array(logits)
        return logits

    def _decode_tokens(self, token_ids: npt.NDArray[np.int64]) -> TimestampedResult:
        parts = [
            token for id in token_ids.tolist() if (token := self._vocab[id]) and not SPECIAL_TOKEN_PATTERN.match(token)
        ]
        data = bytearray(self._byte_decoder[char] for char in "".join(parts))
        return TimestampedResult(data.decode("utf-8", errors="replace").strip())

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch (processed one waveform at a time)."""
        features, _ = self._preprocessor(waveforms, waveforms_len)

        for i, waveform_len in enumerate(waveforms_len):
            audio_embeds, ctc_logits = self._encode(features[i, : int(waveform_len)][None])
            hypothesis = _ctc_collapse(ctc_logits[0].argmax(-1), self._blank_token_id)
            text_ids = _insertion_slots(hypothesis, self._blank_token_id, self._min_edit_sequence_length)
            logits = self._edit(audio_embeds, text_ids)
            yield self._decode_tokens(_ctc_collapse(logits[0].argmax(-1), self._blank_token_id))
