# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/moonshine.py
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
"""Moonshine model implementation."""

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from onnxruntime import OrtValue

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from onnx_asr.onnx import OnnxSessionOptions, TensorRtOptions, get_onnx_device
from onnx_asr.utils import is_float32_array


class Moonshine(BaseAsr):
    """Moonshine (Useful Sensors) model implementation.

    Moonshine is an encoder-decoder ASR model that reads the raw 16 kHz waveform,
    so it needs no mel preprocessor. The ONNX graphs are the transformers.js/optimum
    exports (`encoder_model.onnx` + `decoder_model_merged.onnx`).

    The published models are English only.
    """

    BYTE_TOKEN_PATTERN = re.compile(r"\A<0x([0-9A-Fa-f]{2})>\Z")
    MAX_TOKENS_PER_SECOND = 6
    """Token budget per second of audio, as used by the reference implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)

        with model_files["tokenizer"].open("rt", encoding="utf-8") as f:
            tokenizer = json.load(f)

        vocab: dict[str, int] = dict(tokenizer["model"]["vocab"])
        vocab |= {token["content"]: token["id"] for token in tokenizer.get("added_tokens", ())}
        self._vocab = {id: token for token, id in vocab.items()}
        self._special_ids = {token["id"] for token in tokenizer.get("added_tokens", ())}

        self._bos_token_id: int = self.config.get("decoder_start_token_id", vocab["<s>"])
        self._eos_token_id: int = self.config.get("eos_token_id", vocab["</s>"])
        self._max_sequence_length: int = self.config.get("max_position_embeddings", 512)

        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)
        self._device_type, self._device_id = get_onnx_device(self._encoder)

    @staticmethod
    def _get_excluded_providers() -> list[str]:
        return TensorRtOptions.get_provider_names()

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder_model{suffix}.onnx",
            "decoder": f"**/decoder_model_merged{suffix}.onnx",
            "tokenizer": "tokenizer.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        return "identity"

    def _encode(self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64]) -> OrtValue:
        input_values, _ = self._preprocessor(waveforms, waveforms_len)
        binding = self._encoder.io_binding()
        binding.bind_cpu_input("input_values", input_values)
        binding.bind_output("last_hidden_state", self._device_type, self._device_id)
        self._encoder.run_with_iobinding(binding)
        last_hidden_state: OrtValue = binding.get_outputs()[0]
        return last_hidden_state

    def _create_state(self) -> dict[str, OrtValue]:
        return {
            x.name: OrtValue.ortvalue_from_numpy(np.zeros((0, x.shape[1], 0, x.shape[3]), dtype=np.float32))
            for x in self._decoder.get_inputs()
            if x.name.startswith("past_key_values.")
        }

    def _decode(
        self,
        tokens: npt.NDArray[np.int64],
        prev_state: dict[str, OrtValue],
        encoder_out: OrtValue,
    ) -> tuple[npt.NDArray[np.float32], dict[str, OrtValue]]:
        use_cache = any(x.shape()[0] for x in prev_state.values())

        binding = self._decoder.io_binding()
        binding.bind_cpu_input("input_ids", tokens[:, -1:] if use_cache else tokens)
        binding.bind_ortvalue_input("encoder_hidden_states", encoder_out)
        binding.bind_output("logits")
        binding.bind_cpu_input("use_cache_branch", np.array([use_cache]))
        for key, value in prev_state.items():
            binding.bind_ortvalue_input(key, value)
            binding.bind_output(key.replace("past_key_values.", "present."), self._device_type, self._device_id)

        self._decoder.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        logits = outputs[0].numpy()
        assert is_float32_array(logits)
        return logits, {
            key: next_value if next_value.shape()[0] else prev_value
            for (key, prev_value), next_value in zip(prev_state.items(), outputs[1:], strict=True)
        }

    def _decoding(self, encoder_out: OrtValue, tokens: npt.NDArray[np.int64], max_length: int) -> npt.NDArray[np.int64]:
        state = self._create_state()
        for _ in range(tokens.shape[-1], max_length):
            logits, state = self._decode(tokens, state, encoder_out)
            next_tokens = logits[:, -1].argmax(axis=-1)
            next_tokens[tokens[:, -1] == self._eos_token_id] = self._eos_token_id
            tokens = np.hstack((tokens, next_tokens[:, None]))
            if (tokens[:, -1] == self._eos_token_id).all():
                break

        return tokens

    def _decode_tokens(self, tokens: npt.NDArray[np.int64]) -> TimestampedResult:
        buffer = bytearray()
        for id in tokens:
            if id in self._special_ids:
                continue
            token = self._vocab[id]
            if match := self.BYTE_TOKEN_PATTERN.match(token):
                buffer.append(int(match.group(1), 16))
            else:
                buffer += token.replace("▁", " ").encode("utf-8")

        return TimestampedResult(buffer.decode("utf-8", errors="replace").removeprefix(" "))

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch."""
        encoder_out = self._encode(waveforms, waveforms_len)
        tokens = np.full((len(waveforms), 1), self._bos_token_id, dtype=np.int64)

        seconds = int(waveforms_len.max()) / self._get_sample_rate()
        max_length = min(1 + int(seconds * self.MAX_TOKENS_PER_SECOND), self._max_sequence_length)

        return map(self._decode_tokens, self._decoding(encoder_out, tokens, max_length))
