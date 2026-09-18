# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/models/speech_llm.py
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
"""Speech-LLM model implementation (audio encoder -> projector -> causal LM).

The family covers models that transcribe by prompting a causal language model
with audio embeddings, for example Qwen3-ASR, SLAM-ASR and Cohere Transcribe.
The ONNX export is always three graphs:

`encoder.onnx`
    Log-mel features -> audio embeddings in the language model embedding space.
`embed_tokens.onnx`
    Token ids -> token embeddings.
`decoder.onnx`
    Input embeddings + KV cache -> logits + new KV cache.

The runtime builds one embedding sequence from the prompt prefix embeddings,
the audio embeddings and the prompt suffix embeddings, then runs a greedy loop
over the decoder graph. The prompt token ids are baked into `config.json` at
export time, so no tokenizer encoder is needed at runtime.
"""

import json
import re
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as rt

from onnx_asr.asr import BaseAsr, Preprocessor, TimestampedResult
from onnx_asr.models.whisper import bytes_to_unicode
from onnx_asr.onnx import OnnxSessionOptions
from onnx_asr.utils import is_float32_array

NEG_INF = np.float32(-3.4028235e38)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|.*\|>\Z|<(?:s|/s|unk|pad)>\Z")
BYTE_FALLBACK_PATTERN = re.compile(r"<0x([0-9A-F]{2})>")


def _post_cnn_length(lengths: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Length after three (kernel 3, stride 2, padding 1) convolutions."""
    for _ in range(3):
        lengths = np.where(lengths > 0, (lengths - 1) // 2 + 1, 0)
    return lengths


class SpeechLlm(BaseAsr):
    """Speech-LLM model implementation."""

    def __init__(  # noqa: D107
        self,
        model_files: dict[str, Path],
        preprocessor_factory: Callable[[str], Preprocessor],
        onnx_options: OnnxSessionOptions,
    ):
        super().__init__(model_files, preprocessor_factory, onnx_options)
        config = typing.cast(dict[str, typing.Any], self.config)

        self._encoder = rt.InferenceSession(model_files["encoder"], **onnx_options)
        self._embed_tokens = rt.InferenceSession(model_files["embed_tokens"], **onnx_options)
        self._decoder = rt.InferenceSession(model_files["decoder"], **onnx_options)

        with model_files["vocab"].open("rt", encoding="utf-8") as f:
            tokens: dict[str, int] = json.load(f)
        self._vocab = {id: token for token, id in tokens.items()}
        self._byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self._tokenizer_type = config.get("tokenizer_type", "byte-level")

        self._prefix_ids = config["prompt_prefix_ids"]
        self._suffix_ids = config["prompt_suffix_ids"]
        self._language_prompt_ids: dict[str, list[int]] = config.get("language_prompt_ids", {})
        self._language_suffix_ids: dict[str, list[int]] = config.get("language_suffix_ids", {})
        self._eos_token_ids = set(config["eos_token_ids"])
        self._text_start_token_id = config.get("text_start_token_id")
        self._max_sequence_length = config.get("max_sequence_length", 512)

        self._normalize_waveform = config.get("normalize_waveform", False)
        self._trim_features = int(config.get("trim_features", 0))
        self._suppress_token_ids = np.array(config.get("suppress_token_ids", []), dtype=np.int64)
        self._n_window = config.get("n_window", 50)
        self._n_window_infer = config.get("n_window_infer", 800)
        self._max_frames = config.get("max_frames", 3000)
        self._hop_length = config.get("hop_length", 160)

        self._encoder_inputs = {x.name for x in self._encoder.get_inputs()}
        self._past_names = [x.name for x in self._decoder.get_inputs() if x.name.startswith("past_key_values.")]
        self._present_names = [name.replace("past_key_values.", "present.") for name in self._past_names]
        past_shape = next(x.shape for x in self._decoder.get_inputs() if x.name.startswith("past_key_values."))
        self._num_kv_heads = past_shape[1]
        self._head_dim = past_shape[3]

    @staticmethod
    def _get_model_files(quantization: str | None = None) -> dict[str, str]:
        suffix = "?" + quantization if quantization else ""
        return {
            "encoder": f"**/encoder{suffix}.onnx",
            "embed_tokens": f"**/embed_tokens{suffix}.onnx",
            "decoder": f"**/decoder{suffix}.onnx",
            "vocab": "vocab.json",
        }

    @property
    def _preprocessor_name(self) -> str:
        config = typing.cast(dict[str, typing.Any], self.config)
        return str(config.get("preprocessor", f"whisper{config.get('features_size', 128)}"))

    def _audio_windows(self, feature_len: int) -> tuple[int, npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Compute padded frame count, packing indices and the block-diagonal attention bias."""
        chunk_len = self._n_window * 2
        frames_per_chunk = _post_cnn_length(np.array([chunk_len]))[0]

        padded_frames = -(-feature_len // chunk_len) * chunk_len
        num_chunks = padded_frames // chunk_len
        starts = np.arange(num_chunks) * chunk_len
        chunk_lengths = np.clip(feature_len - starts, 0, chunk_len)
        chunk_out_lengths = _post_cnn_length(chunk_lengths)

        valid_indices = np.concatenate(
            [np.arange(length) + i * frames_per_chunk for i, length in enumerate(chunk_out_lengths)]
        ).astype(np.int64)

        window = int(chunk_out_lengths.max()) * (self._n_window_infer // chunk_len)
        total = int(chunk_out_lengths.sum())
        blocks = [window] * (total // window)
        if total % window:
            blocks.append(total % window)

        bias = np.full((1, 1, total, total), NEG_INF, dtype=np.float32)
        offset = 0
        for size in blocks:
            bias[0, 0, offset : offset + size, offset : offset + size] = 0.0
            offset += size

        return padded_frames, valid_indices, bias

    def _encode(self, features: npt.NDArray[np.float32], feature_len: int) -> npt.NDArray[np.float32]:
        # Encoders with full attention over a fixed feature length (Whisper style) need
        # neither the packing indices nor the block-diagonal mask, so they declare only
        # input_features and get the features unchanged.
        inputs: dict[str, npt.NDArray[typing.Any]]
        if self._encoder_inputs == {"input_features"}:
            # The Whisper preprocessor always pads to 30 s. Models trained on the
            # unpadded features (Audio8 ARK-ASR) need the padding cut away again,
            # because every feature frame becomes an audio embedding. `trim_features`
            # is the frame multiple the encoder needs (subsampling x embedding merge).
            if self._trim_features:
                length = max(feature_len - feature_len % self._trim_features, self._trim_features)
                inputs = {"input_features": features[None, :, :length]}
            else:
                inputs = {"input_features": features[None]}
        elif "input_features_lens" in self._encoder_inputs:
            # NeMo FastConformer encoders take the feature length and mask the padding
            # themselves, so they also return the valid embedding length.
            audio_embeds, lens = self._encoder.run(
                ["audio_embeds", "audio_embeds_lens"],
                {
                    "input_features": features[None, :, :feature_len],
                    "input_features_lens": np.array([feature_len], dtype=np.int64),
                },
            )
            assert is_float32_array(audio_embeds)
            return audio_embeds[:, : int(lens[0])]
        else:
            padded_frames, valid_indices, bias = self._audio_windows(feature_len)
            inputs = {
                "input_features": features[None, :, :padded_frames],
                "valid_indices": valid_indices,
                "attn_bias": bias,
            }

        (audio_embeds,) = self._encoder.run(["audio_embeds"], inputs)
        assert is_float32_array(audio_embeds)
        return audio_embeds

    def _embed(self, token_ids: list[int]) -> npt.NDArray[np.float32]:
        (embeds,) = self._embed_tokens.run(["inputs_embeds"], {"input_ids": np.array([token_ids], dtype=np.int64)})
        assert is_float32_array(embeds)
        return embeds

    def _empty_state(self) -> dict[str, npt.NDArray[np.float32]]:
        empty = np.zeros((1, self._num_kv_heads, 0, self._head_dim), dtype=np.float32)
        return dict.fromkeys(self._past_names, empty)

    def _decode_step(
        self,
        inputs_embeds: npt.NDArray[np.float32],
        state: dict[str, npt.NDArray[np.float32]],
        past_length: int,
    ) -> tuple[npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
        seq_len = inputs_embeds.shape[1]
        total = past_length + seq_len
        bias = np.zeros((1, 1, seq_len, total), dtype=np.float32)
        causal = np.arange(total)[None, :] > (past_length + np.arange(seq_len))[:, None]
        bias[0, 0][causal] = NEG_INF
        position_ids = np.arange(past_length, total, dtype=np.int64)[None]

        outputs = self._decoder.run(
            ["logits", *self._present_names],
            {"inputs_embeds": inputs_embeds, "attn_bias": bias, "position_ids": position_ids, **state},
        )
        logits = outputs[0]
        assert is_float32_array(logits)
        return logits, dict(zip(self._past_names, outputs[1:], strict=True))

    def _generate(self, audio_embeds: npt.NDArray[np.float32], language: str | None) -> list[int]:
        def select(prompts: dict[str, list[int]], default: list[int]) -> list[int]:
            if not language:
                return default
            return prompts.get(language, prompts.get(language.title(), default))

        # The language marker sits before the audio for Qwen3-ASR and after it for
        # Voxtral, so both ends of the prompt can carry a per-language variant.
        prefix_ids = select(self._language_prompt_ids, self._prefix_ids)
        suffix_ids = select(self._language_suffix_ids, self._suffix_ids)

        inputs_embeds = np.concatenate([self._embed(prefix_ids), audio_embeds, self._embed(suffix_ids)], axis=1)
        state = self._empty_state()
        past_length = 0
        tokens: list[int] = []

        for _ in range(self._max_sequence_length):
            logits, state = self._decode_step(inputs_embeds, state, past_length)
            past_length += inputs_embeds.shape[1]
            # Some models (Audio8 ARK-ASR) leave the special tokens in the softmax and
            # rely on the decoder to ban them, so a greedy loop derails without this.
            scores = logits[0, -1]
            if self._suppress_token_ids.size:
                scores = scores.copy()
                scores[self._suppress_token_ids] = NEG_INF
            token = int(scores.argmax())
            if token in self._eos_token_ids:
                break
            tokens.append(token)
            inputs_embeds = self._embed([token])

        return tokens

    def _decode_tokens(self, tokens: list[int]) -> TimestampedResult:
        # Some models answer with a preamble (Qwen3-ASR writes the detected language first)
        # and start the transcription after a marker token.
        if self._text_start_token_id in tokens:
            tokens = tokens[tokens.index(self._text_start_token_id) + 1 :]

        parts = [token for id in tokens if (token := self._vocab[id]) and not SPECIAL_TOKEN_PATTERN.match(token)]

        if self._tokenizer_type == "sentencepiece":
            # Replace the SentencePiece space marker, then resolve byte-fallback tokens.
            data = bytearray()
            for token in parts:
                if (byte := BYTE_FALLBACK_PATTERN.fullmatch(token)) is not None:
                    data.append(int(byte[1], 16))
                else:
                    data += token.replace("▁", " ").encode()
        else:
            data = bytearray(self._byte_decoder[char] for char in "".join(parts))

        return TimestampedResult(data.decode("utf-8", errors="replace").strip())

    def recognize_batch(
        self, waveforms: npt.NDArray[np.float32], waveforms_len: npt.NDArray[np.int64], /, **kwargs: object | None
    ) -> Iterator[TimestampedResult]:
        """Recognize waveforms batch (processed one waveform at a time)."""
        language = typing.cast(str | None, kwargs.get("language"))

        if self._normalize_waveform:
            # SLAM-ASR models normalize the waveform to zero mean and unit variance
            # before the feature extractor.
            mean = waveforms.mean(axis=-1, keepdims=True)
            var = waveforms.var(axis=-1, keepdims=True)
            waveforms = (waveforms - mean) / np.sqrt(var + 1e-5)

        features, _ = self._preprocessor(waveforms, waveforms_len)

        for i, waveform_len in enumerate(waveforms_len):
            feature_len = min(int(-(-int(waveform_len) // self._hop_length)), self._max_frames, features.shape[-1])
            audio_embeds = self._encode(features[i], max(feature_len, 1))
            yield self._decode_tokens(self._generate(audio_embeds, language))
