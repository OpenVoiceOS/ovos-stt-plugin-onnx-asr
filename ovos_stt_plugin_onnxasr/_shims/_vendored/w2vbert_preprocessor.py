# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      src/onnx_asr/preprocessors/numpy_preprocessor.py (class W2vBertPreprocessorNumpy)
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
"""w2v-BERT 2.0 preprocessor, extracted from onnx_asr.preprocessors.numpy_preprocessor."""

import numpy as np
import numpy.typing as npt

from .._core import _NumpyPreprocessor


class W2vBertPreprocessorNumpy(_NumpyPreprocessor):
    """w2v-BERT 2.0 preprocessor implementation with NumPy.

    Reproduces `transformers.SeamlessM4TFeatureExtractor` (the feature extractor of
    `facebook/w2v-bert-2.0`): Kaldi log mel filterbanks with a Povey window, per mel bin
    mean-variance normalization over the valid frames of each utterance, and 2 frame
    stacking into 160 dimensional features.
    """

    _n_fft = 512
    _win_length = 400
    _hop_length = 160
    _preemphasis_coefficient = 0.97
    _waveform_scale = float(2**15)
    _mel_floor = 1.192092955078125e-07
    _norm_eps = 1e-7
    _stride = 2

    def __init__(self, name: str):  # noqa: D107
        assert name == "w2vbert"
        super().__init__(name)
        self._window = (np.hanning(self._win_length) ** 0.85).astype(np.float32)

    def __call__(
        self, waveforms: npt.NDArray[np.float32], waveforms_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Convert waveforms to model features."""
        features_lens = 1 + (waveforms_lens - self._win_length) // self._hop_length

        # The reference feature extractor works in float64 on 16 bit scaled samples.
        strided_input = (
            np.lib.stride_tricks.sliding_window_view(waveforms, self._win_length, axis=1)[
                :, :: self._hop_length
            ].astype(np.float64)
            * self._waveform_scale
        )
        strided_input = strided_input - np.mean(strided_input, axis=-1, keepdims=True)
        offset_strided_input = np.pad(strided_input, ((0, 0), (0, 0), (1, 0)), mode="edge")
        strided_input = strided_input - self._preemphasis_coefficient * offset_strided_input[..., :-1]
        strided_input = strided_input * self._window

        spectrum = np.abs(np.fft.rfft(strided_input, self._n_fft)) ** 2
        mel_energies = np.matmul(spectrum, self._melscale_fbanks.astype(np.float64))
        features = np.log(np.maximum(mel_energies, self._mel_floor))

        mask = np.arange(features.shape[1])[None, :, None] < features_lens[:, None, None]
        counts = features_lens[:, None, None]
        mean = np.divide(np.where(mask, features, 0.0).sum(axis=1, keepdims=True), counts)
        var = np.divide(np.where(mask, (features - mean) ** 2, 0.0).sum(axis=1, keepdims=True), counts - 1)
        features = np.where(mask, (features - mean) / np.sqrt(var + self._norm_eps), 0.0).astype(np.float32)

        if features.shape[1] % self._stride != 0:
            features = np.pad(features, ((0, 0), (0, self._stride - features.shape[1] % self._stride), (0, 0)))

        features = features.reshape(features.shape[0], features.shape[1] // self._stride, -1)
        features_lens = features_lens // self._stride
        # Stacking an odd frame count leaves one trailing half-padded frame, which the
        # reference feature extractor also masks out. Drop it so that the frame count is
        # exactly max(features_lens): ESPnet builds its attention masks with that length,
        # and a longer tensor would put the subsampled mask out of step with the encoder.
        return features[:, : features_lens.max() if features_lens.size else 0], features_lens

