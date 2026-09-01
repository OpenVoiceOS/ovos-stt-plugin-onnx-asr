# Vendored verbatim from onnx-asr.
#
# Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
# Source fork:      https://github.com/TigreGotico/onnx-asr
# Source commit:    8fd5f2b30fdd10d88066dc53a4c5558a208a0512
# Source path:      preprocessors/fbanks.py
#                   (from the root of the fork; this is the build tooling that
#                   makes the shipped filterbanks, not part of src/onnx_asr)
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
"""Filterbank generation for Mel spectrograms."""

from typing import Literal

import numpy as np
import numpy.typing as npt


def _hz_to_mel(
    freq: float | npt.NDArray[np.float64], mel_scale: Literal["htk", "kaldi", "slaney"]
) -> npt.NDArray[np.float64]:
    if mel_scale == "htk":
        return 2595 * np.log10(1.0 + freq / 700.0)
    if mel_scale == "kaldi":
        return 1127 * np.log(1.0 + freq / 700.0)
    return np.where(
        freq < 1000, 3 * freq / 200.0, 15 + 27 * np.log(freq / 1000.0 + np.finfo(np.float32).eps) / np.log(6.4)
    )


def _mel_to_hz(mels: npt.NDArray[np.float64], mel_scale: Literal["htk", "slaney"]) -> npt.NDArray[np.float64]:
    if mel_scale == "htk":
        return 700 * (np.pow(10.0, mels / 2595.0) - 1.0)
    return np.where(mels < 15, 200 * mels / 3.0, 1000 * np.pow(6.4, ((mels - 15) / 27.0)))


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Literal["slaney"] | None = None,
    mel_scale: Literal["htk", "slaney", "kaldi"] = "htk",
) -> npt.NDArray[np.float64]:
    if f_max <= 0.0:
        f_max += sample_rate / 2

    all_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    if mel_scale == "kaldi":
        mel = _hz_to_mel(all_freqs, mel_scale=mel_scale)
    else:
        mel = all_freqs
        m_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    up_slopes = (mel[:, None] - m_pts[:-2]) / (m_pts[1:-1] - m_pts[:-2])
    down_slopes = (m_pts[2:] - mel[:, None]) / (m_pts[2:] - m_pts[1:-1])
    fb = np.maximum(0.0, np.minimum(up_slopes, down_slopes))

    if norm == "slaney":
        fb *= 2.0 / (m_pts[2:] - m_pts[:-2])

    return fb
