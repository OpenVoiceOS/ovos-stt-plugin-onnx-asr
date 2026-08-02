"""Runtime backfills for QuartzNet/Jasper support in released onnx-asr.

Two gaps in onnx-asr <= 0.12.0 (fixed in the TigreGotico fork, pending upstream):

1. No ``nemo64`` preprocessor — QuartzNet/Jasper use 64 mel bins, not 80/128.
2. ``NemoConformerCtc`` always passes a ``length`` input, but QuartzNet exports
   (masked convolutions disabled) have no such graph input.

``ensure_quartznet_support()`` patches both at runtime and is a no-op on
onnx-asr versions that already handle them.
"""

import numpy as np

_MEL_BREAK_HZ = 1000.0


def _hz_to_mel(freq):
    return np.where(
        freq < _MEL_BREAK_HZ,
        3 * freq / 200.0,
        15 + 27 * np.log(freq / _MEL_BREAK_HZ + np.finfo(np.float32).eps) / np.log(6.4),
    )


def _mel_to_hz(mels):
    return np.where(mels < 15, 200 * mels / 3.0, _MEL_BREAK_HZ * np.power(6.4, (mels - 15) / 27.0))


def melscale_fbanks64(n_freqs: int = 257, sample_rate: int = 16_000) -> np.ndarray:
    """Slaney-scale, slaney-normalized 64-bin mel filterbank (torchaudio-compatible)."""
    n_mels = 64
    freqs = np.linspace(0, sample_rate // 2, n_freqs)
    mel_pts = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(sample_rate / 2)), n_mels + 2)
    f_pts = _mel_to_hz(mel_pts)
    f_diff = np.diff(f_pts)
    slopes = f_pts[np.newaxis, :] - freqs[:, np.newaxis]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    fb *= 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    return fb.astype(np.float32)


def ensure_quartznet_support() -> None:
    """Teach installed onnx-asr about nemo64 and length-free NeMo CTC graphs."""
    from onnx_asr import loader
    from onnx_asr.models import nemo as nemo_models
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy

    # 1. nemo64 preprocessor (NumPy implementation, fbanks computed here)
    if not getattr(loader.Manager._create_preprocessor, "_nemo64_patched", False):

        class _Nemo64Numpy(NemoPreprocessorNumpy):
            def __init__(self):
                # skip _NumpyPreprocessor.__init__ — fbanks.npz has no nemo64 key
                self._melscale_fbanks = melscale_fbanks64()

        original_create = loader.Manager._create_preprocessor

        def _create_preprocessor(self, name):
            try:
                return original_create(self, name)
            except Exception:
                if name == "nemo64":
                    return _Nemo64Numpy()
                raise

        _create_preprocessor._nemo64_patched = True
        loader.Manager._create_preprocessor = _create_preprocessor

    # 2. length-optional encode for QuartzNet/Jasper graphs
    if not getattr(nemo_models.NemoConformerCtc._encode, "_length_optional", False):
        original_encode = nemo_models.NemoConformerCtc._encode

        def _encode(self, features, features_lens):
            has_length = getattr(self, "_model_has_length", None)
            if has_length is None:
                has_length = any(i.name == "length" for i in self._model.get_inputs())
                self._model_has_length = has_length
            if has_length:
                return original_encode(self, features, features_lens)
            (logprobs,) = self._model.run(["logprobs"], {"audio_signal": features})
            return logprobs, (features_lens - 1) // self._subsampling_factor + 1

        _encode._length_optional = True
        nemo_models.NemoConformerCtc._encode = _encode
