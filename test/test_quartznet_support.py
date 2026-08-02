"""Tests for the QuartzNet/Jasper runtime backfills (``_quartznet``)."""
import numpy as np

from ovos_stt_plugin_onnxasr._quartznet import ensure_quartznet_support, melscale_fbanks64


class TestFbanks:
    def test_shape_and_dtype(self):
        fb = melscale_fbanks64()
        assert fb.shape == (257, 64)
        assert fb.dtype == np.float32

    def test_slaney_normalized_triangles(self):
        fb = melscale_fbanks64()
        # every filter is non-negative and has exactly one peak region
        assert (fb >= 0).all()
        assert (fb.sum(axis=0) > 0).all()
        # slaney area normalization keeps filter areas roughly equal
        areas = fb.sum(axis=0)
        assert areas.max() / areas.min() < 1.6


class TestPatches:
    def test_idempotent(self):
        from onnx_asr import loader
        from onnx_asr.models import nemo as nemo_models

        ensure_quartznet_support()
        create1 = loader.Manager._create_preprocessor
        encode1 = nemo_models.NemoConformerCtc._encode
        ensure_quartznet_support()
        assert loader.Manager._create_preprocessor is create1
        assert nemo_models.NemoConformerCtc._encode is encode1

    def test_nemo64_preprocessor_output(self):
        from onnx_asr.loader import Manager

        ensure_quartznet_support()
        manager = Manager()
        pre = manager._create_preprocessor("nemo64")
        waveforms = np.random.default_rng(0).standard_normal((1, 16000), dtype=np.float32)
        feats, lens = pre(waveforms, np.array([16000], dtype=np.int64))
        assert feats.shape[1] == 64
        # upstream NemoPreprocessorNumpy semantics: lens = samples // hop_length
        assert lens[0] == 16000 // 160

    def test_existing_preprocessors_untouched(self):
        from onnx_asr.loader import Manager

        ensure_quartznet_support()
        pre = Manager()._create_preprocessor("nemo80")
        waveforms = np.zeros((1, 16000), dtype=np.float32)
        feats, _ = pre(waveforms, np.array([16000], dtype=np.int64))
        assert feats.shape[1] == 80
