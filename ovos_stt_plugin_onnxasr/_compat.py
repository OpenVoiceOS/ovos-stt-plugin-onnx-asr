"""Compatibility entry points for extra onnx-asr model types.

The registration itself lives in :mod:`ovos_stt_plugin_onnxasr._shims`. This module
keeps the names the plugin has always exported.
"""

from ovos_stt_plugin_onnxasr._shims import ensure_model_types

__all__ = ["ensure_model_types", "ensure_wav2vec2_ctc"]


def ensure_wav2vec2_ctc() -> None:
    """Register the ``wav2vec2-ctc`` model type only.

    Does nothing when the installed onnx-asr provides the type.
    """
    ensure_model_types(only={"wav2vec2-ctc"})
