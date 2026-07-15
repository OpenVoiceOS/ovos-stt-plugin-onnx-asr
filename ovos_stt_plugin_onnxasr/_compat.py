"""Runtime registration of the ``wav2vec2-ctc`` model type into onnx-asr.

onnx-asr only learns about ``wav2vec2-ctc`` once TigreGotico/onnx-asr PR #1 is merged
and released. Until then, :func:`ensure_wav2vec2_ctc` teaches an *installed* onnx-asr
about the vendored :class:`~ovos_stt_plugin_onnxasr._wav2vec2.Wav2Vec2Ctc` class so the
OpenVoiceOS wav2vec2 ONNX models load through the normal
``onnx_asr.load_model(<hf-repo-id>)`` path.

The patch targets ``onnx_asr.loader.create_asr_resolver`` — the single chokepoint that
builds the name→class ``model_types`` mapping and hands it to ``Resolver``. It is a
no-op when onnx-asr already ships native support, and can be deleted together with
``_wav2vec2`` once that release is the norm.
"""

from ovos_utils.log import LOG

_SENTINEL = "_ovos_wav2vec2_patched"


def ensure_wav2vec2_ctc() -> None:
    """Idempotently register ``wav2vec2-ctc`` with the installed onnx-asr.

    Does nothing if onnx-asr already provides the type natively (PR #1 merged and
    installed), if the patch is already applied, or if onnx-asr's internal resolver
    API cannot be located (in which case built-in model types are left untouched).
    """
    # Native support present (upstream merged + installed) -> nothing to do.
    try:
        import onnx_asr.models.wav2vec2  # noqa: F401

        return
    except ImportError:
        pass
    except Exception:  # onnx_asr mocked out (e.g. in unit tests) — leave it alone.
        return

    try:
        import onnx_asr.loader as loader
    except Exception:
        return

    original = getattr(loader, "create_asr_resolver", None)
    if original is None or getattr(original, _SENTINEL, False):
        return

    from ._wav2vec2 import Wav2Vec2Ctc

    # Base model_types dict is a local var rebuilt on every call and never exposed.
    # Capture it — without re-listing every built-in type, so we stay correct as
    # upstream adds new ones — by letting the original build it and recording the
    # dict it passes to Resolver. Cached after the first successful capture.
    cache: dict = {}

    def _capture_base_model_types() -> dict:
        if cache:
            return dict(cache)
        captured: dict = {}
        real_resolver = loader.Resolver

        class _Spy:
            def __init__(self, model_types, *args, **kwargs):
                captured.update(model_types)

        loader.Resolver = _Spy
        try:
            original("wav2vec2-ctc")  # spy short-circuits; no network, no validation
        except Exception:
            pass
        finally:
            loader.Resolver = real_resolver
        if captured:
            cache.update(captured)
        return dict(captured)

    def create_asr_resolver(model=None, local_dir=None, *, offline=None):
        model_types = _capture_base_model_types()
        if not model_types:
            # Resolver API drifted — don't risk breaking built-in models.
            LOG.warning(
                "onnx-asr resolver internals changed; wav2vec2-ctc registration skipped"
            )
            return original(model, local_dir, offline=offline)
        model_types.setdefault("wav2vec2-ctc", Wav2Vec2Ctc)
        return loader.Resolver(model_types, model, local_dir, offline=offline)

    setattr(create_asr_resolver, _SENTINEL, True)
    loader.create_asr_resolver = create_asr_resolver
    LOG.debug("registered wav2vec2-ctc model type with onnx-asr")
