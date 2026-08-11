"""Registration of extra onnx-asr model types at runtime.

Some ONNX model families in the plugin registry need a model type that the
installed onnx-asr does not provide. This module carries the missing classes and
registers them with onnx-asr when the plugin starts, so that
``onnx_asr.load_model(<hf-repo-id>)`` loads those models through the normal path.

Presently one type needs this: ``wav2vec2-ctc`` (wav2vec2 / XLS-R CTC fine-tunes).
All other model types in the registry are native to onnx-asr.

Each registration is idempotent, and does nothing when the installed onnx-asr
provides the type. Thus the module becomes inert without a code change, and it can
be deleted when every supported onnx-asr release has the type.

The patch point is ``onnx_asr.loader.create_asr_resolver`` — the one function that
builds the name-to-class ``model_types`` mapping and gives it to ``Resolver``. If
that internal API changes, registration is skipped and the built-in model types
keep working.
"""

from ovos_utils.log import LOG

_SENTINEL = "_ovos_wav2vec2_patched"


def ensure_wav2vec2_ctc() -> None:
    """Idempotently register ``wav2vec2-ctc`` with the installed onnx-asr.

    Does nothing if the installed onnx-asr provides the type, if the patch is
    already applied, or if the internal resolver API cannot be found. In the last
    case the built-in model types stay untouched.
    """
    # Installed onnx-asr provides the type -> nothing to do.
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


def ensure_model_types() -> None:
    """Register every model type the plugin carries with the installed onnx-asr.

    This is the entry point for the plugin. Call it once at start-up. It is safe to
    call it more than once.
    """
    ensure_wav2vec2_ctc()
