"""Registration of extra onnx-asr model types at runtime.

The plugin supports ASR architectures that a released `onnx-asr` does not carry
yet. The classes for them live in :mod:`._vendored`, copied from the onnx-asr fork
named there. :func:`ensure_model_types` registers them with the installed
`onnx-asr`, so that ``onnx_asr.load_model(...)`` loads those models through the
normal path.

Five rules govern the registration:

* The installed `onnx-asr` always wins. A model type it provides is never
  replaced, and a core behaviour it has is never patched.
* Families are independent. A family that fails to import or to patch is logged
  and skipped. The other families still register.
* Registration only adds. A later call adds the families an earlier call left out.
  A call never removes a model type an earlier call added.
* Built-in model types are never at risk. If the resolver internals of `onnx-asr`
  change, registration is skipped as a whole and `onnx-asr` is left untouched.
* A family is deleted wholesale from this package once every supported `onnx-asr`
  release carries it.

:mod:`._model_repos` holds a second table: the short name of each OpenVoiceOS model
mirror. :func:`ensure_model_types` adds it too, under the same rules.

The patch point is ``onnx_asr.loader.create_asr_resolver`` — the one function that
builds the name-to-class ``model_types`` mapping and gives it to ``Resolver``. The
mapping is captured, not re-listed, so new built-in model types keep working. The
capture runs once per process and the result is cached, so the patched function
does no capture work.
"""

import threading
from collections import namedtuple

from ovos_utils.log import LOG

_SENTINEL = "_ovos_model_types_patched"
_REGISTERED = "_ovos_registered_model_types"
_BASE_TYPES = "_ovos_base_model_types"

_LOCK = threading.RLock()
"""Serialises registration. Registration changes the `onnx_asr.loader` module."""


_Family = namedtuple("_Family", "name native_module build patch")


def _family(name, native_module, build, patch=None) -> _Family:
    return _Family(name, native_module, build, patch)


def _families() -> list:
    """Describe every model family this package carries.

    ``native_module`` is the `onnx-asr` module that provides the family natively;
    the family is skipped when that module imports. ``build`` imports the vendored
    classes and returns the model types to add, as a name-to-class mapping.
    ``patch`` adds the core behaviours the family needs; it runs only when the
    family really is registered. Both ``build`` and ``patch`` raise if the family
    cannot work here.
    """

    def build_espnet() -> dict:
        from ._vendored.espnet import EspnetAED, EspnetCtc

        return {"espnet-aed": EspnetAED, "espnet-ctc": EspnetCtc}

    def patch_espnet() -> None:
        from . import _core

        _core.patch_w2vbert_preprocessor()

    def build_granite_nar() -> dict:
        from ._vendored.granite_nar import GraniteNar

        return {"granite-nar": GraniteNar}

    def build_moonshine() -> dict:
        from ._vendored.moonshine import Moonshine

        return {"moonshine": Moonshine, "moonshine-tiny": Moonshine, "moonshine-base": Moonshine}

    def build_omnilingual() -> dict:
        from ._vendored.omnilingual import OmnilingualCtc

        return {"omnilingual-ctc": OmnilingualCtc}

    def build_paraformer() -> dict:
        from ._vendored.paraformer import Paraformer

        return {"paraformer": Paraformer}

    def build_sensevoice() -> dict:
        from ._vendored.sensevoice import SenseVoice

        return {"sensevoice": SenseVoice}

    def build_speech_llm() -> dict:
        from ._vendored.speech_llm import SpeechLlm

        return {"speech-llm": SpeechLlm}

    def build_wav2vec2_adapters() -> dict:
        from ._vendored.wav2vec2_adapters import Wav2Vec2Adapters

        return {"wav2vec2-adapters": Wav2Vec2Adapters}

    def patch_wav2vec2_adapters() -> None:
        from . import _core

        _core.patch_fetcher_support()

    def build_wav2vec2_ctc() -> dict:
        from .._wav2vec2 import Wav2Vec2Ctc

        return {"wav2vec2-ctc": Wav2Vec2Ctc}

    return [
        _family("espnet", "onnx_asr.models.espnet", build_espnet, patch_espnet),
        _family("granite-nar", "onnx_asr.models.granite_nar", build_granite_nar),
        _family("moonshine", "onnx_asr.models.moonshine", build_moonshine),
        _family("omnilingual", "onnx_asr.models.omnilingual", build_omnilingual),
        _family("paraformer", "onnx_asr.models.paraformer", build_paraformer),
        _family("sensevoice", "onnx_asr.models.sensevoice", build_sensevoice),
        _family("speech-llm", "onnx_asr.models.speech_llm", build_speech_llm),
        _family(
            "wav2vec2-adapters",
            "onnx_asr.models.wav2vec2_adapters",
            build_wav2vec2_adapters,
            patch_wav2vec2_adapters,
        ),
        _family("wav2vec2-ctc", "onnx_asr.models.wav2vec2", build_wav2vec2_ctc),
    ]


def _is_native(module_name: str) -> bool:
    """Say if the installed onnx-asr provides the family itself.

    Only an import that works means the family is native. Every other outcome
    leaves the family to the vendored class, so that a native module which does not
    import does not take the family away from the user.
    """
    import importlib

    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    except Exception as err:
        LOG.warning(f"onnx-asr module '{module_name}' does not import ({err}); the vendored model type is used")
        return False
    return True


def _register_model_repos() -> None:
    """Give the OpenVoiceOS model mirrors their short names.

    The table lives in :mod:`._model_repos`. A mapping the installed onnx-asr has is
    never replaced, and a mapping for a model type nobody can load does nothing, so
    this is safe whichever families register.
    """
    from . import _core
    from ._model_repos import MODEL_REPOS

    _core.add_model_repos(MODEL_REPOS)


def _collect_model_types(installed: dict, registered: dict, only=None) -> dict:
    """Prepare every family and return the model types to add.

    One family per ``try`` block, so a failure stops that family only. A family is
    built, and its core patches applied, only when it really adds a model type.

    Args:
        installed: The model types the installed onnx-asr has.
        registered: The model types an earlier call already added.
        only: Prepare these families only. All families by default.
    """
    extra: dict = {}
    for family in _families():
        if only is not None and family.name not in only:
            continue
        if _is_native(family.native_module):
            continue  # installed onnx-asr provides it -> leave it alone
        try:
            types = family.build()
        except Exception as err:
            LOG.warning(f"onnx-asr model family '{family.name}' is not available: {err}")
            continue
        # the installed onnx-asr, and everything an earlier call added, wins
        wanted = {
            name: model_class
            for name, model_class in types.items()
            if name not in installed and name not in registered and name not in extra
        }
        if not wanted:
            continue
        if family.patch is not None:
            try:
                family.patch()
            except Exception as err:
                LOG.warning(f"onnx-asr model family '{family.name}' is not available: {err}")
                continue
        extra.update(wanted)
    return extra


def _capture_base_model_types(loader, original) -> dict:
    """Return the ``model_types`` mapping the original resolver factory builds.

    The mapping is a local variable, rebuilt on every call and never exposed. It is
    captured — without re-listing every built-in type, so the plugin stays correct
    as onnx-asr adds new ones — by letting the original build it and recording the
    mapping it gives to ``Resolver``.

    The recording is done on a copy of the original function that sees a spy in
    place of ``Resolver``. The copy has globals of its own, so no other thread can
    reach the spy. The module attribute is swapped only when the copy records
    nothing, which is what a factory that reads ``Resolver`` from the module at call
    time does.
    """
    captured: dict = {}

    class _Spy:
        def __init__(self, model_types, *args, **kwargs):
            captured.update(model_types)

    def _run(func) -> None:
        try:
            func("whisper")  # the spy stops the work here: no network, no validation
        except Exception:
            pass

    try:
        import types as _types

        copy = _types.FunctionType(
            original.__code__,
            {**original.__globals__, "Resolver": _Spy},
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        copy.__kwdefaults__ = original.__kwdefaults__
        _run(copy)
    except Exception:
        pass
    if captured:
        return captured

    real_resolver = loader.Resolver
    loader.Resolver = _Spy
    try:
        _run(original)
    finally:
        loader.Resolver = real_resolver
    return captured


def ensure_model_types(only=None) -> None:
    """Register every model type the plugin carries with the installed onnx-asr.

    This is the entry point for the plugin. Call it once at start-up. It is safe to
    call it more than once, and safe to call it from more than one thread. A call
    that names some families only still lets a later call add the others.

    Args:
        only: Register these families only. All families by default.
    """
    with _LOCK:
        try:
            import onnx_asr.loader as loader
        except Exception:
            return

        original = getattr(loader, "create_asr_resolver", None)
        if original is None:
            return

        if getattr(original, _SENTINEL, False):
            # Already patched. The patched factory reads the record below on every
            # call, so a later call only has to add to that record.
            registered = getattr(original, _REGISTERED)
            base = getattr(original, _BASE_TYPES)
            patched = True
        else:
            registered = {}
            base = _capture_base_model_types(loader, original)
            if not base:
                # Resolver API drifted — don't risk breaking built-in models.
                LOG.warning("onnx-asr resolver internals changed; extra model types are not registered")
                return
            patched = False

        try:
            _register_model_repos()
        except Exception as err:
            LOG.warning(f"the short names of the OpenVoiceOS model mirrors are not registered: {err}")

        extra = _collect_model_types(base, registered, only)
        if not extra:
            return
        registered.update(extra)
        LOG.debug(f"registered onnx-asr model types: {', '.join(sorted(extra))}")
        if patched:
            return

        def create_asr_resolver(model=None, local_dir=None, *, offline=None):
            model_types = dict(base)
            for type_name, model_class in registered.items():
                model_types.setdefault(type_name, model_class)
            return loader.Resolver(model_types, model, local_dir, offline=offline)

        setattr(create_asr_resolver, _SENTINEL, True)
        setattr(create_asr_resolver, _REGISTERED, registered)
        setattr(create_asr_resolver, _BASE_TYPES, base)
        loader.create_asr_resolver = create_asr_resolver
