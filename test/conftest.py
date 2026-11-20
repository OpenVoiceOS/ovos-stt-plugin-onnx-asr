"""Shared test setup.

The suite never loads a real ASR model, so ``onnx_asr`` is stubbed for every
test that does not install its own fake. ``load_model`` is assigned rather than
left for the stub to conjure on first access, so ``unittest.mock.patch`` finds
a value to restore.

Test order is not fixed, so anything a test leaves behind in ``sys.modules``
surfaces as a failure in an unrelated test on some runs and not others. A test
that swaps the module out must put it back.
"""
import sys
from unittest.mock import MagicMock

if "onnx_asr" not in sys.modules:
    _stub = MagicMock()
    _stub.load_model = MagicMock()
    sys.modules["onnx_asr"] = _stub
