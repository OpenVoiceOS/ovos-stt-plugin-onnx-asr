# Import the real onnx_asr before any test module runs: several test files do
# sys.modules.setdefault("onnx_asr", MagicMock()) as a fallback for environments
# without it, but when the real package is installed everything must patch the
# one real module or plugin and tests end up talking to different objects.
import onnx_asr  # noqa: F401
