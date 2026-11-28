"""The installed distribution declares a model hub, and a runtime as an extra.

`onnx-asr` keeps every runtime behind an extra: `onnxruntime` under `cpu`,
`onnxruntime-gpu` under `gpu`, `huggingface-hub` under `hub`. Its
unconditional requirements are `numpy` and `typing-extensions` alone. So a
plain `onnx_asr` requirement installs a package that cannot run, and this
plugin needs the hub for every model id it resolves.

The runtime stays an extra on this plugin, `[cpu]` or `[gpu]`, and must not be
unconditional. `onnxruntime` and `onnxruntime-gpu` both install the same
`onnxruntime` module, pip reports no conflict, and whichever wheel's files
land last on disk answers import requests: `get_available_providers()` can
drop to `['AzureExecutionProvider', 'CPUExecutionProvider']` with no install
order that is safe to rely on. A required `cpu` extra risks disabling CUDA on
a GPU deployment, silently. A missing runtime instead names itself: the entry
point raises `ModuleNotFoundError: No module named 'onnxruntime'` and
`ovos-plugin-manager` logs it.

The rest of the suite stubs `onnx_asr` in `conftest.py` and never loads a real
model, so no other test can see a missing runtime. This one reads the
packaging metadata instead, which is where the defect lives.
"""
import unittest
from importlib.metadata import metadata, requires

DIST = "ovos-stt-plugin-onnx-asr"


def _onnx_asr_requirements():
    """Every `onnx_asr` requirement string the distribution declares."""
    out = []
    for req in requires(DIST) or []:
        name = req.split()[0].split(";")[0].split("[")[0]
        name = name.split(">")[0].split("<")[0].split("=")[0].strip()
        if name.replace("-", "_") == "onnx_asr":
            out.append(req)
    return out


def _unconditional():
    """The `onnx_asr` requirements that install with no extra asked for."""
    return [r for r in _onnx_asr_requirements() if "extra ==" not in r]


def _for_extra(extra):
    return [r for r in _onnx_asr_requirements()
            if f'extra == "{extra}"' in r or f"extra == '{extra}'" in r]


class TestDeclaredBackend(unittest.TestCase):
    def test_the_distribution_requires_onnx_asr(self):
        """The control. Without it the assertions below pass on a typo."""
        self.assertTrue(_onnx_asr_requirements(),
                        f"{DIST} declares no onnx_asr requirement")

    def test_the_unconditional_requirement_names_the_model_hub(self):
        """Without it `onnx_asr.load_model` raises
        `ModuleNotFoundError: No module named 'huggingface_hub'` for any
        registry id, whichever runtime is installed."""
        reqs = _unconditional()
        self.assertTrue(reqs, f"{DIST} requires onnx_asr only behind an extra")
        self.assertTrue(any("hub" in r for r in reqs),
                        f"{reqs} name no hub extra; huggingface-hub is behind "
                        f"onnx-asr's 'hub' extra")

    def test_no_runtime_is_unconditional(self):
        """A required `cpu` extra puts `onnxruntime` beside `onnxruntime-gpu`
        on every GPU install and hides it, which disables CUDA with no error.
        """
        for req in _unconditional():
            self.assertNotIn("cpu", req,
                             f"{req} makes onnxruntime unconditional")
            self.assertNotIn("gpu", req,
                             f"{req} makes onnxruntime-gpu unconditional")

    def test_the_plugin_offers_a_runtime_extra_for_each_target(self):
        """`pip install ovos-stt-plugin-onnx-asr[cpu]` is the normal install,
        and `[gpu]` is the other one. Each must reach onnx-asr's own extra."""
        declared = (metadata(DIST).get_all("Provides-Extra") or [])
        for extra, runtime in (("cpu", "cpu"), ("gpu", "gpu")):
            with self.subTest(extra=extra):
                self.assertIn(extra, declared,
                              f"{DIST} declares no '{extra}' extra")
                reqs = _for_extra(extra)
                self.assertTrue(reqs,
                                f"the '{extra}' extra requires no onnx_asr")
                self.assertTrue(any(f"[{runtime}]" in r for r in reqs),
                                f"{reqs} do not ask onnx_asr for [{runtime}]")

    def test_onnxruntime_is_importable_in_this_environment(self):
        """The metadata assertions above are about the declaration. This one
        is about the environment the suite runs in, so a green suite on a
        machine with no runtime cannot be mistaken for a working install. CI
        installs `.[cpu,test]`."""
        import onnxruntime  # noqa: F401

    def test_huggingface_hub_is_importable_in_this_environment(self):
        import huggingface_hub  # noqa: F401


if __name__ == "__main__":
    unittest.main()
