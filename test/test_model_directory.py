"""The model download must land in a plain directory, not the HF cache.

A model whose weights sit in an external ``.onnx_data`` file names it as a
sibling. In the Hugging Face cache every file is a blob under ``blobs/`` and
the snapshot holds symlinks, so onnxruntime 1.30 resolves the model to its
blob, looks for the sibling beside the blob, and refuses the load with
"External data path escapes model directory".

Measured on OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx in T-4461:
the cache snapshot fails with that error and the same files in one plain
directory load. 22 of the 47 models the registry names ship ``.onnx_data``.

onnx-asr passes ``path`` straight to ``snapshot_download`` as ``local_dir``,
which writes real files, so the fix is that the plugin always names one.
"""
import os
import unittest
from unittest.mock import MagicMock

import onnx_asr

from ovos_stt_plugin_onnxasr import OnnxASR


class TestModelDirectoryIsPlain(unittest.TestCase):

    def _loader(self):
        calls = []

        def load(model_id, **kwargs):
            calls.append((model_id, kwargs))
            model = MagicMock()
            model.asr.__class__.__name__ = "NemoConformerRnnt"
            return model

        previous = onnx_asr.load_model
        onnx_asr.load_model = load
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        return calls

    def test_load_names_a_directory_of_its_own(self):
        calls = self._loader()
        stt = OnnxASR(config={"lang": "en-US"})
        stt.get_model("nemo-parakeet-tdt-0.6b-v3")

        self.assertTrue(calls, "no model was loaded")
        # every load, not only the one this test asked for: the plugin also
        # loads a default, and a download into the cache is as broken there.
        for model_id, kwargs in calls:
            self.assertIn("path", kwargs,
                          f"'{model_id}' was left to download into the shared "
                          f"Hugging Face cache, where an external-data model "
                          f"cannot be loaded")
            self.assertTrue(kwargs["path"],
                            "path must name a directory, not an empty value")

    def test_the_directory_is_per_model_and_configurable(self):
        calls = self._loader()
        stt = OnnxASR(config={"lang": "en-US", "model_dir": "/tmp/onnxasr-x"})
        stt.get_model("OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx")
        path = calls[-1][1]["path"]

        self.assertTrue(path.startswith("/tmp/onnxasr-x"),
                        "model_dir in the config must name the root")
        # one path segment per model: a repo id must not become a nested
        # directory, and a bare id must not collide with a namespaced one.
        tail = os.path.relpath(path, "/tmp/onnxasr-x")
        self.assertNotIn(os.sep, tail)

        calls.clear()
        stt.get_model("whisper-base")
        self.assertNotEqual(calls[-1][1]["path"], path,
                            "two models must not share one directory")


if __name__ == "__main__":
    unittest.main()
