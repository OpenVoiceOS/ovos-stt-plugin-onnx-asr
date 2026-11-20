"""A model directory that a download left incomplete is downloaded again.

onnx-asr downloads only into a directory that does not exist. Once it exists,
onnx-asr loads from it and never asks the network again, so a download that
was interrupted failed with ``ModelFileNotFoundError`` on every start.
"""
import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

import onnx_asr

from ovos_stt_plugin_onnxasr import OnnxASR

NEEDED = ("config.json", "encoder-model.onnx", "decoder_joint-model.onnx")


class _Hub:
    """What onnx-asr and the Hugging Face download do with a model directory.

    A directory that does not exist is downloaded: the files and the
    download's own bookkeeping under ``.cache/huggingface``. A directory that
    exists is loaded as it is, and a missing file is ``FileNotFoundError``.
    """

    def __init__(self, files=NEEDED):
        self.files = files
        self.downloads = 0
        self.loads = 0

    def download(self, directory, files=None):
        os.makedirs(os.path.join(directory, ".cache", "huggingface"))
        for name in self.files if files is None else files:
            with open(os.path.join(directory, name), "w") as handle:
                handle.write("weights")

    def load_model(self, model_id, path=None, **kwargs):
        self.loads += 1
        if not os.path.isdir(path):
            self.downloads += 1
            self.download(path)
        for name in NEEDED:
            if not os.path.isfile(os.path.join(path, name)):
                raise FileNotFoundError(f"File '{name}' not found in path '{path}'")
        model = MagicMock()
        model.asr.__class__.__name__ = "NemoConformerTDT"
        return model


class TestIncompleteModelDirectory(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="onnxasr-incomplete-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.directory = os.path.join(self.root, "some-model")

    def _stt(self, hub):
        previous = onnx_asr.load_model
        onnx_asr.load_model = hub.load_model
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        return OnnxASR(config={"lang": "en-US", "model": "some-model",
                               "model_dir": self.root})

    def test_a_download_that_was_interrupted_is_made_again(self):
        hub = _Hub()
        hub.download(self.directory, files=NEEDED[:2])   # stopped before the last file
        stt = self._stt(hub)
        self.assertEqual(hub.downloads, 1, "the directory was not downloaded again")
        self.assertIn("some-model", stt._models)
        for name in NEEDED:
            self.assertTrue(os.path.isfile(os.path.join(self.directory, name)))

    def test_a_complete_directory_is_loaded_as_it_is(self):
        hub = _Hub()
        hub.download(self.directory)
        self._stt(hub)
        self.assertEqual((hub.downloads, hub.loads), (0, 1))

    def test_a_directory_filled_by_hand_is_left_alone(self):
        # no download bookkeeping in it: there may be no repository to fetch it from
        os.makedirs(self.directory)
        with open(os.path.join(self.directory, "config.json"), "w") as handle:
            handle.write("mine")
        hub = _Hub()
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual(hub.downloads, 0)
        with open(os.path.join(self.directory, "config.json")) as handle:
            self.assertEqual(handle.read(), "mine")

    def test_a_repository_that_lacks_the_file_is_downloaded_once(self):
        # e.g. ``quantization: int8`` for a model that ships no int8 weights
        hub = _Hub(files=NEEDED[:2])
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual(hub.downloads, 1, "a fresh download that lacks a file was repeated")

    def test_a_directory_still_incomplete_after_the_second_download_is_an_error(self):
        hub = _Hub(files=NEEDED[:2])
        hub.download(self.directory, files=NEEDED[:1])
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual(hub.downloads, 1, "it must be tried once more, not for ever")

    def test_another_failure_removes_nothing(self):
        hub = _Hub()
        hub.download(self.directory)

        def out_of_memory(model_id, path=None, **kwargs):
            raise RuntimeError("CUDA out of memory")

        previous = onnx_asr.load_model
        onnx_asr.load_model = out_of_memory
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        with self.assertRaises(RuntimeError):
            OnnxASR(config={"lang": "en-US", "model": "some-model", "model_dir": self.root})
        for name in NEEDED:
            self.assertTrue(os.path.isfile(os.path.join(self.directory, name)))

    def test_only_the_model_s_own_directory_is_removed(self):
        hub = _Hub()
        other = os.path.join(self.root, "other-model")
        hub.download(other)
        hub.download(self.directory, files=NEEDED[:1])
        self._stt(hub)
        for name in NEEDED:
            self.assertTrue(os.path.isfile(os.path.join(other, name)))

    @unittest.skipIf(os.name == "nt", "no advisory file locks on Windows")
    def test_a_second_loader_waits_for_the_first_one_s_download(self):
        """Two servers on one volume: the second must not find the first one's
        half-finished download, take it for a broken one and remove it."""
        hub = _Hub()
        started, finish = threading.Event(), threading.Event()
        removed = []

        def slow_download(directory, files=None):
            os.makedirs(os.path.join(directory, ".cache", "huggingface"))
            with open(os.path.join(directory, NEEDED[0]), "w") as handle:
                handle.write("weights")
            started.set()
            finish.wait(10)
            for name in NEEDED[1:]:
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write("weights")

        hub.download = slow_download
        rmtree = shutil.rmtree

        def watching(path, *args, **kwargs):
            removed.append(path)
            return rmtree(path, *args, **kwargs)

        shutil.rmtree = watching
        self.addCleanup(setattr, shutil, "rmtree", rmtree)
        previous = onnx_asr.load_model
        onnx_asr.load_model = hub.load_model
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        config = {"lang": "en-US", "model": "some-model", "model_dir": self.root}
        first = threading.Thread(target=OnnxASR, kwargs={"config": dict(config)})
        first.start()
        self.assertTrue(started.wait(10))
        second = threading.Thread(target=OnnxASR, kwargs={"config": dict(config)})
        second.start()
        second.join(0.5)
        self.assertTrue(second.is_alive(), "the second loader did not wait")
        finish.set()
        first.join(10)
        second.join(10)
        self.assertFalse(second.is_alive())
        self.assertEqual(removed, [], "a download in progress was removed")
        self.assertEqual(hub.downloads, 1)


if __name__ == "__main__":
    unittest.main()
