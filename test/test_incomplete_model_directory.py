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
from unittest.mock import MagicMock, patch

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

    def test_a_complete_model_loads_when_no_lock_file_can_be_made(self):
        hub = _Hub()
        hub.download(self.directory)
        denied = PermissionError(13, "Permission denied")
        with patch("ovos_stt_plugin_onnxasr.open", side_effect=denied, create=True) as opened:
            stt = self._stt(hub)
        opened.assert_called_once_with(self.directory + ".lock", "a")
        self.assertIn("some-model", stt._models)
        self.assertEqual((hub.downloads, hub.loads), (0, 1))

    @unittest.skipIf(os.name == "nt" or os.geteuid() == 0,
                     "needs a directory this user cannot write to")
    def test_a_complete_model_loads_from_a_read_only_volume(self):
        hub = _Hub()
        hub.download(self.directory)
        os.chmod(self.root, 0o555)
        self.addCleanup(os.chmod, self.root, 0o755)
        stt = self._stt(hub)
        self.assertIn("some-model", stt._models)
        self.assertEqual((hub.downloads, hub.loads), (0, 1))
        self.assertEqual(os.listdir(self.root), ["some-model"])

    def test_a_directory_that_cannot_be_removed_keeps_the_first_error(self):
        # a read-only volume: the reader must see which file is missing, not why rmtree failed
        hub = _Hub()
        hub.download(self.directory, files=NEEDED[:2])
        with patch("ovos_stt_plugin_onnxasr.shutil.rmtree",
                   side_effect=OSError(30, "Read-only file system")), \
                self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual((hub.downloads, hub.loads), (0, 1))
        self.assertTrue(os.path.isfile(os.path.join(self.directory, NEEDED[0])))

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
        loaded, failed = [], []

        def construct():
            # ``join`` does not raise what its thread raised: keep it for the assertions
            try:
                loaded.append(OnnxASR(config=dict(config)))
            except Exception as error:
                failed.append(error)

        first = threading.Thread(target=construct)
        second = threading.Thread(target=construct)
        first.start()
        try:
            self.assertTrue(started.wait(10))
            second.start()
            second.join(0.5)
            self.assertTrue(second.is_alive(), "the second loader did not wait")
        finally:
            # whatever failed above, no worker may outlive the patched globals
            finish.set()
            first.join(10)
            if second.ident is not None:
                second.join(10)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(failed, [])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(removed, [], "a download in progress was removed")
        self.assertEqual(hub.downloads, 1)


if __name__ == "__main__":
    unittest.main()
