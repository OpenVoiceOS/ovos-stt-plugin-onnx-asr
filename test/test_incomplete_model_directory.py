"""A model directory that a download left incomplete is completed in place.

onnx-asr downloads only into a directory that does not exist. Once it exists,
onnx-asr loads from it and never asks the network again, so a download that
was interrupted failed on every start. The plugin now has the missing files
downloaded into the directory. It removes nothing.
"""
import fnmatch
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx_asr

from ovos_stt_plugin_onnxasr import OnnxASR

GRAPHS = ("encoder-model.onnx", "decoder_joint-model.onnx")
WEIGHTS = "encoder-model.onnx.data"   # kept in a file next to its graph
INT8 = ("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx")
REPOSITORY = ("config.json", *GRAPHS, WEIGHTS, *INT8)


class _Hub:
    """What onnx-asr and the Hugging Face client do with a model directory.

    ``load_model`` downloads into a directory that does not exist and loads
    from one that does, as it is. A missing graph is ``FileNotFoundError``;
    missing weights are an error of the runtime, which is not one.
    ``create_asr_resolver`` and ``snapshot_download`` are the two calls the
    plugin makes to complete a directory.
    """

    def __init__(self, repository=REPOSITORY, needs=("config.json", *GRAPHS, WEIGHTS)):
        self.repository = repository
        self.needs = needs          # what the configured quantization loads
        self.online = True
        self.downloaded = []        # every file fetched, in order
        self.loads = 0
        self.resolvers = 0
        self.asked = []             # (repository, patterns) of each snapshot_download

    @staticmethod
    def put(directory, files, unfinished=()):
        """A download that wrote ``files`` and stopped in the middle of ``unfinished``.

        The bookkeeping is the Hugging Face client's: a lock for each file it
        started, metadata for each file it finished, and the part that had
        arrived under a name of its own.
        """
        parts = os.path.join(directory, ".cache", "huggingface", "download")
        os.makedirs(parts, exist_ok=True)
        for name in files:
            Path(directory, name).write_text("first download of " + name)
            Path(parts, name + ".lock").touch()
            Path(parts, name + ".metadata").write_text("etag")
        for name in unfinished:
            Path(parts, name + ".lock").touch()
            Path(parts, f"hash-of-{name}.etag.1a2b3c4d.incomplete").write_text("half of " + name)

    def _download(self, directory, names):
        if not self.online:
            raise ConnectionError("no route to the hub")
        parts = os.path.join(directory, ".cache", "huggingface", "download")
        os.makedirs(parts, exist_ok=True)
        for name in names:
            target = Path(directory, name)
            assert not target.exists(), f"{name} was there and was downloaded again"
            target.write_text("later download of " + name)
            Path(parts, name + ".lock").touch()
            Path(parts, name + ".metadata").write_text("etag")   # the old part stays behind
            self.downloaded.append(name)

    def _graphs(self):
        return [name for name in self.needs if name.endswith(".onnx")]

    def load_model(self, model_id, path=None, **kwargs):
        self.loads += 1
        if not os.path.isdir(path):
            self._download(path, [name for name in self.needs if name in self.repository])
        for name in ("config.json", *self._graphs()):
            if not os.path.isfile(os.path.join(path, name)):
                raise FileNotFoundError(f"File '{name}' not found in path '{path}'")
        for name in self.needs:
            if not os.path.isfile(os.path.join(path, name)):
                raise RuntimeError(f"[ONNXRuntimeError] : 1 : FAIL : cannot open {name}")
        model = MagicMock()
        model.asr.__class__.__name__ = "NemoConformerTDT"
        return model

    def create_asr_resolver(self, model, local_dir, *, offline=None):
        self.resolvers += 1
        hub = self

        class Resolver:
            repo_id = "someone/" + model

            @staticmethod
            def resolve_model(*, quantization=None):
                assert offline is False, "onnx-asr only downloads when told it is not offline"
                wanted = ("config.json", *hub._graphs())
                hub._download(local_dir, [name for name in wanted if name in hub.repository
                                          and not os.path.isfile(os.path.join(local_dir, name))])
                for name in wanted:
                    if not os.path.isfile(os.path.join(local_dir, name)):
                        raise FileNotFoundError(f"File '{name}' not found in path '{local_dir}'")
                return {name: Path(local_dir, name) for name in wanted}

        return Resolver()

    def snapshot_download(self, repo_id, local_dir=None, allow_patterns=None, **kwargs):
        self.asked.append((repo_id, tuple(allow_patterns)))
        self._download(local_dir, [name for name in self.repository
                                   if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns)
                                   and not os.path.isfile(os.path.join(local_dir, name))])
        return local_dir


class TestIncompleteModelDirectory(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="onnxasr-incomplete-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.directory = os.path.join(self.root, "some-model")

    def _stt(self, hub):
        modules = patch.dict(sys.modules, {
            "onnx_asr.loader": types.SimpleNamespace(create_asr_resolver=hub.create_asr_resolver),
            "huggingface_hub": types.SimpleNamespace(snapshot_download=hub.snapshot_download),
        })
        modules.start()
        self.addCleanup(modules.stop)
        previous = onnx_asr.load_model
        onnx_asr.load_model = hub.load_model
        self.addCleanup(setattr, onnx_asr, "load_model", previous)
        return OnnxASR(config={"lang": "en-US", "model": "some-model",
                               "model_dir": self.root})

    def _contents(self):
        return {name: Path(self.directory, name).read_text()
                for name in os.listdir(self.directory) if name != ".cache"}

    def test_a_download_that_stopped_before_a_graph_is_completed_in_place(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]))
        before = self._contents()
        stt = self._stt(hub)
        self.assertIn("some-model", stt._models)
        self.assertEqual(hub.downloaded, [GRAPHS[1], WEIGHTS])
        after = self._contents()
        for name, text in before.items():
            self.assertEqual(after[name], text, f"{name} did not stay as it was")

    def test_weights_that_stopped_half_way_are_completed(self):
        # every graph is there, so onnx-asr finds nothing missing: the runtime fails,
        # and its error is not a FileNotFoundError
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,))
        stt = self._stt(hub)
        self.assertIn("some-model", stt._models)
        self.assertEqual(hub.downloaded, [WEIGHTS])
        self.assertEqual(hub.asked, [("someone/some-model",
                                      ("encoder-model.onnx?data", "decoder_joint-model.onnx?data"))])

    def test_a_part_left_behind_by_a_finished_download_is_no_reason_to_ask_again(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,))
        self._stt(hub)                         # completes the weights; the old part stays
        leftovers = os.listdir(os.path.join(self.directory, ".cache", "huggingface", "download"))
        self.assertTrue(any(name.endswith(".incomplete") for name in leftovers))

        def out_of_memory(model_id, path=None, **kwargs):
            raise RuntimeError("CUDA out of memory")

        hub.load_model = out_of_memory
        with self.assertRaises(RuntimeError):
            self._stt(hub)
        self.assertEqual(hub.resolvers, 1, "the network was asked again for a complete directory")

    def test_a_complete_directory_is_loaded_without_the_network(self):
        hub = _Hub()
        hub.put(self.directory, hub.needs)
        hub.online = False
        self._stt(hub)
        self.assertEqual((hub.loads, hub.resolvers, hub.asked), (1, 0, []))

    def test_another_quantization_is_added_next_to_the_first(self):
        hub = _Hub(needs=("config.json", *INT8))
        hub.put(self.directory, ("config.json", *GRAPHS, WEIGHTS))   # full precision, complete
        before = self._contents()
        self._stt(hub)
        self.assertEqual(hub.downloaded, list(INT8))
        after = self._contents()
        for name, text in before.items():
            self.assertEqual(after[name], text, f"{name} did not stay as it was")

    def test_without_network_the_first_error_is_raised_and_nothing_is_lost(self):
        hub = _Hub(needs=("config.json", *INT8))
        hub.put(self.directory, ("config.json", *GRAPHS, WEIGHTS))
        hub.online = False
        before = self._contents()
        with self.assertRaises(FileNotFoundError) as raised:
            self._stt(hub)
        self.assertIn(INT8[0], str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, ConnectionError)
        self.assertEqual(self._contents(), before)
        self.assertEqual(hub.loads, 1, "a second load was tried although nothing arrived")

    def test_a_model_with_no_repository_keeps_its_error_and_its_files(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]))

        def no_repository(model, local_dir, *, offline=None):
            raise ValueError("Model name or path must be specified")

        hub.create_asr_resolver = no_repository
        before = self._contents()
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual(self._contents(), before)

    def test_a_directory_filled_by_hand_is_left_alone(self):
        # no download bookkeeping in it: it is not ours to complete
        os.makedirs(self.directory)
        Path(self.directory, "config.json").write_text("mine")
        hub = _Hub()
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual((hub.resolvers, hub.downloaded), (0, []))
        self.assertEqual(self._contents(), {"config.json": "mine"})

    def test_a_repository_that_lacks_a_file_is_an_error_on_every_start(self):
        # e.g. ``quantization: int8`` for a model that ships no int8 graphs
        hub = _Hub(repository=("config.json", GRAPHS[0]), needs=("config.json", *GRAPHS))
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)                     # the first download
        self.assertEqual(hub.resolvers, 0, "a fresh download was followed by a second one")
        with self.assertRaises(FileNotFoundError) as raised:
            self._stt(hub)                     # the next start
        self.assertIn(GRAPHS[1], str(raised.exception))
        self.assertEqual((hub.resolvers, hub.loads), (1, 2))

    def test_another_failure_asks_the_network_nothing(self):
        hub = _Hub()
        hub.put(self.directory, hub.needs)

        def out_of_memory(model_id, path=None, **kwargs):
            raise RuntimeError("CUDA out of memory")

        hub.load_model = out_of_memory
        before = self._contents()
        with self.assertRaises(RuntimeError):
            self._stt(hub)
        self.assertEqual((hub.resolvers, hub.asked), (0, []))
        self.assertEqual(self._contents(), before)

    @unittest.skipIf(os.name == "nt" or os.geteuid() == 0,
                     "needs a directory this user cannot write to")
    def test_a_volume_that_cannot_be_written(self):
        hub = _Hub()
        hub.put(self.directory, hub.needs)
        broken = os.path.join(self.root, "broken", "some-model")
        hub.put(broken, ("config.json", GRAPHS[0]))
        for folder, _, _ in os.walk(self.root, topdown=False):
            os.chmod(folder, 0o555)
        top = self.root
        self.addCleanup(lambda: [os.chmod(folder, 0o755) for folder, _, _ in os.walk(top)])
        # a complete model loads, and nothing is written next to it
        self.assertIn("some-model", self._stt(hub)._models)
        self.assertEqual(sorted(os.listdir(self.root)), ["broken", "some-model"])
        # an incomplete one keeps the error that names the missing file
        self.root = os.path.join(self.root, "broken")
        with self.assertRaises(FileNotFoundError) as raised:
            self._stt(hub)
        self.assertIn(GRAPHS[1], str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, PermissionError)

    def test_weights_are_asked_for_next_to_each_graph_wherever_it_is(self):
        asked = []
        graphs = {"encoder": Path(self.directory, "onnx", "encoder.onnx"),
                  "decoder": Path(self.directory, "decoder.onnx"),
                  "vocab": Path(self.directory, "vocab.txt")}
        resolver = types.SimpleNamespace(repo_id="someone/some-model",
                                         resolve_model=lambda quantization=None: graphs)
        with patch.dict(sys.modules, {
            "onnx_asr.loader": types.SimpleNamespace(
                create_asr_resolver=lambda model, local_dir, offline=None: resolver),
            "huggingface_hub": types.SimpleNamespace(
                snapshot_download=lambda repo_id, **kwargs: asked.append((repo_id, kwargs))),
        }):
            OnnxASR._fetch_missing("some-model", self.directory, None)
        self.assertEqual(asked, [("someone/some-model", {
            "local_dir": self.directory,
            "allow_patterns": ["onnx/encoder.onnx?data", "decoder.onnx?data"]})])


if __name__ == "__main__":
    unittest.main()
