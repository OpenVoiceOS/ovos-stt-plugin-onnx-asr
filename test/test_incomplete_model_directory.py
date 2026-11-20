"""A download into a model directory that was cut short is finished.

onnx-asr downloads only into a directory that does not exist. Once it exists,
onnx-asr loads from it and never asks the network again, so a download that
was interrupted failed on every start. The plugin now fetches the files that
the download had started and not finished. It fetches nothing else, and it
never touches a file that is in the directory.
"""
import importlib.util
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
FULL = ("config.json", *GRAPHS, WEIGHTS)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
NEWEST = "fedcba9876543210fedcba9876543210fedcba98"   # where the repository is today
REPOSITORY = "someone/some-model"


def bookkeeping(directory, *parts):
    return os.path.join(directory, ".cache", "huggingface", *parts)


class _Locks:
    """Stands for the ``filelock`` module: a lock is free or held, by path."""

    class Timeout(Exception):
        pass

    def __init__(self):
        self.held = set()
        self.tried = []
        self.broken = set()         # paths where no lock file can be made

    def FileLock(self, path, timeout=-1):
        locks = self

        class Lock:
            def acquire(self):
                assert timeout == 0, "a lock must be tried, never waited for"
                locks.tried.append(path)
                if path in locks.broken:
                    raise PermissionError(13, "Permission denied", path)
                if path in locks.held:
                    raise locks.Timeout(path)
                locks.held.add(path)

            def release(self):
                locks.held.discard(path)

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *error):
                self.release()

        return Lock()


class _Hub:
    """What onnx-asr and the Hugging Face client do with a model directory.

    ``load_model`` loads from a directory as it is. A missing graph is
    ``FileNotFoundError``; missing weights are an error of the runtime, which
    is not one. ``hf_hub_download`` writes its file whether one is there or
    not, as the client does, so a test sees it when the plugin asks for a
    file that was in the directory.
    """

    def __init__(self, needs=FULL, repository=REPOSITORY, locks=None):
        self.needs = needs          # what the configured quantization loads
        self.repository = repository
        self.locks = locks
        self.online = True
        self.fetched = []           # (repository, file, revision) of each download
        self.loads = 0
        self.resolvers = 0
        self.asked_for_newest = 0
        self.first_downloads = []   # (directory, whether the download lock was held)

    @staticmethod
    def put(directory, finished, unfinished=(), commit=COMMIT):
        """A download that finished some files and stopped in the middle of others.

        The bookkeeping is the client's: a lock for each file it started,
        metadata for each file it finished, and the part that had arrived
        under a name of its own.
        """
        for name in finished:
            Path(directory, name).parent.mkdir(parents=True, exist_ok=True)
            Path(directory, name).write_text("first download of " + name)
        for name in (*finished, *unfinished):
            record = Path(bookkeeping(directory, "download", name))
            record.parent.mkdir(parents=True, exist_ok=True)
            record.with_name(record.name + ".lock").touch()
            if name in finished:
                record.with_name(record.name + ".metadata").write_text(f"{commit}\netag\n0.0\n")
            else:
                record.with_name(f"hash-of-{record.name}.etag.1a2b3c4d.incomplete").write_text("half")

    def load_model(self, model_id, path=None, **kwargs):
        self.loads += 1
        if not os.path.isdir(path):            # onnx-asr downloads into a directory that is not there
            self.first_downloads.append((path, path + ".lock" in self.locks.held))
            self.put(path, self.needs)
        for name in self.needs:
            if not os.path.isfile(os.path.join(path, name)):
                if name.endswith(".onnx") or name == "config.json":
                    raise FileNotFoundError(f"File '{name}' not found in path '{path}'")
                raise RuntimeError(f"[ONNXRuntimeError] : 1 : FAIL : cannot open {name}")
        model = MagicMock()
        model.asr.__class__.__name__ = "NemoConformerTDT"
        return model

    def create_asr_resolver(self, model, local_dir, *, offline=None):
        assert offline is None, "onnx-asr must not be asked to download: it replaces files"
        self.resolvers += 1
        return types.SimpleNamespace(repo_id=self.repository)

    def model_info(self, repo_id):
        if not self.online:
            raise ConnectionError("no route to the hub")
        self.asked_for_newest += 1
        return types.SimpleNamespace(sha=NEWEST)

    def hf_hub_download(self, repo_id, filename, *, revision=None, local_dir=None):
        if not self.online:
            raise ConnectionError("no route to the hub")
        assert revision, "a file was asked for without a commit: two files could then differ"
        self.fetched.append((repo_id, filename, revision))
        Path(local_dir, filename).parent.mkdir(parents=True, exist_ok=True)
        Path(local_dir, filename).write_text(f"resumed download of {filename}")
        record = Path(bookkeeping(local_dir, "download", filename))
        record.with_name(record.name + ".metadata").write_text(f"{revision or COMMIT}\netag\n1.0\n")
        return os.path.join(local_dir, filename)   # the old part stays behind, as it does


class TestIncompleteModelDirectory(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="onnxasr-incomplete-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.directory = os.path.join(self.root, "some-model")
        self.locks = _Locks()

    def _stt(self, hub, filelock=None):
        modules = patch.dict(sys.modules, {
            "onnx_asr.loader": types.SimpleNamespace(create_asr_resolver=hub.create_asr_resolver),
            "huggingface_hub": types.SimpleNamespace(hf_hub_download=hub.hf_hub_download,
                                                     HfApi=lambda: hub),
            "filelock": filelock or types.SimpleNamespace(FileLock=self.locks.FileLock,
                                                          Timeout=self.locks.Timeout),
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

    def _fails_as_before(self, hub, kind, cause):
        before = self._contents()
        with self.assertRaises(kind) as raised:
            self._stt(hub)
        self.assertIsInstance(raised.exception.__cause__, cause)
        self.assertEqual(self._contents(), before, "a file changed although nothing could be fetched")
        self.assertEqual((hub.fetched, hub.loads), ([], 1))

    def test_a_download_cut_short_is_finished(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        before = self._contents()
        stt = self._stt(hub)
        self.assertIn("some-model", stt._models)
        self.assertEqual(hub.fetched, [(REPOSITORY, GRAPHS[1], COMMIT), (REPOSITORY, WEIGHTS, COMMIT)])
        after = self._contents()
        for name, text in before.items():
            self.assertEqual(after[name], text, f"{name} did not stay as it was")

    def test_weights_that_stopped_half_way_are_fetched(self):
        # every graph is there, so onnx-asr finds nothing missing: the runtime fails,
        # and its error is not a FileNotFoundError
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,))
        self.assertIn("some-model", self._stt(hub)._models)
        self.assertEqual(hub.fetched, [(REPOSITORY, WEIGHTS, COMMIT)])

    def test_a_file_that_is_in_the_directory_is_never_fetched(self):
        # the bookkeeping calls the first graph unfinished, but the file is there:
        # somebody put it there, or the download stopped right after it arrived
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[1]), unfinished=(GRAPHS[0], WEIGHTS))
        Path(self.directory, GRAPHS[0]).write_text("mine")
        self._stt(hub)
        self.assertEqual(hub.fetched, [(REPOSITORY, WEIGHTS, COMMIT)])
        self.assertEqual(self._contents()[GRAPHS[0]], "mine")

    def test_when_no_file_was_finished_every_file_comes_from_one_newest_commit(self):
        hub = _Hub()
        hub.put(self.directory, (), unfinished=FULL)
        self._stt(hub)
        self.assertEqual(hub.fetched, [(REPOSITORY, name, NEWEST) for name in sorted(FULL)])
        self.assertEqual(hub.asked_for_newest, 1, "the commit must be asked once, not per file")

    def test_a_first_download_holds_the_download_lock(self):
        hub = _Hub(locks=self.locks)
        self.assertIn("some-model", self._stt(hub)._models)
        self.assertEqual(hub.first_downloads, [(self.directory, True)])
        self.assertEqual(self.locks.held, set(), "the lock was kept after the load")

    def test_a_first_download_goes_on_without_the_lock_as_it_always_did(self):
        for reason in ("held", "broken"):      # another server has it; no lock file can be made
            with self.subTest(reason=reason):
                shutil.rmtree(self.directory, ignore_errors=True)
                locks = _Locks()
                getattr(locks, reason).add(self.directory + ".lock")
                self.locks = locks
                hub = _Hub(locks=locks)
                self.assertIn("some-model", self._stt(hub)._models)
                self.assertEqual(hub.first_downloads, [(self.directory, reason == "held")])

    def test_a_file_in_a_subdirectory_of_the_repository(self):
        hub = _Hub(needs=("config.json", "onnx/encoder.onnx"))
        hub.put(self.directory, ("config.json",), unfinished=("onnx/encoder.onnx",))
        self.assertEqual(OnnxASR._unfinished_files(self.directory), ["onnx/encoder.onnx"])
        self._stt(hub)
        self.assertEqual(hub.fetched, [(REPOSITORY, "onnx/encoder.onnx", COMMIT)])
        self.assertIn(bookkeeping(self.directory, "download", "onnx", "encoder.onnx.lock"),
                      self.locks.tried)

    def test_a_complete_directory_asks_nothing(self):
        hub = _Hub()
        hub.put(self.directory, FULL)
        hub.online = False
        self._stt(hub)
        self.assertEqual((hub.loads, hub.resolvers, hub.fetched, self.locks.tried), (1, 0, [], []))
        self.assertEqual(os.listdir(self.root), ["some-model"], "something was written next to the model")

    def test_another_failure_asks_nothing(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,))
        self._stt(hub)                         # finishes the weights; the old part stays behind
        self.assertTrue(any(name.endswith(".incomplete")
                            for name in os.listdir(bookkeeping(self.directory, "download"))))
        hub.fetched.clear()
        self.locks.tried.clear()

        def out_of_memory(model_id, path=None, **kwargs):
            raise RuntimeError("CUDA out of memory")

        hub.load_model = out_of_memory
        with self.assertRaises(RuntimeError):
            self._stt(hub)
        self.assertEqual((hub.fetched, self.locks.tried), ([], []))

    def test_a_directory_filled_by_hand_is_left_alone(self):
        os.makedirs(self.directory)
        Path(self.directory, "config.json").write_text("mine")
        hub = _Hub()
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual((hub.resolvers, hub.fetched), (0, []))
        self.assertEqual(self._contents(), {"config.json": "mine"})

    def test_a_directory_with_another_quantization_fails_as_before(self):
        # nothing of the int8 files was ever started here: there is nothing to resume
        hub = _Hub(needs=("config.json", *INT8))
        hub.put(self.directory, FULL)
        before = self._contents()
        with self.assertRaises(FileNotFoundError):
            self._stt(hub)
        self.assertEqual((hub.resolvers, hub.fetched), (0, []))
        self.assertEqual(self._contents(), before)

    def test_without_network_the_first_error_is_raised_and_nothing_is_lost(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        hub.online = False
        self._fails_as_before(hub, FileNotFoundError, ConnectionError)

    def test_a_model_with_no_repository_keeps_its_error_and_its_files(self):
        hub = _Hub(repository=None)
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1],))
        self._fails_as_before(hub, FileNotFoundError, LookupError)

    def test_files_from_two_revisions_get_no_third_one(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json",), unfinished=(GRAPHS[1], WEIGHTS))
        hub.put(self.directory, (GRAPHS[0],), commit="f" * 40)
        self._fails_as_before(hub, FileNotFoundError, LookupError)

    def test_bookkeeping_that_names_no_commit_is_not_trusted(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,), commit="main")
        self._fails_as_before(hub, RuntimeError, LookupError)

    def test_while_another_process_downloads_a_file_nothing_is_fetched(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        self.locks.held.add(bookkeeping(self.directory, "download", WEIGHTS + ".lock"))
        self._fails_as_before(hub, FileNotFoundError, RuntimeError)

    def test_while_another_process_holds_the_download_lock_nothing_is_fetched(self):
        # it makes its first download, or it finishes this one
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        self.locks.held.add(self.directory + ".lock")
        self._fails_as_before(hub, FileNotFoundError, RuntimeError)

    def test_without_a_lock_nothing_is_fetched(self):
        hub = _Hub()
        hub.put(self.directory, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        self.locks.broken.add(self.directory + ".lock")
        self._fails_as_before(hub, FileNotFoundError, RuntimeError)

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "filelock is not installed")
    def test_with_the_real_locks(self):
        import filelock
        hub = _Hub()
        hub.put(self.directory, ("config.json", *GRAPHS), unfinished=(WEIGHTS,))
        for held in (self.directory + ".lock",                               # another server's download
                     bookkeeping(self.directory, "download", WEIGHTS + ".lock")):   # the client, on this file
            with self.subTest(held=os.path.basename(held)), filelock.FileLock(held):
                with self.assertRaises(RuntimeError) as raised:
                    self._stt(hub, filelock=filelock)
                self.assertIn("another process", str(raised.exception.__cause__))
                self.assertEqual(hub.fetched, [])
        self._stt(hub, filelock=filelock)      # the download died: now it is finished
        self.assertEqual(hub.fetched, [(REPOSITORY, WEIGHTS, COMMIT)])

    @unittest.skipIf(os.name == "nt" or os.geteuid() == 0,
                     "needs a directory this user cannot write to")
    def test_a_volume_that_cannot_be_written(self):
        hub = _Hub()
        hub.put(self.directory, FULL)
        broken = os.path.join(self.root, "broken", "some-model")
        hub.put(broken, ("config.json", GRAPHS[0]), unfinished=(GRAPHS[1], WEIGHTS))
        top = self.root
        for folder, _, _ in os.walk(top, topdown=False):
            os.chmod(folder, 0o555)
        self.addCleanup(lambda: [os.chmod(folder, 0o755) for folder, _, _ in os.walk(top)])
        # a complete model loads, and nothing is written next to it
        self.assertIn("some-model", self._stt(hub)._models)
        self.assertEqual(sorted(os.listdir(top)), ["broken", "some-model"])
        # an incomplete one keeps the error that names the missing file
        self.root, self.directory = os.path.join(top, "broken"), broken
        with self.assertRaises(FileNotFoundError) as raised:
            self._stt(hub)
        self.assertIn(GRAPHS[1], str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, PermissionError)


if __name__ == "__main__":
    unittest.main()
