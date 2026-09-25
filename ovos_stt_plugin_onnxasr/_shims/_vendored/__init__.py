"""Model type classes copied from onnx-asr.

Upstream project: https://github.com/istupakov/onnx-asr (MIT License)
Source fork:      https://github.com/TigreGotico/onnx-asr

Every file here is a verbatim copy, with only its import lines adjusted. Do not
edit the code: re-sync it from the source commit instead, or delete the file once
the installed onnx-asr carries the family.

Every ``Source path`` header gives the path from the root of the source fork, not
from the ``src/onnx_asr`` package directory. The fork keeps the feature-extraction
build tooling in a ``preprocessors`` directory at the root, beside ``src``, so a
path that starts with ``preprocessors/`` and a path that starts with
``src/onnx_asr/preprocessors/`` name two different files.

:data:`SOURCE_COMMIT` is the single place that records which commit the copies come
from. Update it whenever the copies are re-synced.
"""

SOURCE_COMMIT = "8fd5f2b30fdd10d88066dc53a4c5558a208a0512"
