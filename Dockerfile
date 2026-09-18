# ovos-stt-plugin-onnx-asr served via ovos-stt-server.
#
# The opm.stt entry point (ovos-stt-plugin-onnx-asr) does nothing on its own
# -- it needs a server front-end, so this image runs ovos-stt-server with
# that engine on the server's default port.
FROM python:3.14-slim

RUN useradd -m -u 1000 ovos

WORKDIR /app
COPY . /app

# The onnxruntime extra is named here rather than in requirements.txt, so that
# a GPU deployment can install onnxruntime-gpu against the same plugin. The
# server floor is explicit and is a prerelease: the last stable server pins
# ovos-plugin-manager<1.0.0, which this plugin cannot satisfy, so an unpinned
# name silently backtracks to a stub release that carries no dependencies at
# all and produces an image that builds and then cannot start.
# The [mcp] extra pulls fastmcp (mounts the /mcp endpoint automatically --
# there is no CLI flag for it, unlike ovos-tts-server) so the deployed image
# exposes MCP without any local patching. The floor is 0.25.2a1: it is the
# OLDEST published version whose wheel METADATA declares
# `fastmcp<4,>=3; extra == "mcp"` (0.25.1a3 still pinned the old `mcp` SDK,
# whose FastMCP class was removed upstream in 2.0) AND whose
# ovos_stt_http_server/__init__.py resolves the STT plugin config from
# Configuration() when the server is started without an explicit config
# (0.25.1a4 has fastmcp but still passes config=None straight into the
# plugin, so mycroft.conf's stt.<plugin> section is silently ignored).
RUN pip install --no-cache-dir . "onnx-asr[cpu,hub]" "ovos-stt-http-server[mcp]>=0.26.0a1"

# Create the cache parents OWNED BY ovos before any bind mount lands on them.
# Docker creates missing mount parents as root, so if .cache is absent from
# the image the container gets a root-owned /home/ovos/.cache and the
# unprivileged server cannot create ~/.cache/huggingface inside it.
RUN mkdir -p /home/ovos/.cache/huggingface \
    && chown -R ovos:ovos /home/ovos/.cache /app

USER ovos
ENV HOME=/home/ovos
# Named explicitly rather than left to HOME resolution, so a cold fetch has a
# writable target and degrades to a slow download instead of a crash.
ENV HF_HOME=/home/ovos/.cache/huggingface
# Bound so onnxruntime does not grab every core on a shared host. Tune to the
# deployment; unset to let onnxruntime pick automatically.
ENV OMP_NUM_THREADS=4

EXPOSE 8080

# Models are NOT baked into this image; they download from HuggingFace into
# the mounted /home/ovos/.cache volume. The entrypoint prefetches the
# configured model(s) before the server starts taking traffic -- see
# docs/docker.md for the ONNX_ASR_PREFETCH_MODELS variable.
# --mcp mounts the MCP endpoint at /mcp. From 0.26.0a1 it is opt-in: installing
# the [mcp] extra no longer mounts it automatically, so every OVOS server now
# behaves the same way and the flag has to be passed explicitly.
ENTRYPOINT ["/app/docker-entrypoint.sh", "ovos-stt-server", \
            "--engine", "ovos-stt-plugin-onnx-asr", \
            "--mcp", \
            "--port", "8080", "--host", "0.0.0.0"]
