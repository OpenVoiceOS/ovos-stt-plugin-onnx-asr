# ovos-stt-plugin-onnx-asr served via ovos-stt-server.
#
# The opm.stt entry point (ovos-stt-plugin-onnx-asr) does nothing on its own
# -- it needs a server front-end, so this image runs ovos-stt-server with
# that engine on the server's default port.
FROM python:3.12-slim

RUN useradd -m -u 1000 ovos

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir . ovos-stt-http-server

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
ENTRYPOINT ["/app/docker-entrypoint.sh", "ovos-stt-server", \
            "--engine", "ovos-stt-plugin-onnx-asr", \
            "--port", "8080", "--host", "0.0.0.0"]
