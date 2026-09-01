# Docker

This repo builds and publishes `ghcr.io/openvoiceos/ovos-stt-plugin-onnx-asr`,
an image that runs the plugin behind
[`ovos-stt-server`](https://github.com/OpenVoiceOS/ovos-stt-server) on port
`8080`, using engine id `ovos-stt-plugin-onnx-asr`.

## Quick start

```bash
docker run -p 8080:8080 -v onnx-asr-cache:/home/ovos/.cache \
    ghcr.io/openvoiceos/ovos-stt-plugin-onnx-asr:dev
```

```bash
curl http://localhost:8080/status
```

See [ovos-stt-server](https://github.com/OpenVoiceOS/ovos-stt-server) for the
transcription request format.

## What's in the image

| | |
|---|---|
| Base | `python:3.12-slim` |
| Python | this plugin (local checkout) + `ovos-stt-server` |
| Port | `8080` |
| User | non-root, uid `1000` (`ovos`) |

## Model cache -- mount this

Models are **not** baked into the image; they download from HuggingFace into
the cache on first use and are kept there across restarts. Mount a
persistent volume over the whole cache directory:

```yaml
volumes:
  - onnx-asr-cache:/home/ovos/.cache
```

This is `/home/ovos/.cache`, the parent of both `~/.cache/huggingface` (raw
HF blobs) and onnx-asr's own model files. Mount the whole directory rather
than one subdirectory of it. It is created and `chown`ed to the `ovos` user at
build time, so a bind mount over an empty host directory does not leave it
root-owned and unwritable.

## Prefetching models

The entrypoint script downloads models before the server starts taking
traffic, so a fresh container is warm instead of blocking the first
transcription request on a cold download. Set `ONNX_ASR_PREFETCH_MODELS` to
a comma-separated list of onnx-asr model ids:

```bash
docker run -p 8080:8080 -v onnx-asr-cache:/home/ovos/.cache \
    -e ONNX_ASR_PREFETCH_MODELS="nemo-canary-1b-v2,gigaam-v2-rnnt" \
    ghcr.io/openvoiceos/ovos-stt-plugin-onnx-asr:dev
```

Left unset, the configured `model` is warmed -- the one from a mounted
`mycroft.conf`, or `nemo-canary-1b-v2` when nothing is configured -- so a
fresh container is warm for the model it actually starts with.
Set it to an empty string to skip prefetching entirely and let every model
download lazily on first request instead:

```bash
docker run -e ONNX_ASR_PREFETCH_MODELS="" ...
```

If you route different languages to different models with `lang2model` (see
[deployment.md](deployment.md)), list every model you expect to serve here so
none of them hits a cold-download penalty on first use.

## Building locally

```bash
docker build -t ovos-stt-plugin-onnx-asr .
```

The [`docker` workflow](../.github/workflows/docker.yml) builds a PR that
touches the image or its inputs (build-only, no push) and publishes to
`ghcr.io/openvoiceos/ovos-stt-plugin-onnx-asr` on pushes to `master`
(`latest`), `dev` (`dev`), and version tags.
