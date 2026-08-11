# Deployment

This page covers running the plugin as a long-lived STT server: persisting
the model cache, bounding memory, and checking that configuration actually
took effect. For the container image itself, see [docker.md](docker.md).

## Configuration

The server reads plugin config from `mycroft.conf`, under `stt`, keyed by the
engine id:

```json
{
  "stt": {
    "ovos-stt-plugin-onnx-asr": {
      "model": "nemo-canary-1b-v2",
      "lang2model": {
        "ru": "gigaam-v2-rnnt"
      },
      "max_loaded_models": 2
    }
  }
}
```

Mount it read-only into the container:

```yaml
volumes:
  - ./mycroft.conf:/home/ovos/.config/mycroft/mycroft.conf:ro
```

### Verifying config was applied

A `mycroft.conf` in the wrong location, or a typo in the engine id, is
silently ignored -- the plugin falls back to its defaults instead of erroring.
Check the container logs on startup: the plugin logs the model it loads
eagerly (`loading onnx-asr model: <id>`). If that id does not match `model`
in your config, the file was not read. Confirm the mounted path matches
`/home/ovos/.config/mycroft/mycroft.conf` exactly, and that the JSON parses
(a syntax error also falls back to defaults without raising).

## Cache persistence

Models live under `/home/ovos/.cache` inside the container and are
multi-gigabyte downloads. Mount a named volume over that whole path so
restarting or recreating the container does not re-download everything --
see [docker.md](docker.md#model-cache----mount-this).

## Prefetching

A cold model load blocks the first transcription request for that model
until the download finishes. Prefetch every model you plan to serve before
routing real traffic to the container -- see
[docker.md](docker.md#prefetching-models) for the `ONNX_ASR_PREFETCH_MODELS`
variable.

## Bounding memory

Each loaded onnx-asr model stays resident for the life of the process; by
default nothing evicts them. A deployment that routes many languages through
`lang2model` can end up with one multi-gigabyte model loaded per language
ever requested, with no upper bound.

Set `max_loaded_models` in the plugin config to cap how many models stay
loaded at once. When the cap is reached, the least-recently-used model is
evicted before a new one loads; a later request for the evicted language
reloads it (from the local cache, if prefetched, or from HuggingFace
otherwise). Leave it unset for an unbounded cache.

A value below 1, or one that is not a number, cannot bound anything. It is
reported in the log and treated as unset, rather than refusing to start the
server over a tuning knob.

**A cap bounds what the cache retains, not what is resident.** A model being
transcribed through is held by the thread using it, so eviction cannot
release it. The server answers requests from a threadpool, which means peak
memory is roughly the cap plus one model per request in flight. Size against
that number, not against the cap alone.

Pick the cap based on the container's memory budget and the size of the
models you route to. Pair it with a `mem_limit` on the container (see
[docker-compose.yml](../docker-compose.yml)) as a hard backstop — an
application-level budget states intent, and the container limit is what
enforces it.

## Healthchecks

Check `/status`, never a transcription endpoint. `/status` only reports
plugin metadata and returns immediately; a transcription request against a
model that is not yet cached can block for minutes on the download and would
report the container as unhealthy while it is actually just warming up.

```yaml
healthcheck:
  test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/status')"]
  interval: 60s
  timeout: 10s
  retries: 3
  start_period: 60s
```

The `start_period` gives the entrypoint's prefetch step room to finish before
failed healthchecks start counting against `retries`.
