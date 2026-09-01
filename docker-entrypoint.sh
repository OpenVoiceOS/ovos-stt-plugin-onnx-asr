#!/usr/bin/env bash
# Prefetches onnx-asr models into the HuggingFace cache before the server
# starts taking traffic, then hands off to it as PID 1 so signals (SIGTERM
# from `docker stop`) reach the server directly.
set -uo pipefail

# The server is exec'd only after prefetching, so until then this script is
# PID 1 -- and a PID 1 with no installed handler ignores SIGTERM. Without this
# trap, `docker stop` during a long download waits out the whole grace period
# and then SIGKILLs.
prefetch_pid=""
terminate() {
    [ -n "$prefetch_pid" ] && kill -TERM "$prefetch_pid" 2>/dev/null
    exit 143
}
trap terminate TERM INT

# Comma-separated onnx-asr model ids to warm before serving, e.g.
# "nemo-canary-1b-v2,gigaam-v2-rnnt". Left unset, the configured model is
# warmed, so a fresh container is ready for the model it actually starts
# with. Set to an empty string to skip prefetching entirely and download
# everything lazily on first request instead.
# Set-but-empty means "skip"; unset means "warm the configured model". The
# two cases must be told apart before any default is applied.
if [ -n "${ONNX_ASR_PREFETCH_MODELS+is_set}" ] && [ -z "$ONNX_ASR_PREFETCH_MODELS" ]; then
    skip_prefetch=1
else
    skip_prefetch=0
fi

if [ "$skip_prefetch" = "0" ]; then
    # Prefetch through the plugin rather than onnx_asr directly: the plugin
    # registers extra model types and applies the configured quantization, so
    # fetching without it downloads artifacts the server will not use, and
    # fails outright on model types onnx_asr does not know by itself. Ids are
    # passed as arguments, never interpolated into the program text.
    python3 - "${ONNX_ASR_PREFETCH_MODELS-}" <<'PYEOF' &
import sys

import onnx_asr
from ovos_config import Configuration

from ovos_stt_plugin_onnxasr._compat import ensure_model_types
from ovos_stt_plugin_onnxasr.defaults import quantization_for

PLUGIN_ID = "ovos-stt-plugin-onnx-asr"

# Same section the plugin reads, so prefetch and serving agree on the model.
config = Configuration().get("stt", {}).get(PLUGIN_ID, {})
quantization = config.get("quantization")

requested = [m.strip() for m in sys.argv[1].split(",") if m.strip()]
model_ids = requested or [config.get("model", "nemo-canary-1b-v2")]

ensure_model_types()

for model_id in model_ids:
    print(f"prefetching onnx-asr model: {model_id}", flush=True)
    try:
        # quantization_for, not the raw setting: a registry model whose
        # repository holds fp32 weights only must be fetched as fp32, or the
        # prefetch asks for a file that does not exist and the model the
        # server will actually load stays uncached.
        onnx_asr.load_model(model_id,
                            quantization=quantization_for(model_id, quantization))
    except Exception as exc:
        # A model that cannot be fetched must not stop the server from
        # starting. The plugin falls back to the default model for a language
        # whose model is unavailable, and anything still missing downloads
        # lazily on first use, so serving degrades instead of failing.
        print(f"prefetch failed for '{model_id}': {exc}; continuing",
              file=sys.stderr, flush=True)
PYEOF
    prefetch_pid=$!
    wait "$prefetch_pid"
    prefetch_pid=""
fi

exec "$@"
