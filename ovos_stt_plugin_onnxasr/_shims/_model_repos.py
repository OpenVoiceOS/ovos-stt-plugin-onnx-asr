"""Short names for the OpenVoiceOS model mirrors.

`onnx-asr` loads a model from a Hugging Face repository. The repository is named
either by its full id, or by a short name that `onnx-asr` maps to a repository.
`onnx-asr` maps the short names of the models it publishes itself.

This table is the plugin's own convenience layer over the ONNX conversions in the
[OpenVoiceOS](https://huggingface.co/OpenVoiceOS) organisation. It gives one short
name to one repository, so that the name alone loads a model:

    onnx_asr.load_model("sensevoice")

A key is a model type name, and a value is the repository of the conversion to use
for that name. :func:`ovos_stt_plugin_onnxasr._shims.ensure_model_types` adds every
entry to the table of the installed `onnx-asr`, and never replaces an entry that
`onnx-asr` has. A short name that `onnx-asr` maps keeps the repository `onnx-asr`
gives it.

The table is data only. Add a line to extend it; no other code changes.

Two rules for an entry:

* The repository must exist, and its ``config.json`` must declare the ``model_type``
  the key names.
* One repository per name. A model type with one conversion per language gets no
  entry, because no single name can name a model. Give the repository id for those.
"""

MODEL_REPOS = {
    "espnet-ctc": "OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-ctc-onnx",
    "espnet-aed": "OpenVoiceOS/inesc-id-ebranch-w2vbert2-ep-aed-onnx",
    "granite-nar": "OpenVoiceOS/granite-speech-4.1-2b-nar-onnx",
    "moonshine-tiny": "OpenVoiceOS/moonshine-tiny-onnx",
    "moonshine-base": "OpenVoiceOS/moonshine-base-onnx",
    "omnilingual-ctc": "OpenVoiceOS/omnilingual-asr-ctc-1b-onnx",
    "paraformer": "OpenVoiceOS/paraformer-zh-onnx",
    "sensevoice": "OpenVoiceOS/sensevoice-small-onnx",
    "speech-llm": "OpenVoiceOS/qwen3-asr-0.6b-onnx",
    "wav2vec2-adapters": "OpenVoiceOS/mms-1b-all-onnx",
}
"""Model type name -> Hugging Face repository of a ready-to-use conversion."""
