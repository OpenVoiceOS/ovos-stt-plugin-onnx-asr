# TODO

## Open issues

- [ ] #3 Model Requests
- [ ] #2 Error when defining language

## Gaps

- [ ] No test suite (no `tests/` dir, no pytest config).
- [ ] CI missing standard gh-automations workflows: no `coverage`, no `license-check`, no `opm-check` (this is an `opm.stt` plugin and should be validated by opm-check).
- [ ] `build_tests.yml` is a local inline job, not the shared `OpenVoiceOS/gh-automations` reusable build-tests workflow.
- [ ] Stale packaging: `setup.py` `url` points to `OpenVoiceOS/...` while the repo is at `TigreGotico/...`.
- [ ] Release workflows reference `TigreGotico/gh-automations/...@master`; org convention is `@dev`.
- [ ] No `.gitignore`; `ovos_stt_plugin_onnx_asr.egg-info/` build artifact present on disk (untracked, should be ignored).
- [ ] `available_languages` returns empty `set()` — no declared language support.
- [ ] `__main__` smoke block hardcodes a local WAV path; should be removed or replaced with a real test.

## Code TODOs

None found.
