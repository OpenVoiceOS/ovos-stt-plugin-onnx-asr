# Changelog

## [0.6.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.6.0a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.5.0a1...0.6.0a1)

**Merged pull requests:**

- feat: support fork model families \(espnet-ctc/aed, speech-llm/SALM\) [\#24](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/24) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.5.0a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.4.2a2...0.5.0a1)

**Merged pull requests:**

- feat: add cpu\_models\_only config flag to restrict model selection [\#34](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/34) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.2a2](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.4.2a2) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.4.2a1...0.4.2a2)

**Merged pull requests:**

- build: expose the MCP endpoint from the published image [\#32](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/32) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.2a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.4.2a1) (2026-08-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.4.1a1...0.4.2a1)

**Merged pull requests:**

- fix: drop the Whisper language hint when the vocabulary lacks the token [\#30](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/30) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.1a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.4.1a1) (2026-08-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.4.0a1...0.4.1a1)

**Merged pull requests:**

- fix: serve the configured model and drop registry entries that cannot serve [\#29](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/29) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.4.0a1) (2026-08-11)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.3.0a3...0.4.0a1)

**Merged pull requests:**

- feat: vendor onnx-asr model types so a plain pip install covers every architecture [\#26](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/26) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.0a3](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.3.0a3) (2026-08-11)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.3.0a2...0.3.0a3)

**Merged pull requests:**

- Add container deployment \(Dockerfile, compose, CI, docs\) and bound the model cache [\#25](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/25) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.0a2](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.3.0a2) (2026-08-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.3.0a1...0.3.0a2)

**Merged pull requests:**

- docs: refresh supported-model catalog [\#12](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/12) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.3.0a1) (2026-08-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.2.1a1...0.3.0a1)

**Merged pull requests:**

- feat: lang2model map for per-language multi-model serving [\#20](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/20) ([JarbasAl](https://github.com/JarbasAl))

## [0.2.1a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.2.1a1) (2026-08-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.2.0a1...0.2.1a1)

**Merged pull requests:**

- fix: normalize BCP-47 lang tags to primary subtag for onnx-asr [\#18](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/18) ([JarbasAl](https://github.com/JarbasAl))

## [0.2.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.2.0a1) (2026-07-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.1.1a2...0.2.0a1)

**Merged pull requests:**

- feat: load wav2vec2-ctc models via runtime onnx-asr shim [\#13](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/13) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.1a2](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.1.1a2) (2026-07-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.1.1a1...0.1.1a2)

**Merged pull requests:**

- docs: add funding attribution [\#14](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/14) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.1a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.1.1a1) (2026-07-04)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.1.0a2...0.1.1a1)

**Merged pull requests:**

- fix: only pass language/target\_language to models that accept them [\#6](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/6) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.0a2](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.1.0a2) (2026-07-04)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.1.0a1...0.1.0a2)

**Merged pull requests:**

- Configure Renovate [\#1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/1) ([renovate[bot]](https://github.com/apps/renovate))

## [0.1.0a1](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/tree/0.1.0a1) (2026-06-23)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/compare/0.0.1...0.1.0a1)

**Merged pull requests:**

- feat: GPU execution provider support \(use\_cuda / providers\) [\#4](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr/pull/4) ([JarbasAl](https://github.com/JarbasAl))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
