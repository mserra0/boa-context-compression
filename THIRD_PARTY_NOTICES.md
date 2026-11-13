# Third-party notices

This repository is released under the MIT License (see `LICENSE`). It contains or adapts the following third-party work, which keeps its own license.

The Activation Beacon, SAC and ICAE code on this branch consists of inference-only copies used by the latency benchmark (`benchmarks/latency/`). The full code used to train and evaluate each baseline is on its own branch of this repository (`baseline/activation-beacon`, `baseline/sac` and `baseline/icae`), and each of those branches has its own `THIRD_PARTY_NOTICES.md`.

## Activation Beacon

- Used in: `benchmarks/latency/beacon/` (inference-only subset of the Qwen3 Activation Beacon backend and its helpers, used by the latency benchmark).
- Upstream: <https://github.com/FlagOpen/FlagEmbedding> (`research/Long_LLM/activation_beacon`).
- License: MIT.
- Modifications: adapted to the benchmark (inference path only); see the headers of the files.
- Full training code: `src/baselines/activation_beacon/` on the `baseline/activation-beacon` branch.
- Paper: Zhang et al., "Long Context Compression with Activation Beacon", [arXiv:2401.03462](https://arxiv.org/abs/2401.03462).

```
MIT License

Copyright (c) 2022 staoxiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SAC

- Used in: `benchmarks/latency/sac/modeling.py` and `benchmarks/latency/lora.py`.
- Upstream: <https://github.com/lx-Meteors/SAC> — the upstream README indicates CC BY 4.0 ([license](https://creativecommons.org/licenses/by/4.0/)).
- Modifications: the `path_config` bootstrap is dropped, the KV-cache helpers are rewritten for newer Transformers versions, dtype handling and an on/off flag for the LoRA layers are added (marked `# BENCH:` in the files).
- Full training code: `SAC/` on the `baseline/sac` branch.
- Paper: Liu et al., "Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors", [arXiv:2510.08907](https://arxiv.org/abs/2510.08907).

## ICAE

- Used in: `benchmarks/latency/icae/` (inference path of the ICAE compressor, copied from the `baseline/icae` branch of this repository).
- The ICAE compressor here is a reimplementation of the method; no upstream code is included. The official repository is <https://github.com/getao/icae> (CC0-1.0).
- Full training code: `src/` on the `baseline/icae` branch (see `docs/baselines/icae.md` there).
- Paper: Ge et al., "In-context Autoencoder for Context Compression in a Large Language Model", ICLR 2024, [arXiv:2307.06945](https://arxiv.org/abs/2307.06945).

## MRQA 2019 evaluation metrics

- Used in: `src/evaluation/metrics.py` (vendored; span-overlap EM/F1 primitives).
- Upstream: <https://github.com/mrqa/MRQA-Shared-Task-2019> (`mrqa_official_eval.py`).
- License: MIT.

```
MIT License

Copyright (c) 2019 Machine Reading for Question Answering Workshop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
