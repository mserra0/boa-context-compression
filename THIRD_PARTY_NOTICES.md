# Third-party notices

This repository is released under the MIT License (see `LICENSE`). It contains or adapts the following third-party work, which keeps its own license.

## SAC

- Used in: `SAC/` (model, pretraining, instruction fine-tuning and evaluation utilities; upstream comments are kept as they are).
- Upstream: <https://github.com/lx-Meteors/SAC> — the upstream README indicates CC BY 4.0 ([license](https://creativecommons.org/licenses/by/4.0/)).
- Modifications: Qwen3 integration, local FineWeb-Edu / MRQA data support, Slurm and torchrun launchers.
- Paper: Liu et al., "Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors", [arXiv:2510.08907](https://arxiv.org/abs/2510.08907).

## MRQA 2019 evaluation metrics

- Used in: `src/utils/evaluate_utils.py` (`normalize_answer` and the QA-F1 functions follow the official MRQA 2019 / SQuAD v1.1 evaluation scripts).
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
