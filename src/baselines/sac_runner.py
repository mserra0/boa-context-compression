"""SAC adapter for the repository-wide MRQA evaluation protocol."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


class SACRunner:
    """Expose a fine-tuned SAC checkpoint through EvalRunner's student API."""

    def __init__(self, config: dict, device: str = "cuda"):
        self.settings = dict(config)
        self.work_dir = Path(self.settings["work_dir"]).expanduser()
        self.output_dir = self.work_dir / "output"
        self.adapter_path = Path(
            self.settings.get("adapter_path", self.output_dir / "instruction_adapter.pt")
        ).expanduser()
        self.config_path = Path(
            self.settings.get("sac_config_path", self.output_dir / "config.json")
        ).expanduser()
        if not self.adapter_path.is_file():
            raise FileNotFoundError(f"SAC instruction adapter not found: {self.adapter_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"SAC config not found: {self.config_path}")

        self.device = torch.device(device)
        self.rank = self.device.index if self.device.type == "cuda" and self.device.index is not None else 0
        if self.device.type == "cuda":
            torch.cuda.set_device(self.rank)

        with open(self.config_path) as handle:
            self.sac_config = json.load(handle)
        training_config = self.sac_config["sft_training_config"]
        self.task_config = self.sac_config["sft_task_config"]
        self.model_id = training_config["model_id"]
        self.max_new_tokens = int(self.settings.get("max_new_tokens", 80))
        self.run_name = self.settings.get("run_name", f"sac_{self.work_dir.name}")
        self.config = {
            "llm_model_name": str(self.adapter_path),
            "teacher_model_name": self.model_id,
        }
        self.last_run_stats: list[dict] = []

        sac_root = Path(__file__).resolve().parents[2] / "SAC"
        if str(sac_root) not in sys.path:
            sys.path.insert(0, str(sac_root))
        from model.modeling import get_model, load_adapter

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = get_model(self.model_id, self.task_config, self.rank)
        self.model = load_adapter(model, save_path_and_name=str(self.adapter_path), log=False)
        self.model.eval().requires_grad_(False)

    def _build_inputs(self, sample: dict) -> dict[str, torch.Tensor]:
        context = sample.get("context", "")
        question = sample["question"]
        context_ids = (
            self.tokenizer("### Context:\n", add_special_tokens=False)["input_ids"]
            + self.tokenizer(context, add_special_tokens=False)["input_ids"]
        )
        question_ids = (
            self.tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"]
            + self.tokenizer(question, add_special_tokens=False)["input_ids"]
            + self.tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
        )
        return {
            "input_ids": torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0),
            "lm_targets": torch.tensor(question_ids, dtype=torch.long, device=self.device).unsqueeze(0),
        }

    @torch.no_grad()
    def _generate_one(self, sample: dict) -> str:
        inputs = self._build_inputs(sample)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.rank)
        started = time.perf_counter()
        output_ids = self.model.lm_inference(inputs, generate_num=self.max_new_tokens)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.rank)
        self.last_run_stats.append(
            {
                "input_tokens": int(inputs["input_ids"].shape[1] + inputs["lm_targets"].shape[1]),
                "output_tokens": len(output_ids),
                "latency_seconds": time.perf_counter() - started,
            }
        )
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def run_student(self, samples: list, batch_size: int = 1, progress=None, task_id=None) -> list[str]:
        del batch_size
        self.last_run_stats = []
        predictions = []
        for sample in samples:
            predictions.append(self._generate_one(sample))
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
        return predictions

    def run_teacher(self, samples: list, use_context: bool = True, batch_size: int = 1, progress=None, task_id=None):
        raise NotImplementedError("SACRunner only supports the student condition.")
