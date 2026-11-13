# Source: SAC (https://github.com/lx-Meteors/SAC), modified for this repository. See THIRD_PARTY_NOTICES.md.
import sys
import os
import glob
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='../experiment/analysis_experiment/RARC_1B_MultiChunk', required=False, help='Directory including the configuration file')
    return parser.parse_args()


def _resolve_local_text_files(dataset_repo):
    """Return local text/parquet files when dataset_repo points to local data."""
    if not dataset_repo:
        return []
    expanded = os.path.expanduser(str(dataset_repo))
    if any(char in expanded for char in "*?[]"):
        return sorted(glob.glob(expanded))
    if os.path.isdir(expanded):
        return sorted(glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True))
    if os.path.isfile(expanded):
        return [expanded]
    return []


def _load_text_stream(dataset_repo):
    local_files = _resolve_local_text_files(dataset_repo)
    if local_files:
        return load_dataset("parquet", data_files=local_files, split="train", streaming=True)
    return load_dataset(dataset_repo, split="train", streaming=True)


def _active_target_tokens(example, *, use_ae_loss=True, use_lm_loss=True):
    total = 0
    if use_ae_loss:
        total += int(example["ae_target"].numel())
    if use_lm_loss:
        total += int(example["lm_target"].numel())
    return total


def get_examples(
    model_id,
    dataset_repo,
    samples_num,
    min_len,
    max_len,
    instruction_dataset_repo,
    output_dir,
    target_train_tokens=None,
    eval_samples=1000,
    use_ae_loss=True,
    use_lm_loss=True,
    text_column="text",
):
    """Materialize SAC pretraining examples, stopping as soon as enough are found.

    This cached path is kept for compatibility.  Large token-budget runs should
    use the online streaming dataset in ``pre_trainer.py``; this function now also
    stops early instead of collecting a full ``long_text.json`` first.
    """
    del instruction_dataset_repo
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_id.rstrip('/').split('/')[-1]
    target_tag = f"{int(target_train_tokens)}tok" if target_train_tokens else f"{int(samples_num)}samples"
    loss_tag = f"ae{int(bool(use_ae_loss))}_lm{int(bool(use_lm_loss))}"
    cache_stem = f"{model_name}_{target_tag}_{loss_tag}_{min_len}-{max_len}len_eval{int(eval_samples)}"
    train_data_name = os.path.join(output_dir, f"train_{cache_stem}.pt")
    eval_data_name = os.path.join(output_dir, f"eval_{cache_stem}.pt")

    if os.path.exists(train_data_name) and os.path.exists(eval_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data: train_data_name={train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_examples = []
    eval_examples = []
    train_tokens = 0
    target_train_tokens = int(target_train_tokens or 0)
    eval_samples = int(eval_samples or 0)

    for row in tqdm(_load_text_stream(dataset_repo), desc="Streaming/tokenizing examples"):
        text = row.get(text_column, "") if isinstance(row, dict) else ""
        if not text or len(text) > max_len * 6:
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < min_len or len(ids) > max_len:
            continue

        split = len(ids) // 2
        inputs = ids[:split]
        example = {
            "inputs": torch.LongTensor(inputs),
            "ae_target": torch.LongTensor(inputs + [tokenizer.eos_token_id]),
            "lm_target": torch.LongTensor(ids[split:] + [tokenizer.eos_token_id]),
        }

        if len(eval_examples) < eval_samples:
            eval_examples.append(example)
            continue

        train_examples.append(example)
        train_tokens += _active_target_tokens(
            example,
            use_ae_loss=use_ae_loss,
            use_lm_loss=use_lm_loss,
        )

        if target_train_tokens > 0:
            if train_tokens >= target_train_tokens:
                break
        elif len(train_examples) >= int(samples_num):
            break

    if not train_examples:
        raise RuntimeError("No SAC pretraining examples were produced. Check dataset path and length filters.")
    if target_train_tokens > 0 and train_tokens < target_train_tokens:
        print(
            f"[WARN] Dataset stream ended at {train_tokens:,} active target tokens, "
            f"below requested {target_train_tokens:,}."
        )

    print(
        f"Prepared SAC examples: train={len(train_examples):,}, eval={len(eval_examples):,}, "
        f"active_train_tokens={train_tokens:,}"
    )
    torch.save(train_examples, train_data_name)
    torch.save(eval_examples, eval_data_name)
    return train_examples, eval_examples


if __name__ == "__main__":
    args = parse_args()
    with open(args.work_dir + "/config.json") as f:
        config = json.load(f)

    training_config = config["pretrain_training_config"]
    task_config = config["pretrain_task_config"]
    config["data_config"]["model_id"] = training_config["model_id"]
    config["data_config"]["output_dir"] = os.path.join(args.work_dir, "cache", "pretrain")
    config["data_config"]["target_train_tokens"] = training_config.get("token_budget")
    config["data_config"]["eval_samples"] = training_config.get("eval_samples", config["data_config"].get("eval_samples", 1000))
    config["data_config"]["use_ae_loss"] = task_config.get("use_ae_loss", True)
    config["data_config"]["use_lm_loss"] = task_config.get("use_lm_loss", True)

    train_examples, eval_examples = get_examples(**config["data_config"])
