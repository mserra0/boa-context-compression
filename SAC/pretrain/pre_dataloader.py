# Source: SAC (https://github.com/lx-Meteors/SAC), modified for this repository. See THIRD_PARTY_NOTICES.md.
from torch.utils.data import IterableDataset
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from pre_prepare_data import _load_text_stream, _resolve_local_text_files


class CompressDataset(IterableDataset):
    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        self.examples = examples
        self.batch_size = batch_size

    def __iter__(self):
        input_ids = []
        ae_targets = []
        lm_targets = []
        for example in self.examples:
            input_ids.append(example["inputs"])
            ae_targets.append(example["ae_target"])
            lm_targets.append(example["lm_target"])

            if self.batch_size == len(input_ids):
                yield {
                    "input_ids": torch.stack(input_ids),
                    "ae_targets": torch.stack(ae_targets),
                    "lm_targets": torch.stack(lm_targets),
                }
                input_ids = []
                ae_targets = []
                lm_targets = []


class StreamingCompressDataset(IterableDataset):
    """Online SAC pretraining dataset.

    The upstream SAC implementation first materialized a list of examples before
    training.  For FineWeb-scale runs that is exactly the wrong shape: startup is
    long and caches can become enormous.  This streaming dataset shards the raw
    stream by distributed rank, tokenizes online, and yields SAC's expected
    ``input_ids / ae_targets / lm_targets`` batches directly.
    """

    def __init__(
        self,
        *,
        model_id,
        dataset_repo,
        min_len,
        max_len,
        batch_size,
        rank=0,
        world_size=1,
        text_column="text",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset_repo = dataset_repo
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.text_column = text_column

    def _build_example(self, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < self.min_len or len(ids) > self.max_len:
            return None

        split = len(ids) // 2
        inputs = ids[:split]
        ae_target = inputs + [self.tokenizer.eos_token_id]
        lm_target = ids[split:] + [self.tokenizer.eos_token_id]
        return {
            "inputs": torch.LongTensor(inputs),
            "ae_target": torch.LongTensor(ae_target),
            "lm_target": torch.LongTensor(lm_target),
        }

    def _row_stream(self):
        local_files = _resolve_local_text_files(self.dataset_repo)
        if local_files:
            # File-aware sharding for networked filesystems / parquet.  FineWeb has fewer parquet
            # shards than a 32N x 4GPU job has ranks, so multiple ranks share one
            # file and split rows within that file.  This avoids every rank
            # opening every parquet file.
            file_index = self.rank % len(local_files)
            file_path = local_files[file_index]
            ranks_for_file = [
                rank
                for rank in range(self.world_size)
                if rank % len(local_files) == file_index
            ]
            row_shard_index = ranks_for_file.index(self.rank)
            row_num_shards = len(ranks_for_file)
            dataset = load_dataset("parquet", data_files=[file_path], split="train", streaming=True)
            for raw_index, row in enumerate(dataset):
                if raw_index % row_num_shards == row_shard_index:
                    yield row
            return

        for raw_index, row in enumerate(_load_text_stream(self.dataset_repo)):
            if raw_index % self.world_size == self.rank:
                yield row

    def __iter__(self):
        input_ids = []
        ae_targets = []
        lm_targets = []
        for row in self._row_stream():
            text = row.get(self.text_column, "") if isinstance(row, dict) else ""
            if not text:
                continue
            # Cheap char-length filter before tokenization.  Keep it permissive;
            # exact filtering happens after tokenization.
            if len(text) > self.max_len * 6:
                continue
            example = self._build_example(text)
            if example is None:
                continue
            input_ids.append(example["inputs"])
            ae_targets.append(example["ae_target"])
            lm_targets.append(example["lm_target"])
            if len(input_ids) == self.batch_size:
                yield {
                    "input_ids": torch.stack(input_ids),
                    "ae_targets": torch.stack(ae_targets),
                    "lm_targets": torch.stack(lm_targets),
                }
                input_ids = []
                ae_targets = []
                lm_targets = []


def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)

    raise Exception("Don't exist [{task_type}] task.")


def get_streaming_dataset(task_type, data_config, batch_size, rank, world_size):
    if task_type == "Compress":
        return StreamingCompressDataset(
            model_id=data_config["model_id"],
            dataset_repo=data_config["dataset_repo"],
            min_len=data_config["min_len"],
            max_len=data_config["max_len"],
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            text_column=data_config.get("text_column", "text"),
        )

    raise Exception("Don't exist [{task_type}] task.")
