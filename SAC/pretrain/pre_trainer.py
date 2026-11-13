# Source: SAC (https://github.com/lx-Meteors/SAC), modified for this repository. See THIRD_PARTY_NOTICES.md.
import random
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time
import json
from tqdm import tqdm
import argparse

from pre_prepare_data import get_examples
from model.modeling import get_model, save_adapter, load_adapter
from pre_dataloader import get_dataset, get_streaming_dataset

import logging
import wandb

from util.utils import get_wsd_scheduler, training_step, setup, count_parameters

# 配置日志，同时输出到屏幕和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='project',required=False, help='Directory including the configuration file, for saving model')
    parser.add_argument('--port', type=str, default='14527', required=False, help='port for ddp training')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=None, required=False, help='Local rank passed by torchrun when present')
    return parser.parse_args()


def _count_active_target_tokens(inputs, task_config):
    tokens = 0
    if task_config.get("use_ae_loss", False) and inputs.get("ae_targets") is not None:
        tokens += int(inputs["ae_targets"].numel())
    if task_config.get("use_lm_loss", False) and inputs.get("lm_targets") is not None:
        tokens += int(inputs["lm_targets"].numel())
    return tokens


def _global_sum_int(value, device):
    if isinstance(device, int):
        device = torch.device("cuda", device)
    tensor = torch.tensor(int(value), device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _estimate_training_steps(training_config, task_config, data_config, token_budget):
    if not token_budget:
        return None
    avg_len = (int(data_config["min_len"]) + int(data_config["max_len"])) / 2
    # SAC splits each document into compressed prefix + continuation target.
    # AE supervises the prefix; LM supervises the continuation.
    target_tokens_per_sample = 0
    if task_config.get("use_ae_loss", False):
        target_tokens_per_sample += avg_len / 2 + 1
    if task_config.get("use_lm_loss", False):
        target_tokens_per_sample += avg_len / 2 + 1
    target_tokens_per_sample = max(1, target_tokens_per_sample)
    tokens_per_optimizer_step = (
        target_tokens_per_sample
        * int(training_config["batch_size_per_device"])
        * int(training_config["device_count"])
        * int(training_config["gradient_accumulation_steps"])
    )
    return max(1, int(np.ceil(int(token_budget) / tokens_per_optimizer_step)))



# Training process
def train(local_rank, args, world_size, global_rank=None):

    if global_rank is None:
        global_rank = local_rank

    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    if global_rank == 0:
        wandb_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", "boa-context-compression"),
            "name": os.environ.get("WANDB_NAME"),
            "mode": os.environ.get("WANDB_MODE", "offline"),
            "config": config,
        }
        if os.environ.get("WANDB_ENTITY"):
            wandb_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        wandb.init(**wandb_kwargs)

    setup(global_rank, world_size, args.port)
    torch.cuda.set_device(local_rank)

    training_config = config["pretrain_training_config"]
    task_config = config['pretrain_task_config']

    assert world_size == training_config["device_count"], "device_count wrong"
    assert training_config["total_batch_size"] == training_config['batch_size_per_device']*training_config["device_count"]*training_config["gradient_accumulation_steps"]
    assert training_config["chunk_size"] == task_config["chunk_size"]
    assert task_config["mem_size"]*task_config["compress_ratio"] == task_config["chunk_size"]

    config["data_config"]["model_id"] = training_config["model_id"]
    output_dir = "output"
    cache_dir = os.path.join(args.work_dir, "cache", "pretrain")
    config["data_config"]["output_dir"] = cache_dir
    token_budget = int(training_config.get("token_budget") or 0)
    config["data_config"]["target_train_tokens"] = token_budget if token_budget > 0 else None
    config["data_config"]["eval_samples"] = training_config.get("eval_samples", config["data_config"].get("eval_samples", 1000))
    config["data_config"]["use_ae_loss"] = task_config.get("use_ae_loss", True)
    config["data_config"]["use_lm_loss"] = task_config.get("use_lm_loss", True)
    use_streaming = bool(training_config.get("streaming", config["data_config"].get("streaming", token_budget > 0)))

    train_examples = None
    if use_streaming:
        training_steps = _estimate_training_steps(training_config, task_config, config["data_config"], token_budget)
        if training_steps is None:
            raise ValueError("SAC streaming pretraining requires pretrain_training_config.token_budget.")
        if global_rank == 0:
            logging.info(
                f"[INFO] streaming FineWeb online | token_budget={token_budget:,} | "
                f"estimated_optimizer_steps={training_steps}"
            )
    else:
        train_examples, eval_examples = get_examples(**config["data_config"])

        # cal the total step
        training_steps = len(train_examples)//training_config["total_batch_size"]
        if training_steps <= 0:
            raise RuntimeError("Not enough SAC examples for one global batch. Lower total_batch_size or collect more data.")

        # drop last examples
        train_examples = train_examples[:training_steps*training_config["total_batch_size"]]
        if global_rank==0:
            logging.info(f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}")

        indices = [i for i in range(len(train_examples))]

        # The data is interleaved to obtain a batch containing samples of similar length.
        train_examples = train_examples[global_rank::world_size]

        logging.info(f"[INFO] global_rank{global_rank}/local_rank{local_rank} training examples: {indices[global_rank::world_size][:4]} ... {indices[global_rank::world_size][-4:]}| example_nums:{len(train_examples)} | training_steps:{training_steps}")

    model = get_model(training_config["model_id"], task_config, local_rank)
    
    # check non-frozen parameters
    if global_rank == 0:
        count_parameters(model, config)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # Instantiate the data loader
    if use_streaming:
        dataset = get_streaming_dataset(
            task_config["task_type"],
            config["data_config"],
            training_config['batch_size_per_device'],
            global_rank,
            world_size,
        )
    else:
        dataset = get_dataset(task_config["task_type"], train_examples, training_config['batch_size_per_device'])
    loader = DataLoader(dataset, batch_size=None)
    # Instantiate  optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=training_config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_wsd_scheduler(optimizer, training_steps)
    accumulation_steps = training_config["gradient_accumulation_steps"]
    step_num = 0
    
    optimizer.zero_grad()
    ddp_model.train()
    
    info_list = []
    start_time = time.time()
    tokens_processed = 0

    for epoch in range(1):

        def save():
            if global_rank!=0:
                return
            with open(os.path.join(args.work_dir,f"{output_dir}/info.json"),'w') as f:
                json.dump(info_list,f,indent=4)

            with open(os.path.join(args.work_dir,f"{output_dir}/config.json"),'w') as f:
                json.dump(config,f,indent=4)
                
            save_adapter(ddp_model.module,save_path_and_name=os.path.join(args.work_dir,f"{output_dir}/adapter.pt"))

        if global_rank == 0:
            if token_budget > 0:
                progress_bar = tqdm(total=token_budget, unit="tok", unit_scale=True, desc="SAC pretraining")
            else:
                progress_bar = tqdm(total=training_steps*accumulation_steps, desc="SAC pretraining")

        for inputs in loader:
            step_num += 1
            local_tokens = _count_active_target_tokens(inputs, task_config)
            global_tokens = _global_sum_int(local_tokens, local_rank)
            will_reach_budget = token_budget > 0 and (tokens_processed + global_tokens) >= token_budget
            should_sync = (step_num % accumulation_steps == 0) or will_reach_budget

            if should_sync:
                loss = training_step(ddp_model, inputs, local_rank, accumulation_steps)
            else:
                with ddp_model.no_sync():
                    loss = training_step(ddp_model, inputs, local_rank, accumulation_steps)

            tokens_processed += global_tokens
            info_list.append({
                "run_time(hours)": (time.time() - start_time) / 3600,
                "total_steps": training_steps,
                "steps": step_num / accumulation_steps,
                "tokens_processed": tokens_processed,
                "token_budget": token_budget if token_budget > 0 else None,
                "training_loss": loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            if global_rank == 0:
                wandb.log({
                    "run_time(hours)": (time.time() - start_time) / 3600,
                    "total_steps": training_steps,
                    "steps": step_num / accumulation_steps,
                    "tokens_processed": tokens_processed,
                    "token_budget": token_budget if token_budget > 0 else None,
                    "training_loss": loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                })

            if should_sync:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step_num % (training_config["log_step"] * accumulation_steps) == 0:
                if global_rank == 0:
                    logging.info(info_list[-1])
            if step_num % (training_config["save_step"] * accumulation_steps) == 0:
                save()
            if global_rank == 0:
                if token_budget > 0:
                    progress_bar.update(min(global_tokens, max(0, token_budget - progress_bar.n)))
                    progress_bar.set_postfix(tokens=f"{tokens_processed:,}")
                else:
                    progress_bar.update(1)

            if will_reach_budget:
                if global_rank == 0:
                    logging.info(f"[INFO] token budget reached: {tokens_processed:,}/{token_budget:,}")
                break
        if global_rank == 0:
            progress_bar.close()
        save()



# Launch training.
if __name__ == "__main__":
    args = parse_args()

    output_dir = "output"
    if not os.path.exists(os.path.join(args.work_dir, output_dir)):
        os.makedirs(os.path.join(args.work_dir, output_dir))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun/Slurm path: one Python process per GPU, with global ranks
        # provided by torchrun. This is required for multi-node training.
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
        train(local_rank, args, world_size, global_rank=global_rank)
    else:
        # Backward-compatible local path: one parent process spawns one worker per
        # visible GPU.
        world_size = torch.cuda.device_count()
        mp.spawn(
            train,
            args=(args, world_size),
            nprocs=world_size,
            join=True,
        )

"""
CUDA_VISIBLE_DEVICES=5,7 python ./pre_trainer.py --work_dir '../experiment/debug/quick' --port 14529
"""




