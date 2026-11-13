import os
import sys
import argparse
import torch
import wandb
import math
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from colorama import init as colorama_init, Fore
from src import (CompressionDataset,
                 collate_fn,
                 collate_fn_mamba2,
                 load_mrqa_dataset,
                 PromptDestiller,
                 train,
                 save_checkpoint,
                 init_main,
                 init_services,
                 DistributedLengthGroupedSampler)
from src.dataset import OfflineCompressionDataset, collate_fn_offline
from src.utils.offline_kd import OfflineTeacherIndex

from transformers import get_cosine_schedule_with_warmup

torch.set_float32_matmul_precision('high')
colorama_init(autoreset=True)

def main(config: dict, device: torch.device):
    
    SEED = config['seed']
    if config['global_rank'] == 0: 
        init_services(config)
    else: 
        sys.stdout = open(os.devnull, 'w') 
    
    print("Initializing model...")
    
    compressor_type = config.get('compressor_type', 'mamba')

    destiller = PromptDestiller(
        mamba_model_name=config['mamba_model_name'],
        llm_model_name=config['llm_model_name'],
        device=device,
        compression_ratio=config['compression_ratio'],
        temperature=config['temperature'],
        max_new_tokens=config['max_new_tokens'],
        compressor_type=compressor_type,
        loss_type=config.get('loss_type', 'kl'),
        confidence_mode=config.get('confidence_mode', 'max_prob'),
        compile_llm=config.get('compile_llm', False),
        student_template=config.get('student_template', 'bare'),
        compress_question=config.get('compress_question', True),
    )
    print("Loading tokenizers...")
    llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'], padding_side='left')

    # Mamba2 weights (state-spaces/mamba2-*) do not ship a tokenizer.
    compressor_tokenizer_name = config.get('compressor_tokenizer_name') or (
        'EleutherAI/gpt-neox-20b' if compressor_type == 'mamba2' else config['mamba_model_name']
    )
    
    # clean_up_tokenization_spaces=False: the compressor tokenizer is BPE (GPT-NeoX)
    compressor_tokenizer = AutoTokenizer.from_pretrained(
        compressor_tokenizer_name, padding_side='left', clean_up_tokenization_spaces=False
    )

    print("Loading train and val datasets...")
    data_dir = config.get('data_dir', '/path/to/data/MRQA')
    train_dataset = load_mrqa_dataset(os.path.join(data_dir, 'train'), seed=SEED)
    val_dataset = load_mrqa_dataset(os.path.join(data_dir, 'dev'), seed=SEED)
    
    kd_cfg = config.get('kd') or {}
    offline_kd = bool(kd_cfg.get('offline', False))

    student_template = config.get('student_template', 'bare')
    compress_question = config.get('compress_question', True)

    max_samples = config.get('max_samples')
    if max_samples:
        # Compute val budget from the ORIGINAL sizes before any select().
        max_val_samples = max(1, round(max_samples * len(val_dataset) / len(train_dataset)))

        # Offline KD: the dump indexed the RAW train order, so train must NOT be
        # shuffled/subset here (handled below via Subset(range(total_examples))).
        # Val has no join constraint, so max_samples still applies to it even offline —
        # otherwise every offline run would validate on the full ~100k-example MRQA dev
        # set each epoch (val stays fully online: real teacher generation either way).
        if not offline_kd:
            train_dataset = train_dataset.shuffle(seed=SEED)
            train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

        val_dataset = val_dataset.shuffle(seed=SEED)
        val_dataset = val_dataset.select(range(min(max_val_samples, len(val_dataset))))

    train_data = CompressionDataset(train_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'], context=config['use_context'], student_template=student_template, compress_question=compress_question)
    val_data = CompressionDataset(val_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'], context=config['use_context'], student_template=student_template, compress_question=compress_question)

    if offline_kd:
        # Offline join: the dump indexed the RAW load_mrqa_dataset(train, SEED) order, so the
        # train_data above must NOT be shuffled/subset. Restrict to the dumped coverage.
        teacher_index = OfflineTeacherIndex(kd_cfg['dump_dir'], split='train')
        if len(train_data) > teacher_index.total_examples:
            train_data = torch.utils.data.Subset(train_data, range(teacher_index.total_examples))
        train_data = OfflineCompressionDataset(train_data, teacher_index)
        print(f"Offline KD: {teacher_index.total_examples} dumped examples, top_k={teacher_index.top_k}")

    # Adjust batch size and workers based on dataset size
    effective_batch_size = min(config['batch_size'], len(train_data))
    num_workers = min(8, len(train_data) // effective_batch_size) if len(train_data) > effective_batch_size else 0
    
    print(f"Using batch_size={effective_batch_size}, num_workers={num_workers}")
    
    # Mamba ot others: group by length to reduce left-padding waste.
    # Mamba2: standard distributed sampler (no padding in packed batches).
    # Offline KD: train_data is wrapped (Subset/OfflineCompressionDataset), which does not
    # expose the raw `.dataset` attribute DistributedLengthGroupedSampler needs for length
    # lookups — fall back to the plain sampler regardless of compressor_type.
    if compressor_type == 'mamba' or offline_kd:
        train_sampler = DistributedSampler(train_data, shuffle=True)
    else:
        train_sampler = DistributedLengthGroupedSampler(
            train_data,
            batch_size=effective_batch_size,
            shuffle=True,
        )
        
    val_sampler = DistributedSampler(val_data, shuffle=False)

    active_collate = collate_fn_mamba2 if compressor_type == 'mamba2' else collate_fn
    if offline_kd:
        _base_collate = active_collate
        train_collate = lambda batch: collate_fn_offline(
            batch, llm_tokenizer, compressor_tokenizer, base_collate=_base_collate,
            max_length=config['max_seq_len'])
    else:
        train_collate = lambda batch: active_collate(
            batch, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len']
        )

    train_loader = DataLoader(
        train_data,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=train_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda batch: active_collate(
            batch, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len']
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # broadcast_buffers=False prevents synchronizing frozen LLM buffers every step (speedup)
    destillerDDP = DDP(destiller, device_ids=[config['local_rank']], broadcast_buffers=False)
    
    optimizer = torch.optim.AdamW(
        destillerDDP.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'] 
    )

    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    
    decay_epochs = config.get('scheduler_epochs', config['num_epochs'])
    max_train_steps = decay_epochs * num_update_steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=max_train_steps
    )
    print("\nStarting training loop...")
    try:
        train(config, device, train_loader, val_loader, destillerDDP, optimizer, scheduler)
    except KeyboardInterrupt:
        print(Fore.RED + "\n\nInterrupting training...\n")
        wandb.finish()
        raise 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Context Compressor')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    args = parser.parse_args()
    
    try:
        config, device = init_main(args.config)
        main(config, device)
        destroy_process_group()
        print(Fore.GREEN + "\n\nTraining completed.")
        
    except KeyboardInterrupt:
        wandb.finish()
        destroy_process_group()
    except Exception as e:
        print(Fore.RED + f"\n\nTraining failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        wandb.finish()
