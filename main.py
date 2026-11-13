import os
import sys
import argparse
import torch
import wandb
import math
from functools import partial
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
                 collate_fn_sft_mamba2,
                 collate_fn_sft_mamba1,
                 load_mrqa_dataset,
                 PromptDestiller,
                 MRQASupervisedTrainer,
                 train,
                 save_checkpoint,
                 init_main,
                 init_services,
                 DistributedLengthGroupedSampler,
                 DistributedTokenBudgetSampler,
                 LMPretrainer,
                 PretrainDataset,
                 collate_fn_pretrain,
                 collate_fn_pretrain_mamba1,
                 load_fineweb_dataset,
                 build_token_budget_loader,
                 resolve_stride,
                 pretrain)

from transformers import get_cosine_schedule_with_warmup

torch.set_float32_matmul_precision('high')
colorama_init(autoreset=True)

def _resolve_anchor_token_ids(config: dict):
    """Resolve the KV-anchor scaffold token ids from config.
    Prefers the list `anchor_init_token_ids`; falls back to the legacy singular
    `anchor_init_token_id`; returns None so the model applies its default scaffold."""
    ids = config.get('anchor_init_token_ids')
    if ids is not None:
        return list(ids)
    single = config.get('anchor_init_token_id')
    return [single] if single is not None else None

def main(config: dict, device: torch.device):
    if config.get('objective') == 'lm':
        return _main_pretrain(config, device)
    if config.get('objective') == 'sft':
        return _main_sft(config, device)

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
        stride=resolve_stride(config),
        temperature=config['temperature'],
        max_new_tokens=config['max_new_tokens'],
        compressor_type=compressor_type,
        loss_type=config.get('loss_type', 'kl'),
        confidence_mode=config.get('confidence_mode', 'max_prob'),
        compile_llm=config.get('compile_llm', False),
        student_template=config.get('student_template', 'bare'),
        compress_question=config.get('compress_question', True),
        prompt_mode=config.get('prompt_mode', 'embed'),
        anchor_init_token_ids=_resolve_anchor_token_ids(config),
        compressor_gradient_checkpointing=config.get('compressor_gradient_checkpointing', True),
        llm_dtype=getattr(torch, config.get('llm_dtype', 'bfloat16')),
        attn_implementation=config.get('attn_implementation', 'sdpa'),
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
    
    max_samples = config.get('max_samples')
    if max_samples:
        # Compute val budget from the ORIGINAL sizes before any select().
        max_val_samples = max(1, round(max_samples * len(val_dataset) / len(train_dataset)))

        train_dataset = train_dataset.shuffle(seed=SEED)
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

        val_dataset = val_dataset.shuffle(seed=SEED)
        val_dataset = val_dataset.select(range(min(max_val_samples, len(val_dataset))))
    
    student_template = config.get('student_template', 'bare')
    compress_question = config.get('compress_question', True)
    train_data = CompressionDataset(train_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'], context=config['use_context'], student_template=student_template, compress_question=compress_question)
    val_data = CompressionDataset(val_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'], context=config['use_context'], student_template=student_template, compress_question=compress_question)
    
    # Adjust batch size and workers based on dataset size
    effective_batch_size = min(config['batch_size'], len(train_data))
    num_workers = min(8, len(train_data) // effective_batch_size) if len(train_data) > effective_batch_size else 0
    
    print(f"Using batch_size={effective_batch_size}, num_workers={num_workers}")
    
    # Mamba ot others: group by length to reduce left-padding waste.
    # Mamba2: standard distributed sampler (no padding in packed batches).
    if compressor_type == 'mamba':
        train_sampler = DistributedSampler(train_data, shuffle=True)
    else:
        train_sampler = DistributedLengthGroupedSampler(
            train_data,
            batch_size=effective_batch_size,
            shuffle=True,
        )
        
    val_sampler = DistributedSampler(val_data, shuffle=False)

    active_collate = collate_fn_mamba2 if compressor_type == 'mamba2' else collate_fn

    # Opt-in token-budget packing (Mamba2 only); 'none' (default) uses the fixed-batch-size path.
    packing = config.get('packing', 'none')

    if compressor_type == 'mamba2' and packing == 'token_budget':
        train_loader = build_token_budget_loader(
            train_data, llm_tokenizer, compressor_tokenizer,
            tokens_per_batch=config['tokens_per_batch'],
            max_seq_len=config['max_seq_len'],
            max_seqs_per_batch=config.get('max_seqs_per_batch'),
            seed=config.get('seed', 42),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=effective_batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=lambda batch: active_collate(
                batch, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len']
            ),
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


def _main_sft(config, device):
    """MRQA supervised-finetuning ablation (objective: 'sft').

    Same student geometry, warm-start (`start_from`), optimizer, scheduler and DDP loop as
    the KD `main`, but trains the compressor on ground-truth MRQA answers with cross-entropy
    instead of distilling a teacher. Scope: Mamba2 + KV B-ctx (wrapped, compress_question
    False), matching the current MRQA-KV frontier so the KD-vs-SFT comparison is clean.
    """
    SEED = config['seed']
    if config['global_rank'] == 0:
        init_services(config)
    else:
        sys.stdout = open(os.devnull, 'w')

    compressor_type = config.get('compressor_type', 'mamba2')
    if compressor_type not in ('mamba', 'mamba2'):
        raise NotImplementedError(
            f"objective='sft' supports compressor_type in {{'mamba','mamba2'}} "
            f"(got {compressor_type!r}).")
    if config.get('prompt_mode') != 'kv':
        raise NotImplementedError(
            "objective='sft' currently supports prompt_mode='kv' only "
            "(B-ctx wrapped, the current MRQA-KV frontier).")

    print("Initializing MRQASupervisedTrainer...")
    model = MRQASupervisedTrainer(
        mamba_model_name=config['mamba_model_name'],
        llm_model_name=config['llm_model_name'],
        device=device,
        stride=resolve_stride(config),
        temperature=config.get('temperature', 2.0),          # ignored for the sft objective (no teacher)
        max_new_tokens=config.get('max_new_tokens', 50),     # ignored for the sft objective
        compressor_type=compressor_type,
        loss_type=config.get('loss_type', 'kl'),             # ignored for the sft objective
        confidence_mode=config.get('confidence_mode', 'max_prob'),
        compile_llm=config.get('compile_llm', False),
        student_template=config.get('student_template', 'wrapped'),
        compress_question=config.get('compress_question', False),
        prompt_mode=config.get('prompt_mode', 'kv'),
        anchor_init_token_ids=_resolve_anchor_token_ids(config),
    )

    print("Loading tokenizers...")
    llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'], padding_side='left')
    # Mamba2 weights don't ship a tokenizer (default to GPT-NeoX); Mamba1 uses its own.
    compressor_tokenizer_name = config.get('compressor_tokenizer_name') or (
        'EleutherAI/gpt-neox-20b' if compressor_type == 'mamba2' else config['mamba_model_name']
    )
    compressor_tokenizer = AutoTokenizer.from_pretrained(
        compressor_tokenizer_name, padding_side='left', clean_up_tokenization_spaces=False
    )

    print("Loading train and val datasets...")
    data_dir = config.get('data_dir', '/path/to/data/MRQA')
    train_dataset = load_mrqa_dataset(os.path.join(data_dir, 'train'), seed=SEED)
    val_dataset = load_mrqa_dataset(os.path.join(data_dir, 'dev'), seed=SEED)

    max_samples = config.get('max_samples')
    if max_samples:
        max_val_samples = max(1, round(max_samples * len(val_dataset) / len(train_dataset)))
        train_dataset = train_dataset.shuffle(seed=SEED).select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.shuffle(seed=SEED).select(range(min(max_val_samples, len(val_dataset))))

    student_template = config.get('student_template', 'wrapped')
    compress_question = config.get('compress_question', False)
    train_data = CompressionDataset(train_dataset, llm_tokenizer, compressor_tokenizer,
                                    max_length=config['max_seq_len'], context=config['use_context'],
                                    student_template=student_template, compress_question=compress_question,
                                    supervised=True)
    val_data = CompressionDataset(val_dataset, llm_tokenizer, compressor_tokenizer,
                                  max_length=config['max_seq_len'], context=config['use_context'],
                                  student_template=student_template, compress_question=compress_question,
                                  supervised=True)

    effective_batch_size = min(config['batch_size'], len(train_data))
    num_workers = min(8, len(train_data) // effective_batch_size) if len(train_data) > effective_batch_size else 0
    print(f"Using batch_size={effective_batch_size}, num_workers={num_workers}")

    # Backend-matched SFT collate: Mamba2 packs [1, T]; Mamba1 left-pads [B, I].
    sft_collate = lambda batch: (
        collate_fn_sft_mamba2(batch, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'])
        if compressor_type == 'mamba2'
        else collate_fn_sft_mamba1(batch, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'])
    )

    packing = config.get('packing', 'none')
    if compressor_type == 'mamba2' and packing == 'token_budget':
        train_loader = build_token_budget_loader(
            train_data, llm_tokenizer, compressor_tokenizer,
            tokens_per_batch=config['tokens_per_batch'],
            max_seq_len=config['max_seq_len'],
            max_seqs_per_batch=config.get('max_seqs_per_batch'),
            seed=SEED, shuffle=True, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: collate_fn_sft_mamba2(
                batch, llm_tokenizer, compressor_tokenizer,
                max_length=config['max_seq_len'], tokens_per_batch=config['tokens_per_batch'],
            ),
        )
    else:
        # Mamba1: standard distributed sampler (mirrors KD main); Mamba2 fixed-batch:
        # length-grouped to cut left-padding waste.
        if compressor_type == 'mamba':
            train_sampler = DistributedSampler(train_data, shuffle=True)
        else:
            train_sampler = DistributedLengthGroupedSampler(
                train_data, batch_size=effective_batch_size, shuffle=True)
        train_loader = DataLoader(
            train_data, batch_size=effective_batch_size, shuffle=False,
            sampler=train_sampler, collate_fn=sft_collate, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    val_sampler = DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(
        val_data, batch_size=effective_batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=sft_collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    modelDDP = DDP(model, device_ids=[config['local_rank']], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        modelDDP.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    decay_epochs = config.get('scheduler_epochs', config['num_epochs'])
    max_train_steps = decay_epochs * num_update_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=max_train_steps)

    print("\nStarting SFT training loop...")
    try:
        train(config, device, train_loader, val_loader, modelDDP, optimizer, scheduler)
    except KeyboardInterrupt:
        print(Fore.RED + "\n\nInterrupting training...\n")
        wandb.finish()
        raise


def _main_pretrain(config, device):
    SEED = config['seed']
    if config['global_rank'] == 0:
        init_services(config)
    else:
        sys.stdout = open(os.devnull, 'w')

    print("Initializing LMPretrainer...")
    model = LMPretrainer(
        mamba_model_name=config['mamba_model_name'],
        llm_model_name=config['llm_model_name'],
        device=device,
        stride=resolve_stride(config),
        compressor_type=config.get('compressor_type', 'mamba2'),
        prompt_mode=config.get('prompt_mode', 'embed'),
        memory_profile=config.get('memory_profile', False),
        compressor_gradient_checkpointing=config.get('compressor_gradient_checkpointing', True),
        llm_dtype=getattr(torch, config.get('llm_dtype', 'bfloat16')),
        attn_implementation=config.get('attn_implementation', 'sdpa'),
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'], padding_side='left')
    compressor_tokenizer = AutoTokenizer.from_pretrained(
        config.get('compressor_tokenizer_name', 'EleutherAI/gpt-neox-20b'),
        padding_side='left', clean_up_tokenization_spaces=False,
    )
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    if compressor_tokenizer.pad_token is None:
        compressor_tokenizer.pad_token = compressor_tokenizer.eos_token

    print("Loading FineWeb-EDU...")
    full = load_fineweb_dataset(config['data_dir'], seed=SEED, max_samples=config.get('max_samples'))
    val_n = config.get('val_samples', 2000)
    val_hf = full.select(range(val_n))
    train_hf = full.select(range(val_n, len(full)))
    min_doc = config.get('min_doc_tokens', 320)
    train_data = PretrainDataset(train_hf, min_doc_tokens=min_doc)
    val_data = PretrainDataset(val_hf, min_doc_tokens=min_doc)

    bs = config['batch_size']
    compressor_type = config.get('compressor_type', 'mamba2')
    max_ctx = config.get('max_ctx', 1024)
    max_pred = config.get('max_pred', 256)
    nw = min(8, os.cpu_count() or 1)
    pin = torch.cuda.is_available()

    pretrain_collate_fn = collate_fn_pretrain if compressor_type == 'mamba2' else collate_fn_pretrain_mamba1
    train_collate = partial(pretrain_collate_fn, llm_tokenizer=llm_tokenizer,
                            compressor_tokenizer=compressor_tokenizer,
                            max_ctx=max_ctx, max_pred=max_pred)

    packing = config.get('packing', 'none')
    if compressor_type == 'mamba2' and packing == 'token_budget':
        cap = max_ctx + max_pred
        train_loader = build_token_budget_loader(
            train_data, llm_tokenizer, compressor_tokenizer,
            tokens_per_batch=config['tokens_per_batch'], max_seq_len=cap,
            max_seqs_per_batch=config.get('max_seqs_per_batch'),
            seed=SEED, shuffle=True, num_workers=nw, pin_memory=pin,
            lengths=train_data.token_lengths, collate_fn=train_collate,
        )
        if config.get('global_rank', 0) == 0:
            print(f"Token-budget packing (mamba2): {config['tokens_per_batch']} tok/batch, cap {cap}, "
                  f"{len(train_loader)} batches/rank")
    else:
        # Mamba1 left-padded: cannot pack -> fixed batch_size + padding.
        train_sampler = DistributedLengthGroupedSampler(train_data, batch_size=bs, shuffle=True)
        train_loader = DataLoader(train_data, batch_size=bs, sampler=train_sampler,
                                  collate_fn=train_collate, num_workers=nw, pin_memory=pin,
                                  persistent_workers=nw > 0, prefetch_factor=2 if nw > 0 else None)

    val_sampler = DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=bs, sampler=val_sampler,
                            collate_fn=train_collate, num_workers=0, pin_memory=pin)

    modelDDP = DDP(model, device_ids=[config['local_rank']], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(modelDDP.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_steps'])

    print("\nStarting pretraining loop...")
    pretrain(config, device, train_loader, val_loader, modelDDP, optimizer, scheduler)


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
