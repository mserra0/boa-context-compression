import os
import sys
import argparse
import time
import hashlib
import torch
import wandb
import math
import numpy as np
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist
from colorama import init as colorama_init, Fore
from src import (CompressionDataset,
                 collate_fn,
                 collate_fn_mamba2,
                 load_mrqa_dataset,
                 PromptDestiller,
                 ICAEDestiller,
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
                 fast_indexed_column_numpy,
                 build_token_budget_loader,
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

    SEED = config['seed']
    if config['global_rank'] == 0: 
        init_services(config)
    else: 
        sys.stdout = open(os.devnull, 'w') 
    
    print("Initializing model...")

    compressor_type = config.get('compressor_type', 'mamba')

    if compressor_type == 'icae':
        # encoder_model_name: the LLM to adapt with LoRA as the ICAE encoder.
        # Defaults to llm_model_name so encoder and decoder share the same architecture.
        encoder_model_name = config.get('encoder_model_name', config['llm_model_name'])
        # ICAE compresses into a FIXED memory-slot count sized to the compressed span's
        # max length (see CompressionDataset docstring, src/dataset.py) rather than a
        # hand-picked constant.
        n_mem = round(config.get('max_ctx', 1024) * config['compression_ratio'])
        print(f"[INFO] compression_ratio={config['compression_ratio']} -> computed n_mem={n_mem}")
        icae_objective = config.get('icae_objective', 'ae')
        destiller = ICAEDestiller(
            encoder_model_name=encoder_model_name,
            llm_model_name=config['llm_model_name'],
            device=device,
            n_mem=n_mem,
            compression_ratio=config['compression_ratio'],
            temperature=config['temperature'],
            max_new_tokens=config['max_new_tokens'],
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.05),
            objective=icae_objective,
            loss_type=config.get('loss_type', 'kl'),
            confidence_mode=config.get('confidence_mode', 'max_prob'),
            pretrain_ae_weight=config.get('pretrain_ae_weight', 0.5),
            compress_question=config.get('compress_question', True),
        )
    else:
        icae_objective = None
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
            prompt_mode=config.get('prompt_mode', 'embed'),
            anchor_init_token_ids=_resolve_anchor_token_ids(config),
        )

    print("Loading tokenizers...")
    llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'], padding_side='left')

    # ICAE encoder uses the same tokenizer as the LLM.
    # Mamba2 weights (state-spaces/mamba2-*) do not ship a tokenizer; fall back to GPT-NeoX.
    if compressor_type == 'icae':
        compressor_tokenizer_name = config.get('encoder_model_name', config['llm_model_name'])
    else:
        compressor_tokenizer_name = config.get('compressor_tokenizer_name') or (
            'EleutherAI/gpt-neox-20b' if compressor_type == 'mamba2' else config['mamba_model_name']
        )

    # clean_up_tokenization_spaces=False: the compressor tokenizer is BPE (GPT-NeoX)
    compressor_tokenizer = AutoTokenizer.from_pretrained(
        compressor_tokenizer_name, padding_side='left', clean_up_tokenization_spaces=False
    )

    print("Loading train and val datasets...")
    data_dir = config.get('data_dir', '/path/to/data/MRQA')
    dataset_format = config.get('dataset_format', 'mrqa')

    if dataset_format == 'fineweb':
        # Generic-text continuation pretraining (ICAE 'lm'/'ae_lm' over FineWeb-EDU,
        # not MRQA): single pile of documents, so train/val is a held-out slice
        # rather than separate train/dev directories.
        full_dataset = load_fineweb_dataset(data_dir, seed=SEED, max_samples=config.get('max_samples'))
        min_doc_tokens = config.get('min_doc_tokens', 320)
        distributed = dist.is_available() and dist.is_initialized()
        if 'token_count' in full_dataset.column_names:
            # Vectorized column read + in-memory filter instead of Dataset.filter()'s
            # row-by-row Python callback over the whole corpus (PretrainDataset
            # in src/dataset.py reads its columns the same way).
            # Rank-gated like load_fineweb_dataset's own load_dataset() call just above:
            # .select() below still caches its indices mapping to the shared HF cache,
            # so every rank racing to build it concurrently would hit the same PermissionError
            # as load_dataset() there — rank 0 builds it, everyone else waits then
            # reuses the now-warm cache.
            if distributed and dist.get_rank() != 0:
                dist.barrier()
            t0 = time.time()
            print(f"[fineweb] Filtering to token_count >= {min_doc_tokens}...")
            # The filter result is fully determined by (data_dir, max_samples, seed,
            # min_doc_tokens, row count) -- cache the kept-row indices on disk so
            # repeat runs with the same knobs (e.g. iterating on other config values
            # during debugging) skip the counts materialization below entirely.
            filter_cache_dir = os.path.join(
                os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
                'fineweb_filter_cache',
            )
            os.makedirs(filter_cache_dir, exist_ok=True)
            cache_key = hashlib.sha256(
                f"{data_dir}|{config.get('max_samples')}|{SEED}|{min_doc_tokens}|{len(full_dataset)}".encode()
            ).hexdigest()[:16]
            filter_cache_path = os.path.join(filter_cache_dir, f"keep_indices_{cache_key}.npy")
            if os.path.exists(filter_cache_path):
                keep_indices = np.load(filter_cache_path)
                print(f"[fineweb] Loaded cached filter indices from {filter_cache_path}")
            else:
                # fast_indexed_column_numpy avoids both the per-row boxing of
                # full_dataset['token_count'] and the stale-indices hazard of reading
                # full_dataset.data.column(...) directly (that reads the underlying
                # Arrow table and ignores any .select() indices mapping). With
                # max_samples null no .select() has been applied yet, so both reads
                # agree; this stays correct if max_samples gates a .shuffle().select()
                # upstream in load_fineweb_dataset.
                counts = fast_indexed_column_numpy(full_dataset, 'token_count')
                keep_indices = np.nonzero(counts >= min_doc_tokens)[0]
                np.save(filter_cache_path, keep_indices)
            full_dataset = full_dataset.select(keep_indices.tolist())
            print(f"[fineweb] Kept {len(full_dataset)} rows in {time.time() - t0:.1f}s")
            if distributed and dist.get_rank() == 0:
                dist.barrier()
        full_dataset = full_dataset.rename_column('text', 'context')

        # Shuffle before slicing off val_samples, so val is a genuine random holdout
        # instead of whatever happens to sort first in the parquet shards (the slices
        # below are positional). Rank-gated for the same reason as the filter above:
        # .shuffle() writes a permutation index to the shared HF cache, and every rank
        # computes the identical seeded permutation, so letting them all race to build
        # it concurrently would hit the same PermissionError.
        if distributed and dist.get_rank() != 0:
            dist.barrier()
        t0 = time.time()
        print("[fineweb] Shuffling full dataset...")
        full_dataset = full_dataset.shuffle(seed=SEED)
        print(f"[fineweb] Shuffle done in {time.time() - t0:.1f}s")
        if distributed and dist.get_rank() == 0:
            dist.barrier()

        # Same shared-HF-cache race as the filter/shuffle above: these .select() calls
        # write indices-mapping cache files too, so gate them identically.
        val_samples = config.get('val_samples', 2000)
        if distributed and dist.get_rank() != 0:
            dist.barrier()
        t0 = time.time()
        val_dataset = full_dataset.select(range(min(val_samples, len(full_dataset))))
        train_dataset = full_dataset.select(range(val_samples, len(full_dataset)))
        print(f"[fineweb] Train/val split done in {time.time() - t0:.1f}s")
        if distributed and dist.get_rank() == 0:
            dist.barrier()
    else:
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

    include_answer = (icae_objective == 'gt')
    include_continuation = (icae_objective in ('lm', 'ae_lm'))
    continuation_split_range = tuple(config.get('continuation_split_range', (0.5, 0.9)))
    max_ctx = config.get('max_ctx', 1024)
    max_pred = config.get('max_pred', 256)
    student_template = config.get('student_template', 'bare')
    compress_question = config.get('compress_question', True)
    train_data = CompressionDataset(
        train_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'],
        context=config['use_context'], include_answer=include_answer,
        include_continuation=include_continuation, continuation_split_range=continuation_split_range,
        max_ctx=max_ctx, max_pred=max_pred,
        student_template=student_template, compress_question=compress_question,
    )
    val_data = CompressionDataset(
        val_dataset, llm_tokenizer, compressor_tokenizer, max_length=config['max_seq_len'],
        context=config['use_context'], include_answer=include_answer,
        include_continuation=include_continuation, continuation_split_range=continuation_split_range,
        max_ctx=max_ctx, max_pred=max_pred,
        student_template=student_template, compress_question=compress_question,
    )
    
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
    # include_continuation items are already capped per-field in __getitem__ (compressor
    # input <= max_ctx, continuation <= max_pred); pass a collate max_length that's a
    # no-op clip for both instead of config['max_seq_len'], which would otherwise
    # re-truncate the compressor input back down whenever max_seq_len < max_ctx.
    collate_max_length = max(max_ctx, max_pred) if include_continuation else config['max_seq_len']

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
                batch, llm_tokenizer, compressor_tokenizer, max_length=collate_max_length
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
            batch, llm_tokenizer, compressor_tokenizer, max_length=collate_max_length
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

    # ICAE continuation-pretraining (lm/ae_lm) uses the step-based driver, matching
    # Mamba's fineweb pretrain recipe (src/pretrain.py) -> config sets max_steps
    # instead of num_epochs. Every other objective (ae/gt/distill, MRQA-style
    # fine-tuning) keeps the epoch-based driver (src/train.py).
    if include_continuation and 'max_steps' in config:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['max_steps'],
        )
        print("\nStarting pretraining loop (step-based)...")
        pretrain(config, device, train_loader, val_loader, destillerDDP, optimizer, scheduler)
    else:
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
        compression_ratio=config['compression_ratio'],
        compressor_type=config.get('compressor_type', 'mamba2'),
        prompt_mode=config.get('prompt_mode', 'embed'),
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
    # Same shared-HF-cache race as load_fineweb_dataset's internal .select() calls:
    # gate these too so ranks don't race to build the same indices-mapping cache file.
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_rank() != 0:
        dist.barrier()
    val_hf = full.select(range(val_n))
    train_hf = full.select(range(val_n, len(full)))
    if distributed and dist.get_rank() == 0:
        dist.barrier()
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
