import torch
import os
import glob
import yaml
from colorama import Fore
import wandb
from .config import load_config, validate_config
import torch.nn as nn
import random
import numpy as np
from datetime import datetime, timedelta
from torch.distributed import init_process_group
import torch.distributed as dist
from typing import Any, Optional

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any,
                   epoch: int, train_loss: float, val_loss: float, config: dict,
                   best_val_loss: float = 100.0, total_steps: int = 0,
                   is_interrupted: bool = False, step_in_epoch: int = 0,
                   is_best: bool = True, quiet: bool = False) -> Optional[str]:
    
    # [DDP] Synchronize all processes before saving to prevent race conditions
    if dist.is_initialized() and not is_interrupted:
        dist.barrier()
        
    if config.get('global_rank', 0) == 0:
        
        save_dir = os.path.join(config['checkpoint_dir'], f"{config['wandb_name']}_{config['start_time']}")
        os.makedirs(save_dir, exist_ok=True)

        config_yaml_path = os.path.join(save_dir, "config.yaml")
        if not os.path.exists(config_yaml_path):
            with open(config_yaml_path, "w") as f:
                yaml.safe_dump(config, f)

        if wandb.run:
            config['wandb_id'] = wandb.run.id
            
        # Save all trained student params (everything EXCEPT the frozen LLM): the
        # compressor plus, in prompt_mode='kv', the kv_projector and anchor_embed —
        # which live on the destiller, not the compressor. Saving compressor-only
        # would omit the kv_projector/anchor weights -> garbage on reload.
        #
        # For ICAE, the LoRA-adapted encoder shares its base_lm object with the
        # frozen decoder (ICAEDestiller.llm, see model.py) — the injected lora_A/
        # lora_B adapter weights are the only trained part of that shared module,
        # but they still live under the 'llm.' prefix. A blanket 'llm.' exclusion
        # (correct for Mamba/PromptDestiller, where 'llm.' truly is 100% frozen)
        # would drop every ICAE checkpoint's LoRA weights, leaving only
        # compressor.mem_embeddings — an effectively untrained encoder on reload.
        core = model.module if hasattr(model, 'module') else model
        # Move to CPU explicitly: avoids hangs/errors
        model_state = {
            k: v.cpu() for k, v in core.state_dict().items()
            if not k.startswith('llm.') or '.lora_' in k
        }

        raw_opt = optimizer.state_dict()
        opt_state = {
            'state': {
                idx: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in param_state.items()}
                for idx, param_state in raw_opt['state'].items()
            },
            'param_groups': raw_opt['param_groups'],
        }

        checkpoint_dict = {
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': opt_state,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config,
            'best_val_loss': best_val_loss,
            'total_steps': total_steps,
            'interrupted': is_interrupted,
            'compressor_type': config.get('compressor_type', 'mamba'),
        }

        # 1. Base save (always save to a temporary file first)
        if is_interrupted:
            primary_name = "interrupted_model.pt"
        elif is_best:
            primary_name = "best_model.pt"
        else:
            primary_name = f"epoch_{epoch}_model.pt"
            
        primary_path = os.path.join(save_dir, primary_name)
        temp_path = primary_path + ".tmp"
        
        torch.save(checkpoint_dict, temp_path)
        os.replace(temp_path, primary_path)
        if not quiet:
            print(f"{Fore.GREEN}Model successfully saved to: {primary_path}{Fore.RESET}")

        is_save_interval = config.get('save_interval') and epoch % config.get('save_interval') == 0
        if primary_name == "best_model.pt" and is_save_interval:
            interval_path = os.path.join(save_dir, f"epoch_{epoch}_model.pt")
            torch.save(checkpoint_dict, interval_path)
            if not quiet:
                print(f"{Fore.GREEN}Interval checkpoint also saved to: {interval_path}{Fore.RESET}")
        
        return primary_path
    return None

def init_services(config: dict) -> Any:
    """Initializes logging and directories strictly on Rank 0."""
    if config.get('global_rank', 0) == 0:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(f"{config['checkpoint_dir']}/profiling", exist_ok=True)

        wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity'),
            config=config,
            name=config['wandb_name'],
            tags=config.get('wandb_tags', []),
            id=config.get('wandb_id'),
            resume="allow"
        )
    return wandb

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                   scheduler: Optional[Any] = None, device: Optional[torch.device] = None, is_resume: bool = True):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    _core = model.module if hasattr(model, 'module') else model
    _ckpt_ctype = checkpoint.get('compressor_type')
    _model_ctype = getattr(_core, 'compressor_type', None)
    if _ckpt_ctype and _model_ctype and _ckpt_ctype != _model_ctype:
        raise ValueError(
            f"Checkpoint compressor_type='{_ckpt_ctype}' does not match "
            f"model compressor_type='{_model_ctype}'. Use the correct model architecture."
        )

    state_dict = checkpoint['model_state_dict']

    # Strip 'module.' prefix if present. We always load into `core` (never DDP-wrapped)
    ckpt_keys = list(state_dict.keys())
    if ckpt_keys and ckpt_keys[0].startswith('module.'):
        print("Removing 'module.' prefix from checkpoint state_dict...")
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    # Detect checkpoint format for legacy support: compressor vs full.
    is_compressor_only = not any(k.startswith('llm.') or k.startswith('compressor.') for k in state_dict.keys())

    if is_compressor_only:
        _core.compressor.load_state_dict(state_dict)
    else:
        # Legacy full-model checkpoint. Keep strict=False because some saved runs
        # include the frozen LLM whose keys may differ after torch.compile wrapping.
        missing, unexpected = _core.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"{Fore.YELLOW}Legacy checkpoint had {len(unexpected)} unexpected keys (ignored).{Fore.RESET}")
        if missing:
            # Only the compressor must be fully loaded; missing LLM keys are fine.
            critical = [k for k in missing if k.startswith('compressor.')]
            if critical:
                raise RuntimeError(f"Compressor keys missing from checkpoint: {critical[:5]}...")
    
    if not is_resume:
        print(f"Loaded weights from {checkpoint_path} for a fresh start. Skipping optimizer and scheduler schemas.")
        return None

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    is_interrupted = checkpoint.get('interrupted', False)
    saved_epoch = checkpoint['epoch']
    
    if is_interrupted:
        print(f"Recovering interrupted run at epoch {saved_epoch}")
        start_epoch = saved_epoch
    else:
        print(f"Recovering completed run from epoch {saved_epoch}")
        start_epoch = saved_epoch + 1

    resume_info = {
        'start_epoch': start_epoch, 
        'step_in_epoch': checkpoint.get('step_in_epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf'))),
        'total_steps': checkpoint.get('total_steps', 0),
        'config': checkpoint.get('config', {})
    }
    
    print(f"Resuming from epoch {resume_info['start_epoch']}, step {resume_info['step_in_epoch']}, total steps: {resume_info['total_steps']}")
    return resume_info

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_main(config_path):
    errors = validate_config(config_path)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    
    config = load_config(config_path)
    config['start_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    set_deterministic(config['seed'])
    
    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    
    torch.cuda.set_device(local_rank)

    init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(minutes=30),
    )

    config['local_rank'] = local_rank
    config['global_rank'] = global_rank
    
    return config, torch.device(f"cuda:{local_rank}")

def get_latest_checkpoint(config):
    """
    Finds the latest checkpoint from the latest run directory.
    Checks for both 'best_model.pt' and 'interrupted_model.pt' and returns the newer one.
    """
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        print(f"{Fore.YELLOW}Checkpoint directory {checkpoint_dir} does not exist.{Fore.RESET}")
        return None
        
    # Get all subdirectories in checkpoint_dir
    subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) 
               if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    if not subdirs:
        print(f"{Fore.YELLOW}No previous runs found in {checkpoint_dir}{Fore.RESET}")
        return None
        
    # Sort by modification time, newest first to find the latest run
    latest_run_dir = max(subdirs, key=os.path.getmtime)
    
    print(f"Found latest run directory: {latest_run_dir}")
    
    interrupted_ckpt = os.path.join(latest_run_dir, "interrupted_model.pt")
    best_ckpt = os.path.join(latest_run_dir, "best_model.pt")
    epcoh_ckpt_pattern = os.path.join(latest_run_dir, "epoch_*_model.pt")
    
    candidates = []
    if os.path.exists(interrupted_ckpt):
        candidates.append(interrupted_ckpt)
    if os.path.exists(best_ckpt):
        candidates.append(best_ckpt)
    candidates.extend([f for f in glob.glob(epcoh_ckpt_pattern) if os.path.isfile(f)])
    
    if not candidates:
        print(f"{Fore.RED}No checkpoints found in {latest_run_dir}{Fore.RESET}")
        return None
        
    # Return the file with the most recent modification time
    latest_ckpt = max(candidates, key=os.path.getmtime)
    print(f"Selected latest checkpoint: {latest_ckpt}")
    return latest_ckpt
