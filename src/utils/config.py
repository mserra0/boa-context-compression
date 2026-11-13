import argparse
import math
import yaml
import sys
from pathlib import Path

REQUIRED_SECTIONS = {
    'model': ['mamba_model_name', 'llm_model_name', 'temperature', 'max_new_tokens'],
    'training': ['num_epochs', 'batch_size', 'learning_rate', 'weight_decay', 'log_interval',
                 'save_interval', 'gradient_clip_max_norm', 'gradient_accumulation_steps'],
    'data': ['dataset_name', 'data_dir', 'max_samples', 'seed', 'max_seq_len', 'use_context'],
    'paths': ['checkpoint_dir', 'output_dir'],
    'wandb': ['project', 'entity', 'name', 'tags'],
}

LM_REQUIRED_SECTIONS = {
    'model': ['mamba_model_name', 'llm_model_name', 'objective'],
    'training': ['max_steps', 'warmup_steps', 'batch_size', 'learning_rate', 'weight_decay',
                 'log_interval', 'save_interval', 'eval_interval', 'gradient_clip_max_norm',
                 'gradient_accumulation_steps'],
    'data': ['data_dir', 'compressor_tokenizer_name', 'seed'],
    'paths': ['checkpoint_dir', 'output_dir'],
    'wandb': ['project', 'entity', 'name', 'tags'],
}


def resolve_compression_ratio(model: dict) -> float:
    """Resolve the compression ratio from an explicit 'stride' or the legacy 'compression_ratio'.

    `stride` S is preferred (ratio = 1/S, the same convention as on `main`). If both keys are given
    they must describe the same stride (`int(1 / compression_ratio) == stride`).
    """
    stride = model.get('stride')
    ratio = model.get('compression_ratio')
    if stride is None:
        if ratio is None:
            raise KeyError("config requires 'stride' (or legacy 'compression_ratio')")
        return ratio
    stride = int(stride)
    if ratio is not None:
        if int(1 / ratio) != stride:
            raise ValueError(f"'stride' ({stride}) and 'compression_ratio' ({ratio}) disagree")
        return ratio
    derived = 1.0 / stride
    while int(1.0 / derived) < stride:  # float rounding (e.g. stride 93): keep int(1 / ratio) == stride
        derived = math.nextafter(derived, 0.0)
    return derived


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with automatic flattening"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if section == 'wandb':
                    flat_config[f'wandb_{key}'] = value
                else:
                    flat_config[key] = value
        else:
            flat_config[section] = values

    if 'stride' in flat_config or 'compression_ratio' in flat_config:
        flat_config['compression_ratio'] = resolve_compression_ratio(flat_config)

    return flat_config


def validate_config(config_path: str):
    if not Path(config_path).exists():
        return [f"File not found: {config_path}"]

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"Invalid YAML: {e}"]

    errors = []
    is_lm = config.get('model', {}).get('objective') == 'lm'
    required = LM_REQUIRED_SECTIONS if is_lm else REQUIRED_SECTIONS
    for section, keys in required.items():
        if section not in config:
            errors.append(f"Missing section: '{section}'")
            continue

        for key in keys:
            if key not in config[section]:
                errors.append(f"Missing key: '{key}' in section '{section}'")

    model = config.get('model', {})
    if 'stride' not in model and 'compression_ratio' not in model:
        errors.append("Missing key: 'stride' (or legacy 'compression_ratio') in section 'model'")
    else:
        try:
            resolve_compression_ratio(model)
        except ValueError as e:
            errors.append(str(e))

    # Optional 'dataloading' block. token_budget means packed [1,T] for Mamba2 and
    # padded B*maxX budgeting for Mamba1. Absent entirely -> defaults to packing: none
    # at load_config time, no error here.
    dl = config.get('dataloading')
    if dl is not None:
        packing = dl.get('packing', 'none')
        if packing not in ('none', 'token_budget'):
            errors.append(
                f"dataloading.packing must be 'none' or 'token_budget', got '{packing}'"
            )
        elif packing == 'token_budget' and 'tokens_per_batch' not in dl:
            errors.append(
                "dataloading.packing=token_budget requires dataloading.tokens_per_batch"
            )

    return errors


def print_summary(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfiguration: {config_path}")
    print("-" * 60)
    print(f"Model: {config['model']['llm_model_name']}")
    print(f"Compressor: {config['model']['mamba_model_name']}")
    print(f"Compression: {resolve_compression_ratio(config['model'])}")
    print(f"Epochs: {config['training']['num_epochs']}, Batch: {config['training']['batch_size']}")
    print(f"Dataset: {config['data'].get('data_dir', 'Unknown')} (max: {config['data'].get('max_samples') or 'all'})")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Validate config file')
    parser.add_argument('config', type=str, help='Path to config YAML')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    args = parser.parse_args()
    
    errors = validate_config(args.config)
    
    if errors:
        print(f"Invalid config: {args.config}")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print(f"Valid config: {args.config}")
    if args.summary:
        print_summary(args.config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
