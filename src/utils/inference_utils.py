import os
import torch
from transformers import AutoTokenizer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

from .. import PromptDestiller, load_config, CompressionDataset, collate_fn, collate_fn_mamba2

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "teacher": "blue",
    "student": "magenta"
})
console = Console(theme=custom_theme)


class _SingleSample:
    """Adapter so CompressionDataset can consume a single (context, question) pair
    via the same `dataset[idx]["context"|"question"]` protocol it expects from HF."""

    def __init__(self, context: str, question: str):
        self._item = {"context": context or "", "question": question}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._item

def _build_aligned_batch(context_text: str, question_text: str, llm_tokenizer, compressor_tokenizer,
                         max_seq_len: int, use_context: bool, compressor_type: str = "mamba",
                         student_template: str = "bare", compress_question: bool = True):
    """Tokenize a single inference sample using the EXACT training data path:
    per-tokenizer context truncation -> shared chat template -> left-padded collate.
    `student_template` must match the model so the compressor sees the same input shape
    (bare: full template; wrapped: content only). `compress_question` must also match the
    model (Design B-ctx): when False, the collated batch carries `question_ids` +
    `context_start` in 'compressor_inputs'. Returns a dict with 'llm_inputs' and
    'compressor_inputs'."""
    sample = _SingleSample(context_text, question_text)
    ds = CompressionDataset(
        sample,
        llm_tokenizer,
        compressor_tokenizer,
        max_length=max_seq_len,
        context=use_context,
        student_template=student_template,
        compress_question=compress_question,
    )
    fn = collate_fn_mamba2 if compressor_type == "mamba2" else collate_fn
    return fn([ds[0]], llm_tokenizer, compressor_tokenizer, max_length=max_seq_len)

def load_resources(config_path, checkpoint_path, device='cuda'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    compressor_type = config.get('compressor_type', 'mamba')

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task_model = progress.add_task("[cyan]Loading model architecture...", total=None)

        model = PromptDestiller(
            mamba_model_name=config['mamba_model_name'],
            llm_model_name=config['llm_model_name'],
            device=device,
            compression_ratio=config['compression_ratio'],
            temperature=config['temperature'],
            max_new_tokens=config['max_new_tokens'],
            compressor_type=compressor_type,
            student_template=config.get('student_template', 'bare'),
            compress_question=config.get('compress_question', True),
            prompt_mode=config.get('prompt_mode', 'embed'),
            anchor_init_token_ids=config.get('anchor_init_token_ids'),
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if state_dict and next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        is_compressor_only = not any(k.startswith('llm.') or k.startswith('compressor.') for k in state_dict.keys())
        if is_compressor_only:
            model.compressor.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        progress.update(task_model, completed=True, description="[green]Model loaded successfully!")

        task_tok = progress.add_task("[cyan]Loading tokenizers...", total=None)
        llm_tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'])
        compressor_tokenizer_name = config.get('compressor_tokenizer_name') or (
            'EleutherAI/gpt-neox-20b' if compressor_type == 'mamba2' else config['mamba_model_name']
        )
        # BPE tokenizer (GPT-NeoX): disable the destructive WordPiece cleanup
        compressor_tokenizer = AutoTokenizer.from_pretrained(
            compressor_tokenizer_name, clean_up_tokenization_spaces=False
        )

        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_tokenizer.padding_side = 'left'

        if compressor_tokenizer.pad_token is None:
            compressor_tokenizer.pad_token = compressor_tokenizer.eos_token
        # Mirror training: both tokenizers use left padding so the last real token
        # sits at index seq_len-1 — the anchor the model was trained against.
        compressor_tokenizer.padding_side = 'left'

        progress.update(task_tok, completed=True, description="[green]Tokenizers loaded!")

    return model, llm_tokenizer, compressor_tokenizer, config, compressor_type
