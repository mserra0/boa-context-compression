import os
import sys
import time
import json
import random
import torch
import argparse
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.align import Align

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.inference_utils import console, _build_aligned_batch, load_resources


def chat_loop(config_path, checkpoint_path, device='cuda', context_file=None):
    fixed_context = ""
    if context_file:
        with open(context_file, 'r', encoding='utf-8') as f:
            fixed_context = f.read().strip()

    try:
        model, llm_tokenizer, compressor_tokenizer, config, compressor_type = load_resources(config_path, checkpoint_path, device)
    except Exception as e:
        console.print(f"[error]Failed to load resources: {e}[/error]")
        return

    val_samples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "val_samples.json")
    val_samples = []
    if os.path.exists(val_samples_path):
        try:
            with open(val_samples_path, 'r') as f:
                val_samples = json.load(f)
        except Exception as e:
            console.print(f"[warning]Failed to load validation samples from {val_samples_path}: {e}[/warning]")

    train_samples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "train_samples.json")
    train_samples = []
    if os.path.exists(train_samples_path):
        try:
            with open(train_samples_path, 'r') as f:
                train_samples = json.load(f)
        except Exception as e:
            console.print(f"[warning]Failed to load training samples from {train_samples_path}: {e}[/warning]")

    console.print(Panel(
        Align.left(
            "[bold green]Interactive Inference Session[/bold green]\n"
            "Compare [blue]Teacher[/blue] (Baseline) vs [magenta]Student[/magenta] (Compressed)\n"
            "Type [yellow]'/example'[/yellow] to load a random validation sample.\n"
            "Type [yellow]'/example train'[/yellow] to load a random training sample.\n"
            "Type [yellow]'exit'[/yellow] or [yellow]'quit'[/yellow] to end."
        ),
        title="Boa",
        border_style="green",
        expand=True
    ))

    while True:
        try:
            console.print("\n[bold green]User > [/bold green]", end="")
            question = input()

            if not question.strip():
                continue

            if question.lower() in ['exit', 'quit']:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            context_text = fixed_context
            question_text = question

            if question.lower().startswith('/example'):
                is_train = question.lower().strip() == '/example train'

                if is_train:
                    if not train_samples:
                        console.print(f"[red]No train samples loaded (could not find {train_samples_path}).[/red]")
                        continue
                    sample = random.choice(train_samples)
                    src_name = "Train"
                else:
                    if not val_samples:
                        console.print(f"[red]No validation samples loaded (could not find {val_samples_path}).[/red]")
                        continue
                    sample = random.choice(val_samples)
                    src_name = "Validation"

                q_text = sample.get("question", "")
                c_text = sample.get("context", "")
                a_text = sample.get("answer", "")

                use_context_cfg = config.get("use_context", False)
                question_text = q_text
                context_text = c_text if use_context_cfg else ""

                console.print(f"[blue]Loaded {src_name} Example Question:[/blue] {q_text}")
                if use_context_cfg:
                    console.print(f"[cyan]Loaded {src_name} Example Context (truncated):[/cyan] {c_text[:300]}...")
                console.print(f"[green]Expected Answer:[/green] {a_text}")

            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                max_seq_len = config.get('max_seq_len', 256)

                # ---------------------------------------------------------
                # Build inputs through the EXACT training data path:
                #   per-tokenizer context truncation -> shared chat template
                #   -> left-padded collate. This guarantees teacher & student
                #   see the same distribution of inputs they saw during KD.
                # ---------------------------------------------------------
                compress_question = getattr(model, "compress_question", True)
                batch = _build_aligned_batch(
                    context_text=context_text,
                    question_text=question_text,
                    llm_tokenizer=llm_tokenizer,
                    compressor_tokenizer=compressor_tokenizer,
                    max_seq_len=max_seq_len,
                    use_context=bool(context_text),
                    compressor_type=compressor_type,
                    student_template=getattr(model, "student_template", "bare"),
                    compress_question=compress_question,
                )
                llm_inputs = {k: v.to(device) for k, v in batch['llm_inputs'].items()}
                comp_inputs = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch['compressor_inputs'].items()
                }

                pid = model.llm.config.pad_token_id
                pad_token_id = pid if pid is not None else model.llm.config.eos_token_id

                use_cuda = torch.device(device).type == 'cuda'

                ### Teacher Path ###
                with torch.no_grad():
                    if use_cuda: torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    teacher_outputs = model.llm.generate(
                        **llm_inputs,
                        max_new_tokens=config['max_new_tokens'],
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                    if use_cuda: torch.cuda.synchronize()
                    t_teacher = time.perf_counter() - t0

                input_len = llm_inputs['input_ids'].shape[1]
                teacher_new_tokens = int(teacher_outputs.shape[1] - input_len)
                teacher_answer = llm_tokenizer.decode(
                    teacher_outputs[0, input_len:], skip_special_tokens=True
                )

                ### Student Path ###
                with torch.no_grad():
                    if use_cuda: torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    if compressor_type == 'mamba2':
                        cstart = comp_inputs.get("context_start")
                        soft_prompts, compressed_mask = model.compressor.compress_single(
                            comp_inputs['input_ids'], context_start=cstart
                        )
                        qids = comp_inputs.get("question_ids") if compress_question is False else None
                        # eos defaults to the model's full eos set (Qwen2.5 stops on both <|im_end|> and <|endoftext|>).
                        student_outputs_ids = model.generate_compressed(
                            soft_prompts,
                            compressed_mask=compressed_mask,
                            max_new_tokens=config['max_new_tokens'],
                            question_ids=qids,
                        )
                        if use_cuda: torch.cuda.synchronize()
                        t_student = time.perf_counter() - t0

                        student_input_tokens = int(soft_prompts.shape[1])
                        student_new_tokens = int(student_outputs_ids.shape[1])
                        student_answer = llm_tokenizer.decode(student_outputs_ids[0], skip_special_tokens=True)

                    elif compressor_type == 'icae':
                        # Fixed n_mem memory slots (all-ones mask) — no pad-embed
                        # replacement needed, unlike mamba's variable-length slots.
                        icae_compressor_inputs = {
                            'input_ids': comp_inputs['input_ids'],
                            'attention_mask': comp_inputs['attention_mask'],
                        }
                        soft_prompts, compressed_mask = model.compressor(icae_compressor_inputs)
                        llm_dtype = next(model.llm.parameters()).dtype
                        soft_prompts = soft_prompts.to(llm_dtype)
                        qids = comp_inputs.get("question_ids") if compress_question is False else None
                        student_outputs = model.generate_compressed(
                            soft_prompts,
                            compressed_mask=compressed_mask,
                            max_new_tokens=config['max_new_tokens'],
                            eos_token_id=llm_tokenizer.eos_token_id,
                            question_ids=qids,
                        )
                        if use_cuda: torch.cuda.synchronize()
                        t_student = time.perf_counter() - t0

                        student_input_tokens = int(soft_prompts.shape[1])
                        student_new_tokens = int(student_outputs.shape[1])
                        student_answer = llm_tokenizer.decode(student_outputs[0], skip_special_tokens=True)

                    elif compressor_type == 'mamba':
                        mamba_compressor_inputs = {
                            'input_ids': comp_inputs['input_ids'],
                            'attention_mask': comp_inputs['attention_mask'],
                        }
                        if 'context_start' in comp_inputs:
                            mamba_compressor_inputs['context_start'] = comp_inputs['context_start']
                        soft_prompts, compressed_mask = model.compressor(mamba_compressor_inputs)
                        pad_embed = model.llm.get_input_embeddings()(
                            torch.tensor(pad_token_id, device=device)
                        )
                        # Dtype alignment: compressor (Mamba + projector) may run in a
                        # different dtype than the LLM (e.g. fp32 vs bf16), since each
                        # is loaded from its own checkpoint's native dtype.
                        soft_prompts = soft_prompts.to(pad_embed.dtype)
                        soft_prompts, compressed_mask = model._replace_pad_soft_prompts(
                            soft_prompts, compressed_mask, pad_embed
                        )
                        if getattr(model, "student_template", "bare") == "wrapped":
                            # Wrapped student: wrap soft prompts in the template via
                            # generate_compressed (T_0 from the last suffix token) to match
                            # the trained geometry instead of generating from bare prompts.
                            qids = comp_inputs.get("question_ids") if compress_question is False else None
                            student_outputs = model.generate_compressed(
                                soft_prompts,
                                compressed_mask=compressed_mask,
                                max_new_tokens=config['max_new_tokens'],
                                question_ids=qids,
                            )
                        else:
                            student_outputs = model.llm.generate(
                                inputs_embeds=soft_prompts,
                                attention_mask=compressed_mask,
                                max_new_tokens=config['max_new_tokens'],
                                do_sample=False,
                                pad_token_id=pad_token_id,
                            )
                        if use_cuda: torch.cuda.synchronize()
                        t_student = time.perf_counter() - t0

                        student_input_tokens = int(soft_prompts.shape[1])
                        student_new_tokens = int(student_outputs.shape[1])
                        student_answer = llm_tokenizer.decode(student_outputs[0], skip_special_tokens=True)
                    else:
                        raise ValueError(f"Unsupported compressor type: {compressor_type}")

            console.print(Panel(
                Markdown(teacher_answer),
                title="[bold blue]Teacher (Baseline)[/bold blue]",
                border_style="blue",
                expand=False
            ))

            console.print(Panel(
                Markdown(student_answer),
                title="[bold magenta]Student (Compressed)[/bold magenta]",
                border_style="magenta",
                expand=False
            ))

            # Performance table: input/output tokens + latency, then compression/speedup.
            stats = Table(show_header=True, header_style="bold cyan", title="Performance")
            stats.add_column("Path", style="bold")
            stats.add_column("Input tokens", justify="right")
            stats.add_column("Output tokens", justify="right")
            stats.add_column("Latency (ms)", justify="right")
            stats.add_row("Teacher", str(input_len), str(teacher_new_tokens), f"{t_teacher * 1000:.1f}")
            stats.add_row("Student", str(student_input_tokens), str(student_new_tokens), f"{t_student * 1000:.1f}")
            console.print(stats)

            comp_ratio = input_len / student_input_tokens if student_input_tokens > 0 else float('inf')
            lat_ratio = t_teacher / t_student if t_student > 0 else float('inf')
            comp_label = "compression" if comp_ratio >= 1 else "expansion"
            comp_value = comp_ratio if comp_ratio >= 1 else 1.0 / comp_ratio
            lat_label = "faster" if lat_ratio >= 1 else "slower"
            lat_value = lat_ratio if lat_ratio >= 1 else 1.0 / lat_ratio
            lat_style = "green" if lat_ratio >= 1 else "red"
            console.print(
                f"[bold green]Input: {comp_value:.2f}× {comp_label}[/bold green]  ·  "
                f"[bold {lat_style}]Latency: {lat_value:.2f}× {lat_label}[/bold {lat_style}]"
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[error]Error during generation: {e}[/error]")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interactive inference comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context-file", type=str, default=None,
                        help="Text file used as the context for every question (the teacher sees it in full, the student sees it compressed)")

    args = parser.parse_args()

    chat_loop(args.config, args.checkpoint, args.device, args.context_file)