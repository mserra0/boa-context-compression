import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib
from colorama import Fore

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def decode_sequnce(logits: torch.Tensor, tokenizer) -> list[str]:
    """Decode model logits to text strings using greedy decoding."""
    probs = F.softmax(logits, dim=-1)
    token_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    sentences = []
    for seq in token_ids:
        sentence = tokenizer.decode(seq, skip_special_tokens=True)
        sentences.append(sentence)
    return sentences

def save2text(teacher_decoded: list[str], student_decoded: list[str], path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("Teacher Outputs:\n")
            f.write("="*40 + "\n")
            for i, sentence in enumerate(teacher_decoded):
                f.write(f"Sample {i+1}:\n{sentence}\n\n")
            
            f.write("\nStudent Outputs:\n")
            f.write("="*40 + "\n")
            for i, sentence in enumerate(student_decoded):
                f.write(f"Sample {i+1}:\n{sentence}\n\n") 
    except Exception as e:
        print(f"Artifacts - Error saving text outputs: {e}")

def plot_token_probabilities(teacher_logits, student_logits, tokenizer):
    def get_data(logits):
        probs = F.softmax(logits, dim=-1)
        token_ids = torch.argmax(probs, dim=-1)
        token_probs = probs[torch.arange(len(token_ids)), token_ids].detach().cpu().numpy()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        return tokens, token_probs

    t_tokens, t_probs = get_data(teacher_logits)
    s_tokens, s_probs = get_data(student_logits)

    plt.figure(figsize=(12, 6))
    plt.plot(t_probs, label='Teacher', marker='o')
    plt.plot(s_probs, label='Student', marker='x')
    
    plt.xticks(range(len(t_tokens)), t_tokens, rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title('Token Probabilities (Greedy Decode)')
    plt.legend()
    plt.tight_layout()
    
    return plt

def save_artifact(teacher_logits, student_logits, config, path):
    """Save logits, decoded text, and token probability plot as WandB artifact."""
    tokenizer = AutoTokenizer.from_pretrained(config.get("llm_model_name"))
    
    t_sample = teacher_logits[0:1]
    s_sample = student_logits[0:1]
    
    teacher_decoded = decode_sequnce(t_sample, tokenizer)
    student_decoded = decode_sequnce(s_sample, tokenizer)
    
    os.makedirs(path, exist_ok=True)
    logits_path = os.path.join(path, 'logits.pt')
    decode_path = os.path.join(path, 'decoded.txt')
    plot_path = os.path.join(path, 'token_probabilities.png')
    
    torch.save({
        'teacher_logits': t_sample.detach().cpu(),
        'student_logits': s_sample.detach().cpu()
    }, logits_path)
    
    save2text(teacher_decoded, student_decoded, decode_path)

    plot = plot_token_probabilities(teacher_logits[0], student_logits[0], tokenizer)
    plot.savefig(plot_path)
    plt.close()

    artifact = wandb.Artifact(name=f"sample_comparison_epoch_{os.path.basename(path).split('_')[-1]}", 
                              type='output', 
                              description='Comparison of teacher and student outputs')
    artifact.add_file(decode_path)
    artifact.add_file(plot_path)

    try:
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"Warning: failed to log artifact to wandb: {e}")

    print(f"{Fore.BLUE}Artifacts saved at {path} and logged to WandB.{Fore.RESET}")
    return artifact