import torch
import torch.nn.functional as F


def valid_positions_mask(generated_ids: torch.Tensor, eos_token_ids) -> torch.Tensor:
    """Boolean [B, G] mask keeping tokens up to and including the first EOS token.

    EOS-aware mask: Qwen's pad ID (<|endoftext|>) matches an EOS ID, so `!= pad`
    would drop valid stop tokens. The first EOS provides training signal; subsequent
    tokens are trailing padding.
    """
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos = torch.tensor(list(eos_token_ids), device=generated_ids.device)
    is_eos = (generated_ids.unsqueeze(-1) == eos).any(-1)   # [B, G]
    eos_seen = is_eos.cumsum(dim=1)                          # [B, G], inclusive
    return (eos_seen == 0) | (is_eos & (eos_seen == 1))


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float,
    eos_token_ids,
) -> torch.Tensor:
    """Token-uniform KL distillation loss scaled by T^2 over valid positions."""
    valid = valid_positions_mask(generated_ids, eos_token_ids).to(student_logits.dtype)  # [B, G]
    per_tok_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='none',
    ).sum(-1)  # [B, G]
    return (per_tok_kl * valid).sum() / valid.sum().clamp_min(1.0) * (temperature ** 2)


def confidence_weighted_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float,
    eos_token_ids,
    mode: str = "max_prob",
) -> torch.Tensor:
    """Confidence-weighted KL distillation loss weighting high-confidence teacher tokens."""
    valid = valid_positions_mask(generated_ids, eos_token_ids).to(student_logits.dtype)  # [B, G]

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # [B, G, V]

    if mode == "max_prob":
        weights = teacher_probs.max(dim=-1).values  # [B, G]
    elif mode == "neg_entropy":
        entropy = -(teacher_probs * (teacher_probs + 1e-9).log()).sum(-1)  # [B, G]
        weights = 1.0 / (entropy + 1e-6)
    else:
        raise ValueError(f"Unknown confidence mode: {mode!r}. Choose 'max_prob' or 'neg_entropy'.")

    weights = weights * valid
    denom = weights.sum().clamp_min(1e-9)

    per_tok_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        teacher_probs,
        reduction='none',
    ).sum(-1)  # [B, G]

    return (per_tok_kl * weights).sum() / denom * (temperature ** 2)


def lm_ce_loss(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Token-mean cross-entropy over unpadded target positions."""
    B, G, V = student_logits.shape
    mask = target_mask.to(student_logits.dtype)  # [B, G]
    per_tok = F.cross_entropy(
        student_logits.reshape(-1, V),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(B, G)  # [B, G]
    return (per_tok * mask).sum() / mask.sum().clamp_min(1.0)

