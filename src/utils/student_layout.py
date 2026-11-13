"""Student input layout construction and wrapped-student geometry."""
import torch


def split_template_around_doc(tokenizer, system_prompt: str,
                             placeholder: str = "__DOC_SENTINEL__"):
    """Split chat template into prefix and suffix token IDs around document placeholder.

    Layout: [ prefix | document slot | suffix ].
    Splits on rendered string before tokenization to ensure clean boundary splice.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": placeholder},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = full.split(placeholder)
    if len(parts) != 2:
        raise ValueError(
            f"template placeholder {placeholder!r} appeared {len(parts) - 1} times in the "
            f"rendered chat template; expected exactly once."
        )
    prefix_str, suffix_str = parts
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    return prefix_ids, suffix_ids


def wrapped_geometry(prefix_mask: torch.Tensor, compressed_mask: torch.Tensor,
                     suffix_len: int, gen_mask: torch.Tensor):
    """Compute layout geometry for wrapped student with arbitrary prefix mask.

    Layout: [ prefix_region(prefix_mask) | soft prompts(C) | suffix(S) | gen(G) ]

    Returns:
        - full_padding_mask [B, P_total+C+S+G]
        - position_ids [B, P_total+C+S+G]: cumsum(mask)-1; pad slots filled with 1
        - anchor_offset (int): P_total + C + S; T_0 predicted from index anchor_offset-1
    """
    B = compressed_mask.shape[0]
    suffix_ones = compressed_mask.new_ones((B, suffix_len))
    full_padding_mask = torch.cat([prefix_mask, compressed_mask, suffix_ones, gen_mask], dim=1)

    position_ids = full_padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(full_padding_mask == 0, 1)

    anchor_offset = prefix_mask.shape[1] + compressed_mask.shape[1] + suffix_len
    return full_padding_mask, position_ids, anchor_offset


def wrapped_student_geometry(compressed_mask: torch.Tensor, gen_mask: torch.Tensor,
                            prefix_len: int, suffix_len: int):
    """Compute fixed-prefix geometry for template-wrapped student layout.

    Layout: [ prefix(P) | soft prompts(C) | suffix(S) | gen(G) ]
    Returns full_padding_mask [B, P+C+S+G], position_ids [B, P+C+S+G], and anchor_offset (P+C+S).
    """
    B = compressed_mask.shape[0]
    prefix_mask = compressed_mask.new_ones((B, prefix_len))
    return wrapped_geometry(prefix_mask, compressed_mask, suffix_len, gen_mask)


def as_id_list(token_id) -> list:
    """Normalize a HF token-id field (None, int, or sequence) to a list of ints."""
    if token_id is None:
        return []
    if isinstance(token_id, (list, tuple)):
        return [int(x) for x in token_id]
    return [int(token_id)]


def first_id(token_id):
    """Return first ID of a HF token-id field, or None if empty."""
    ids = as_id_list(token_id)
    return ids[0] if ids else None

