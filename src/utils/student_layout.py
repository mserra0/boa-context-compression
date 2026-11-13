"""Student-input construction: chat-template splitting and wrapped-student geometry.

These helpers build the layout the frozen LLM sees around the compressed soft prompts
(Designs B-qc and B-ctx). See docs/architecture/student-templates.md on the `main` branch.
"""
import torch


def split_template_around_doc(tokenizer, system_prompt: str,
                             placeholder: str = "__DOC_SENTINEL__"):
    """Split the Qwen chat template into the token ids that sit BEFORE and AFTER the
    document slot, by rendering the template with a sentinel document and splitting on
    it. Never hardcodes special tokens — derives them from `apply_chat_template`.

        prefix = <|im_start|>system ... <|im_start|>user\n      (opening scaffold)
        [ document slot — replaced by soft prompts in the model ]
        suffix = <|im_end|>\n<|im_start|>assistant\n             (closing + gen cue)

    The split happens on the rendered STRING before tokenization, so the placeholder
    never reaches the tokenizer and the prefix/suffix boundaries are exactly where the
    (non-tokenized) soft prompts will be spliced in.
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
    """Geometry for a wrapped student with an arbitrary (possibly padded) literal prefix:

        [ prefix_region(prefix_mask) | soft prompts(C) | suffix(S) | gen(G) ]

    `prefix_mask` [B, P_total] is 1 over real tokens, 0 over left-pad slots inside the
    literal prefix region (tpl_prefix + optional literal question + sep). Only the soft
    block and the question sub-region carry padding; suffix and gen are always valid.

      - full_padding_mask [B, P_total+C+S+G]
      - position_ids: HF cumsum(mask)-1; pad slots -> 1 (never the anchor)
      - anchor = P_total + C + S; T_0 is predicted from index anchor-1 (last suffix token).
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
    """Pure geometry for the template-wrapped student layout (Design B-qc):

        [ prefix(P) | soft prompts(C) | suffix(S) | gen(G) ]

    The prefix/suffix are real Qwen chat-template token embeddings; they are always
    valid (never padded). Only the soft-prompt block carries left-padding, encoded in
    `compressed_mask`. Returns the geometry the wrapped forward needs:

      - full_padding_mask [B, P+C+S+G]: ones over prefix, then compressed_mask, then
        ones over suffix and gen.
      - position_ids [B, P+C+S+G]: HF cumsum(mask)-1 recipe so left-pad soft slots
        consume no position; pad slots are filled with 1 (never the anchor).
      - anchor_offset (int) = P + C + S: index just past the last suffix token. The
        student logits slice is [:, anchor_offset-1 : -1], i.e. T_0 is predicted from
        the last suffix token (the `assistant\n` cue) — the same position the teacher
        predicts from.

    Delegates to `wrapped_geometry` with a FIXED, all-valid prefix of width `prefix_len`
    (kept as a thin wrapper so existing callers are unchanged).
    """
    B = compressed_mask.shape[0]
    prefix_mask = compressed_mask.new_ones((B, prefix_len))
    return wrapped_geometry(prefix_mask, compressed_mask, suffix_len, gen_mask)


def build_question_block(question_ids_list, dtype, embed, hidden_size: int, device):
    """Embed per-sample literal question ids and LEFT-pad to the batch max.

    `embed` is the frozen LLM's input-embedding module (e.g. `self.llm.get_input_embeddings()`).
    `hidden_size` can't be derived from `embed` generically (it may not always be a plain
    `nn.Embedding`), so it's passed explicitly.

    Returns (q_block [B, Qmax, H], q_mask [B, Qmax]) with q_mask 1 over real tokens.
    """
    B = len(question_ids_list)
    lens = [len(q) for q in question_ids_list]
    Qmax = max(1, max(lens))
    q_block = torch.zeros((B, Qmax, hidden_size), device=device, dtype=dtype)
    q_mask = torch.zeros((B, Qmax), device=device, dtype=torch.long)
    for i, q in enumerate(question_ids_list):
        n = len(q)
        if n == 0:
            continue
        ids = torch.tensor(q, device=device)
        with torch.no_grad():
            q_block[i, Qmax - n:] = embed(ids).to(dtype)
        q_mask[i, Qmax - n:] = 1
    return q_block, q_mask


def as_id_list(token_id) -> list:
    """Normalize a HF token-id field (None | int | list/tuple) to a list of ints."""
    if token_id is None:
        return []
    if isinstance(token_id, (list, tuple)):
        return [int(x) for x in token_id]
    return [int(token_id)]


def first_id(token_id):
    """First id of a HF token-id field (handles scalar or list); None if empty."""
    ids = as_id_list(token_id)
    return ids[0] if ids else None
