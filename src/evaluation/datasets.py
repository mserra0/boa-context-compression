"""MRQA dev set loader keeping all gold answers for official evaluation."""

import gzip
import json
import os

# Dataset key to local gz filename mappings
MRQA_INDOMAIN = {
    "squad": "SQuAD.jsonl.gz",
    "newsqa": "NewsQA.jsonl.gz",
    "triviaqa": "TriviaQA.jsonl.gz",
    "hotpotqa": "HotpotQA.jsonl.gz",
    "searchqa": "SearchQA.jsonl.gz",
    "naturalquestions": "NaturalQuestions.jsonl.gz",
}

MRQA_OOD = {
    "bioasq": "BioASQ.jsonl.gz",
    "drop": "DROP.jsonl.gz",
    "duorc": "DuoRC.jsonl.gz",
    "race": "RACE.jsonl.gz",
    "relationextraction": "RelationExtraction.jsonl.gz",
    "textbookqa": "TextbookQA.jsonl.gz",
}

INDOMAIN_SUBDIR = "dev"
OOD_SUBDIR = "dev_ood"


def _load_file(file_path, max_samples=None):
    rows = []
    for i, row in enumerate(mrqa_eval_generator(file_path)):
        if max_samples is not None and i >= max_samples:
            break
        rows.append(row)
    return rows


def load_eval_datasets(root, in_domain=None, out_of_domain=None, max_samples=None):
    """Load enabled MRQA dev sets grouped by distribution ('in_domain' / 'out_of_domain')."""
    in_domain = in_domain or {}
    out_of_domain = out_of_domain or {}

    groups = {"in_domain": {}, "out_of_domain": {}}
    for group_key, toggles, file_map, subdir in (
        ("in_domain", in_domain, MRQA_INDOMAIN, INDOMAIN_SUBDIR),
        ("out_of_domain", out_of_domain, MRQA_OOD, OOD_SUBDIR),
    ):
        for name, enabled in toggles.items():
            if not enabled:
                continue
            if name not in file_map:
                raise ValueError(
                    f"Unknown MRQA {group_key} dataset '{name}'. "
                    f"Valid: {sorted(file_map)}"
                )
            path = os.path.join(root, subdir, file_map[name])
            groups[group_key][name] = _load_file(path, max_samples)

    return groups


def mrqa_eval_generator(file_path):
    """Yield QA samples with full gold answer lists from an MRQA gz file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if "header" in data or "context" not in data:
                continue

            context = data["context"]
            for qa in data["qas"]:
                answers = qa.get("answers")
                if not answers:
                    continue
                yield {
                    "context": context,
                    "question": qa["question"],
                    "answers": list(answers),
                    "qid": qa.get("qid"),
                }
