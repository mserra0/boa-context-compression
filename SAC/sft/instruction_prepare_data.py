# Source: SAC (https://github.com/lx-Meteors/SAC), modified for this repository. See THIRD_PARTY_NOTICES.md.
import sys
import os
import gzip
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=False, default="instruction_rank-128_lm_mrqa",help='Directory including the configuration file')
    return parser.parse_args()


MRQA_SUBSET_ALIASES = {
    "SQuAD": "SQuAD",
    "NewsQA": "NewsQA",
    "TriviaQA": "TriviaQA-web",
    "SearchQA": "SearchQA",
    "HotpotQA": "HotpotQA",
    "NaturalQuestions": "NaturalQuestionsShort",
    "BioASQ": "BioASQ",
    "DROP": "DROP",
    "DuoRC": "DuoRC.ParaphraseRC",
    "RACE": "RACE",
    "RelationExtraction": "RelationExtraction",
    "TextbookQA": "TextbookQA",
}


def _iter_mrqa_file(path, subset):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "header" in record:
                continue
            context = record.get("context", "")
            for qa in record.get("qas", []):
                answers = qa.get("answers") or []
                if not answers:
                    for detected in qa.get("detected_answers", []):
                        text = detected.get("text")
                        if text:
                            answers.append(text)
                if not answers:
                    continue
                yield {
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": answers,
                    "subset": subset,
                }


def _local_mrqa_files(root, split):
    root = Path(root)
    if split == "train":
        dirs = [root / "train"]
    else:
        # SAC calls split="test" for evaluation. Match this repo's MRQA
        # benchmark by evaluating both in-domain dev and OOD dev_ood files.
        dirs = [root / "dev", root / "dev_ood"]
    files = []
    for directory in dirs:
        if directory.exists():
            files.extend(sorted(directory.glob("*.jsonl.gz")))
    return files


def _load_local_mrqa_examples(instruction_dataset_repo, split):
    files = _local_mrqa_files(instruction_dataset_repo, split)
    examples = []
    for path in files:
        subset = MRQA_SUBSET_ALIASES.get(path.stem.replace(".jsonl", ""), path.stem.replace(".jsonl", ""))
        examples.extend(_iter_mrqa_file(path, subset))
    return examples


def get_examples_list(instruction_dataset_repo, split, cache_dir="output"):
    instruction_dataset_repo_name = instruction_dataset_repo.rstrip("/").split('/')[-1]
    # cache long text for preventing full dataset traversal on each preparation.
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{instruction_dataset_repo_name}_{split}_instruction_dataset.json')
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            examples_list =  json.load(f)
        return examples_list
#############################gather all in-domain and out-of-domain dataset for testing##############################################
    examples_list = []
    if os.path.isdir(os.path.expanduser(str(instruction_dataset_repo))):
        examples_list = _load_local_mrqa_examples(instruction_dataset_repo, split)
    elif split == "train":
        dataset = load_dataset(instruction_dataset_repo, split=split, streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)
    elif split != "train":
        dataset = load_dataset(instruction_dataset_repo, split="test", streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)
        dataset = load_dataset(instruction_dataset_repo, split="validation", streaming=True)
        for example in tqdm(dataset, desc="Processing examples"):
            examples_list.append(example)    

        
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list

    
def get_ids(instruction_dataset_repo_name, examples_list, tokenizer, split):

    examples = []
    # info_list = []
    minn = 999999
    maxn = 0
    for example in tqdm(examples_list, desc="Processing examples"):

        ##########################取其中一个答案用于训练即可##############################
        # answer = ""
        # for i in range(len(example["answers"])):
        #     if i == len(example["answers"])-1:
        #         answer += example["answers"][i]
        #     else:
        #         if example[("answers")][i] == example[("answers")][i+1]:
        #             answer += example["answers"][i]
        #             break
        #         answer += example[("answers")][i] + " "
        ################################################################################
        answer = example["answers"][0]

        context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
        prompt = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(answer, add_special_tokens=False)["input_ids"]
        
        context_ids =  tokenizer("### Context:\n", add_special_tokens=False)["input_ids"] + context
        question_ids = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt \
                       + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
        answer_ids = answer + [tokenizer.eos_token_id] # tokenizer("</s>", add_special_tokens=False)["input_ids"]

        tot_len = len(context_ids) + len(question_ids) + len(answer_ids)
        minn = min(minn,tot_len)
        maxn = max(maxn,tot_len)

        # 生成 instruction_target，它是一个标签，用来指导模型学习预测目标
        instruction_target = [-100 for x in question_ids] + [x for x in answer_ids]
        instruction_target = instruction_target[1:]

        inputs = torch.LongTensor(context_ids)
        # 如果是训练的时候？
        if split == 'train':
            lm_target = torch.LongTensor(question_ids + answer_ids)
        else:
            lm_target = torch.LongTensor(question_ids)

        instruction_target = torch.LongTensor(instruction_target)

        if split == "test":
            examples.append({"input_ids":inputs,"lm_targets":lm_target})
        else:
            examples.append({"input_ids":inputs,"lm_targets":lm_target,
                            "instruction_target":instruction_target})
        # info_list.append(example)
    print(f"len range: [{minn}:{maxn}]")
    # with open(f'output/{instruction_dataset_repo_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
    #     json.dump(info_list, f, ensure_ascii=False)
    return examples

def get_examples(
    model_id,
    instruction_dataset_repo,
    samples_num,
    min_len,
    max_len,
    dataset_repo,
    cache_dir="output",
    streaming=None,
    **_unused_kwargs,
):
    
    model_name = model_id.split('/')[-1]
    instruction_dataset_repo_name = instruction_dataset_repo.split('/')[-1]
    os.makedirs(cache_dir, exist_ok=True)
    train_data_name = os.path.join(cache_dir, f"{instruction_dataset_repo_name}_train_"+model_name+f"_{samples_num}samples_instruction.pt")
    eval_data_name = os.path.join(cache_dir, f"{instruction_dataset_repo_name}_eval_"+model_name+f"_{samples_num}samples_instruction.pt")

    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    train_examples_list = get_examples_list(instruction_dataset_repo, split="train", cache_dir=cache_dir)
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test", cache_dir=cache_dir)

    random.seed(0)
    random.shuffle(train_examples_list)
    train_examples_list = train_examples_list[:samples_num]

    train_data = get_ids(instruction_dataset_repo_name, train_examples_list, tokenizer, split="train")
    test_data = get_ids(instruction_dataset_repo_name, test_examples_list, tokenizer, split="test")


    torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)
    
    return train_data, test_data
    
if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)

    if not os.path.exists("output"):
        os.makedirs("output")
    
    training_config = config["sft_training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]
    
    print(config["data_config"])
    train_examples, eval_examples = get_examples(**config["data_config"])
    print(len(train_examples))
    print(train_examples[50])
    print(len(eval_examples))
    print(eval_examples[50])

"""
python instruction_prepare_data.py --work_dir '../experiment/debug/quick'
"""