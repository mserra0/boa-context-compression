#!/usr/bin/env bash
# Download the MRQA 2019 shared-task files used by Boa and store them under the file names
# expected by src/dataset.py and src/evaluation.
#
# Usage:
#   bash scripts/download_mrqa.sh [DEST] [--ood]
#     DEST   target directory (default: data/MRQA)
#     --ood  also download the six out-of-domain dev sets into DEST/dev_ood (needed for the
#            full evaluation in src/evaluation; not needed for training)
#
# Resulting layout:
#   DEST/train/{SQuAD,NewsQA,TriviaQA,HotpotQA,SearchQA,NaturalQuestions}.jsonl.gz
#   DEST/dev/{SQuAD,NewsQA,TriviaQA,HotpotQA,SearchQA,NaturalQuestions}.jsonl.gz
#   DEST/dev_ood/{BioASQ,DROP,DuoRC,RACE,RelationExtraction,TextbookQA}.jsonl.gz   (with --ood)
#
# Source: https://github.com/mrqa/MRQA-Shared-Task-2019 (official release v2). The in-domain
# train split is about 1.3 GB. Already downloaded files are kept, interrupted ones are resumed.
set -euo pipefail

DEST="data/MRQA"
OOD=0
for arg in "$@"; do
    case "$arg" in
        --ood) OOD=1 ;;
        *) DEST="$arg" ;;
    esac
done

BASE="https://s3.us-east-2.amazonaws.com/mrqa/release/v2"

fetch() {  # fetch <split> <official name> <local name> <target subdir>
    local split=$1 official=$2 local_name=$3 sub=$4
    local out="$DEST/$sub/$local_name.jsonl.gz"
    mkdir -p "$DEST/$sub"
    if [ -f "$out" ]; then
        echo "exists: $out"
        return
    fi
    echo "downloading $split/$official -> $out"
    curl -fL --retry 3 -C - -o "$out.part" "$BASE/$split/$official.jsonl.gz"
    mv "$out.part" "$out"
}

for split in train dev; do
    fetch "$split" SQuAD SQuAD "$split"
    fetch "$split" NewsQA NewsQA "$split"
    fetch "$split" TriviaQA-web TriviaQA "$split"
    fetch "$split" HotpotQA HotpotQA "$split"
    fetch "$split" SearchQA SearchQA "$split"
    fetch "$split" NaturalQuestionsShort NaturalQuestions "$split"
done

if [ "$OOD" = 1 ]; then
    fetch dev BioASQ BioASQ dev_ood
    fetch dev DROP DROP dev_ood
    fetch dev DuoRC.ParaphraseRC DuoRC dev_ood
    fetch dev RACE RACE dev_ood
    fetch dev RelationExtraction RelationExtraction dev_ood
    fetch dev TextbookQA TextbookQA dev_ood
fi
echo "done: $DEST"
