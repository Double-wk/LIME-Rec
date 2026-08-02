# LIME-Rec

Official implementation and reproducibility artifacts for
**Auditing Semantic Gains in Sequential Recommendation: A Lightweight Recovery Test**.

## Overview

LIME-Rec is a lightweight and auditable recovery test for examining how much
of the performance gain in semantic sequential recommendation can be recovered
without serving-time language-model inference. It combines three independently
inspectable full-catalog experts:

1. a SASRec sequential expert;
2. a training-only ItemCF co-occurrence expert;
3. a semantic expert based on frozen item-text embeddings.

The expert scores are normalized per user, combined through validation-only
score-level fusion, and adjusted using bounded history calibration. The fusion
gate is initialized at `(0.60, 0.15, 0.25)`, with `score_scale=20` and
`max_penalty=0.10`. Evaluation uses full-catalog ranking with
`mask_history=false`. Test interactions are used only for final evaluation.

## Main Results

The following results are averaged over three random seeds `{0,1,2}` under the
full-catalog, repeat-allowed evaluation protocol. R and N denote Recall and
NDCG, respectively.

| Dataset | R@5 | N@5 | R@10 | N@10 |
|---|---:|---:|---:|---:|
| Amazon Beauty | 0.0699 | 0.0491 | 0.0996 | 0.0587 |
| Amazon Toys | 0.0786 | 0.0561 | 0.1105 | 0.0664 |
| Amazon Sports | 0.0407 | 0.0281 | 0.0593 | 0.0341 |

Relative to GRAM, LIME-Rec improves R@10 by 12.0%, 12.0%, and 7.0% on
Beauty, Toys, and Sports, respectively.

## Repository Structure

```text
configs/        dataset configurations for Beauty, Toys, Sports, and Yelp
lime_rec/       core data, expert, model, and evaluation library
scripts/        data preparation, training, evaluation, and analysis CLIs
tests/          fast unit tests
workflows/      end-to-end and multi-seed reproduction wrappers
output/
  results/      precomputed experimental results
  provenance/   checkpoint training settings and protocol metadata
```

## Installation

LIME-Rec requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

LIME-Rec supports both CPU and GPU execution. Select `--device auto`, `cpu`,
`cuda`, or `mps` as appropriate. Exact bitwise equality is not guaranteed
across different hardware or PyTorch builds, but the reported metrics reproduce
within seed-level variation.

## Data

The main experiments use the 5-core Amazon Reviews 2014 Beauty, Toys, and
Sports datasets in the GRAM-preprocessed format, with a leave-two-out split.
Yelp is included only as a preliminary probe and is not part of the main Amazon
evaluation suite.

Download and convert the datasets with:

```bash
python -m scripts.data.download_gram_data beauty toys sports yelp
```

The converted files are written to
`data/<dataset>/{train.jsonl,meta.jsonl}`. Raw datasets are not redistributed;
the upstream repositories and original data providers define the applicable
licenses and terms. Model checkpoints are not included and can be regenerated
using the training commands below.

## Reproducing Table 1

The following example reproduces one Beauty run from data preparation through
final evaluation:

```bash
python -m scripts.data.download_gram_data beauty

python -m scripts.training.build_itemcf \
  --config configs/amazon_beauty.json \
  --out outputs/models/amazon_beauty_itemcf.json

python -m scripts.training.build_item_embeddings \
  --config configs/amazon_beauty.json \
  --model BAAI/bge-base-en-v1.5 \
  --device auto \
  --out outputs/embeddings/amazon_beauty_bge_base.npz

python -m scripts.training.train_sasrec \
  --config configs/amazon_beauty.json \
  --output outputs/models/amazon_beauty_sasrec_seed0.pt \
  --seed 0 \
  --hidden 64 --maxlen 50 --layers 2 --heads 2 --dropout 0.2 \
  --loss ce --validation-no-mask \
  --device auto

python -m scripts.evaluation.run_repeat_aware_gate \
  --config configs/amazon_beauty.json \
  --sasrec-model outputs/models/amazon_beauty_sasrec_seed0.pt \
  --itemcf-model outputs/models/amazon_beauty_itemcf.json \
  --semantic-emb outputs/embeddings/amazon_beauty_bge_base.npz \
  --device auto \
  --seed 0 \
  --initial-weights 0.60,0.15,0.25 \
  --score-scale 20 --max-penalty 0.10 \
  --out output/results/tab_main/repeat_aware_beauty_seed0.json
```

Repeat the experiment for seeds `{0,1,2}` and datasets
`{beauty,toys,sports}`, then summarize the results with:

```bash
python -m scripts.evaluation.summarize_repeat_aware_multiseed \
  output/results/tab_main/repeat_aware_beauty_seed*.json
```

The complete workflows are also available as shell wrappers:

```bash
bash workflows/run_full_pipeline.sh
bash workflows/run_multiseed.sh
```

## License

The original LIME-Rec source code and documentation are released under the
[Apache License 2.0](LICENSE). This license does not grant rights to third-party
datasets or pretrained encoders; their original providers define the applicable
terms.
