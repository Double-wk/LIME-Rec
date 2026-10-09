# LIME-Rec

Lightweight late fusion for recovery-based mechanism auditing in sequential recommendation.

LIME-Rec combines a **SASRec sequential expert**, a **training-history ItemCF expert**,
and **offline semantic item embeddings**. At serving time, it scores and fuses these
sources without autoregressive language-model decoding. Pretrained language-derived
representations are still used to construct the semantic embeddings offline.

## Overview

![Figure 1: Recovery-based mechanism auditing](assets/figure1.png)

The figure presents the broader generative and agentic audit framework. This repository
contains the LIME-Rec sequential-recommendation implementation, training and evaluation
scripts, and archived experiment reports. DART-Rec controller code, controlled GRAM
baseline runners, and the manuscript/supplementary PDF are not included in this checkout.

## Installation

Python **3.10+**, Git, and `curl` are required. Run the commands below from the repository
root in a Bash shell. Install a PyTorch build appropriate for your hardware; CUDA is
recommended for training and embedding construction, while CPU is also supported.

```bash
git clone https://github.com/Double-wk/LIME-Rec.git
cd LIME-Rec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

Runtime dependencies and the optional test dependencies are declared in
[pyproject.toml](pyproject.toml).

## Quick start: Beauty pipeline

The [full pipeline](workflows/run_full_pipeline.sh) downloads preprocessed benchmark
files, builds ItemCF and semantic embeddings, trains SASRec, and evaluates a fixed
three-expert gate:

```bash
DEVICE=cuda DATASETS="amazon_beauty" SEED=0 \
  bash workflows/run_full_pipeline.sh
```

Use `DEVICE=cpu` for CPU execution or `DEVICE=auto` for automatic device selection in
training and embedding construction. The workflow's shared-weight evaluation uses
its evaluator's default CPU device.

To run all three Amazon domains:

```bash
DEVICE=cuda DATASETS="amazon_beauty amazon_toys amazon_sports" SEED=0 \
  bash workflows/run_full_pipeline.sh
```

The defaults use `BAAI/bge-base-en-v1.5` embeddings and fixed weights
`0.50,0.20,0.30` in the order SASRec, ItemCF, Semantic. Set `RUN_GATE_SELECTION=1`
to select global weights on validation data, or `RUN_PAIRWISE_ABLATION=1` to also
run pairwise ablations; the latter enables gate selection automatically.

Generated data, models, embeddings, and logs are written under `data/`, `outputs/`,
and `logs/`. Existing nonempty embeddings and SASRec checkpoints are reused. For a
fresh training run, move the corresponding generated checkpoint aside first.
These quick-start defaults are a runnable baseline, not a complete specification
for reproducing the archived repeat-aware results.

## Data

[Dataset configurations](configs) cover Amazon Beauty, Toys, Sports, and Yelp. The
[downloader](scripts/data/download_gram_data.py) fetches sequence and item-text files
from the GRAM repository and converts them to this project's JSONL format:

```bash
python -m scripts.data.download_gram_data beauty toys sports
# Optional boundary dataset:
python -m scripts.data.download_gram_data yelp
```

Amazon data is written to `data/amazon/<domain>/`; Yelp data is written to `data/yelp/`.
Each configuration specifies interaction/metadata paths, filtering thresholds, and a
leave-two-out split. Downloaded sequences retain their order through synthetic
monotonically increasing timestamps. Datasets and model weights are not bundled.

## Repeat-aware fusion

The [repeat-aware gate](scripts/evaluation/run_repeat_aware_gate.py) learns user-specific
expert weights and a bounded penalty for previously interacted items on the validation
split, then evaluates once on test. It uses full-catalog ranking with history items
eligible for recommendation. This differs from the quick-start shared-weight evaluator,
which masks history items by default.

After the Beauty pipeline has created its three expert assets, run:

```bash
python -m scripts.evaluation.run_repeat_aware_gate \
  --config configs/amazon_beauty.json \
  --sasrec-model outputs/models/amazon_beauty_sasrec.pt \
  --itemcf-model outputs/models/amazon_beauty_itemcf.json \
  --semantic-emb outputs/embeddings/amazon_beauty_bge_base.npz \
  --device cuda --seed 0 \
  --out outputs/repeat_aware_beauty_seed0.json
```

Use `--device cpu` for CPU evaluation. This example fits a new gate using the
quick-start checkpoint; its output is a new local run. Archived results also depend
on their original checkpoints, training settings, and evaluation protocol.

## Multiple seeds

Prepare data and embeddings with the full pipeline first, then use the
[multi-seed workflow](workflows/run_multiseed.sh):

```bash
DEVICE=cuda DATASETS="amazon_beauty amazon_toys amazon_sports" \
  SEEDS="0 1 2" bash workflows/run_multiseed.sh
```

This workflow trains seed-specific SASRec checkpoints and evaluates fixed shared
weights. It writes reports to `output/results/supplementary/multiseed/` and summarizes
them when at least two seeds are provided. It does not fit the repeat-aware gate;
run that evaluator separately with the desired seed-specific checkpoint.

## Archived results

The tracked [main summary](output/results/tab_main/summary.json) records the following
repeat-aware fusion results as mean ± sample standard deviation across seeds 0, 1, 2:

| Dataset | Recall@10 | NDCG@10 |
|---|---:|---:|
| Amazon Beauty | 0.09964 ± 0.00090 | 0.05874 ± 0.00047 |
| Amazon Toys | 0.11053 ± 0.00049 | 0.06636 ± 0.00026 |
| Amazon Sports | 0.05927 ± 0.00079 | 0.03411 ± 0.00053 |

The summary identifies validation-only gate fitting, full-catalog ranking, no hard
history masking, BGE-base embeddings, and initial weights `0.60,0.15,0.25`. These are
archived reports, not results regenerated by the quick-start commands above.

Additional artifacts are available in [output/results](output/results), including
expert-subset ablations, mechanism and isolation analyses, bootstrap intervals,
semantic-shuffling and fixed-penalty controls, encoder robustness, and Yelp boundary
runs. The [checkpoint manifest](output/provenance/checkpoint_manifest.json) records
historical asset hashes and training metadata; the referenced weights and training
logs are not bundled.

Recovery comparisons apply to the tested protocols. Offline semantic evidence remains
part of the system, and recommendation accuracy alone does not establish conversational
quality, explanation quality, or long-horizon utility.

## Repository map

| Path | Contents |
|---|---|
| [lime_rec](lime_rec) | Data loading, SASRec, expert scoring, and evaluation helpers |
| [configs](configs) | Dataset configurations |
| [scripts/data](scripts/data) | Benchmark download and conversion |
| [scripts/training](scripts/training) | SASRec training, ItemCF construction, and offline embeddings |
| [scripts/evaluation](scripts/evaluation) | Fusion, gates, ablations, controls, and result summaries |
| [workflows](workflows) | Full-pipeline and multi-seed Bash entry points |
| [tests](tests) | Unit tests for models, ItemCF, evaluation, and complementarity |
| [output](output) | Archived experiment reports and provenance |
| [assets](assets) | README overview figure |

## License

The code is distributed under the [Apache License 2.0](LICENSE).
