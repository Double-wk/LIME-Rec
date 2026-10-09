# Anonymous submission revision experiments

Run from the `lime-rec/` repository directory in the existing project environment.
All commands below require real input artifacts; variable names are placeholders
for their actual paths. New output directories must not already exist. The
original baseline outputs are never overwritten. These programs have local CPU
checks; new trained-model results are **NOT_RUN** unless a result file says otherwise.

## Witness ladder and gate diagnosis (E1/E4/E6)

Export each seed separately, using checkpoints actually trained at maxlen=20:

```bash
python -m scripts.revision.export_expert_scores \
  --config "$REV_CONFIG" --sasrec-model "$REV_SASREC" \
  --itemcf-model "$REV_ITEMCF" --semantic-emb "$REV_EMBEDDINGS" \
  --history-cap 20 --device cuda --out "$REV_SCORE_CACHE"
python -m scripts.revision.witness_ladder \
  --cache "$REV_SCORE_CACHE" --out "$REV_LADDER_ALLOWED"
python -m scripts.revision.witness_ladder \
  --cache "$REV_SCORE_CACHE" --masked --out "$REV_LADDER_MASKED"
python -m scripts.revision.plot_surface \
  --summary "$REV_LADDER_ALLOWED/summary.json" --out "$REV_SURFACE_PDF"
```

The ladder includes individual experts, pairs, calibration variants, full fusion,
equal weights, CombSUM, Borda, and RRF. Global weights and history penalties are
selected by validation R@10, with complexity/penalty tie breaks. Validation CE is
a diagnostic at the declared scale. Borda/RRF CE must not be compared to weighted
fusion CE because their score scales differ. The default search has 66 simplex
weights × 4 penalties plus 10 named candidates, and records every configuration.
`frozen_validation_selection.json` is written before reading test shards.

The optional `--exploratory-test-surface` evaluates all test configurations after
selection, and labels this as exploratory. The reported fraction at the selected
witness's point utility is **not** a statistically supported recovery region.
Use paired target predictions to establish recovery; that region remains pending.
For learned-gate collapse, use the existing `run_repeat_aware_gate.py` with its
`--dump-mechanism` output and record weight distributions together with validation
CE/R@10. The new simplex runner does not implement a new learned-gate trainer or
automatically select gate hyperparameters. Those sweeps remain pending.

Repeat masking occurs before min–max normalization and ignores masked extrema.
If a held-out target occurs in the allowed history, the masked ladder rejects the
comparison. Define a common repeat-excluded evaluation cohort for **both** systems
or a different target convention explicitly before running; never silently drop
users and compare different populations. Score-cache storage grows as
users × 3 experts × catalog size; use shards and provision disk before exporting.

Three independently trained SASRec seeds can be combined as one ensemble:

```bash
python -m scripts.revision.sasrec_ensemble \
  --cache "$REV_CACHE0" --cache "$REV_CACHE1" --cache "$REV_CACHE2" \
  --out "$REV_ENSEMBLE"
```

Add `--masked` for the masked convention. This is one three-checkpoint ensemble,
not three independent ensemble replicates. Additional seeds 3/4 require new
training of targets and witnesses under the same frozen protocol.

## Paired statistics (W6)

Create a JSON manifest enumerating the **entire intended family**, for example:

```json
{"margin":0.05,"comparisons":[
  {"id":"beauty_seed0","target":"target.jsonl","witness":"witness.jsonl"}
]}
```

```bash
python -m scripts.revision.audit_predictions \
  --manifest "$REV_FAMILY" --out "$REV_PAIRED_REPORT"
python -m scripts.revision.failure_profiles \
  --config "$REV_CONFIG" --target "$REV_TARGET_PREDICTIONS" \
  --witness "$REV_WITNESS_PREDICTIONS" --out "$REV_ERROR_PROFILE"
```

Predictions must have unique `user_id`, identical user sets and `target_item_id`,
and a duplicate-free `ranking`. Sampling holds each trained model fixed. The
joint bootstrap includes target uncertainty in `R-(1-eta)A`; the older plug-in
tolerance result is retained for comparison. Finite-sample checks use conservative
Clopper–Pearson bounds on paired outcome cells, and Holm adjusts conservative
Hoeffding p-value bounds. This is not a Tango interval or a McNemar test for a
nonzero margin. Conservative checks may be inconclusive when bootstrap recovery
holds. Neither test establishes uncertainty over the training-seed population.
Existing test analyses are retrospective sensitivity, never preregistration.
Failure strata use full history and training popularity and are exploratory.

## Global temporal split (E3)

```bash
python -m scripts.revision.temporal_split \
  --config "$REV_CONFIG" --raw-interactions "$REV_TIMESTAMPED_SOURCE" \
  --train-end "$REV_TRAIN_CUTOFF" --validation-end "$REV_VAL_CUTOFF" \
  --out "$REV_TEMPORAL_SPLIT"
```

**Current converted Amazon JSONLs contain per-user sequence positions as timestamps.**
They cannot support a global temporal audit. Use the original timestamped source,
freeze cutoffs before model results, and retrain all compared systems. This exporter
constructs the k-core/catalog using training events only and reports excluded
future users/items. The generated `.config.json` is supported by the local loader,
evaluator and SASRec trainer. External GRAM/TIGER/LIGER data exporters still need
explicit adaptation to this same manifest; reusing their original splits is invalid.
Metadata provenance and post-cutoff textual fields must also be checked.

## Oracle and constrained controllers (E8/E10)

```bash
python -m scripts.revision.tool_headroom \
  --data "$REV_PREPARED_TEST" --budget 2 --fixed-tools sasrec,itemcf \
  --controller-predictions "$REV_CONTROLLER_PREDICTIONS" --out "$REV_HEADROOM"
python scripts/agentic/run_budgeted_tools.py evaluate \
  --data "$REV_PREPARED_TEST" --router "$REV_VALIDATION_ROUTER" \
  --llm --json-schema --format-retries 2 --out-dir "$REV_SCHEMA_RUN"
```

The fixed tools above must be independently frozen on validation, rather than
copied if another dataset has a different selected reference. Repeat budget 1
with a single frozen tool. Oracle maximizes R@10 and NDCG separately over legal
tool subsets with the unchanged final fusion. It is a hindsight diagnostic, not
a deployable policy. Prepared caches and per-controller predictions are currently
missing locally; aggregate summaries cannot reconstruct oracle headroom.

Configure the existing `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` privately. Schema
runs require server JSON-schema support; HTTP rejection fails the run instead of
silently falling back. Shared-pool `run_agentic_test.py` also has `--json-schema`;
the dedicated adapted-controller CLI has not yet been extended. Keep controller
model, users, budget, prompt and final fusion constant when comparing variants.
Retain failures/retries/tokens/latency and report overall plus valid-output-only
paired results. Synthetic checks do not establish real-server schema compatibility.

## Remaining evidence work

E2 official-target audit is in the supplementary protocol note. New five-seed
target replications, global temporal targets, Yelp signed-calibration experiments,
Sports full-history witnesses, learned-gate sweeps, stronger controllers and
statistically supported parameter regions remain pending. P2 task expansion is
deferred. Do not insert outcomes for these into the paper until raw outputs exist.
