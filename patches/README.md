# External-baseline patches (provenance)

Per the repository's `.gitignore` policy, external baseline checkouts
(`external/GRAM/`, `external/LETTER/`) are not committed. This directory
records the exact modifications applied to them for the 2026-09-22
revision-round experiments.

## `gram_native_masked_decoding.patch` + `generation_trie_masked.py`

Native repeat-masked decoding for GRAM (`external/GRAM`):

- `src/utils/generation_trie_masked.py` (new): `CountedTrie` with per-node
  sequence counts, so each user's capped history window can be subtracted from
  the candidate constraint without rebuilding the trie; token allowed iff
  `count_global(prefix+t) > count_blocked(prefix+t)`.
- `src/arguments.py`: new flag `--mask_history_items`.
- `src/runner/distributed_runner_gram.py`: per-user blocked tries in
  `test_dataset_task` when the flag is on; test/validation history window
  matches `max_his`; TSV writer sanitizes embedded `\n`/`\r` in decoded strings
  (fixes lost user rows; newline-containing strings can never match a candidate
  lex_id at import, so evaluation semantics are unchanged).
- `src/main_generative_gram.py`: `--train 0` runs test-only inference from
  `--rec_model_path` (used to re-evaluate the final-epoch checkpoints that
  produced the repeat-allowed predictions).

Apply: `cd external/GRAM && git apply ../../patches/gram_native_masked_decoding.patch`
and copy `generation_trie_masked.py` to `src/utils/`.

## `letter_lime_audit_compat.patch`

Compatibility patches to run the LETTER-TIGER second-target audit
(`external/LETTER`, upstream `HonghuiBao2000/LETTER@master`):

- `RQ-VAE/models/rqvae.py`: skip the CF loss when `--alpha 0` (vanilla TIGER
  tokenizer on semantic embeddings only).
- `RQ-VAE/main.py`: do not load the CF embedding file when `--alpha 0`.
- `RQ-VAE/main.py`, `RQ-VAE/trainer.py`, `RQ-VAE/models/{rqvae,vq}.py`:
  `wandb` import made optional (imported but never used at runtime).
- `LETTER-TIGER/finetune.py`, `LETTER-TIGER/test.py`:
  `os.environ.setdefault("CUDA_VISIBLE_DEVICES", ...)` instead of hardcoded
  assignment so the launcher can pin GPUs.

Data conversion, training drivers, and the prediction exporter live in this
repository (`scripts/baselines/prepare_tiger_data.py`,
`scripts/baselines/export_tiger_predictions.py`,
`workflows/run_tiger_{rqvae,pipeline}.sh`).
