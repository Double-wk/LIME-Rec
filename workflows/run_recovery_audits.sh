#!/usr/bin/env bash
# Run every recovery audit that has complete predictions, then aggregate.
set -uo pipefail
cd "$(dirname "$0")/.."
P=output_final/results/controlled_gram/predictions
R=output_final/results/controlled_gram

have_all () { # prefix -> all 9 files present?
  local prefix=$1
  for ds in beauty toys sports; do for s in 0 1 2; do
    [[ -f "$P/${prefix}_amazon_${ds}_seed${s}.jsonl" ]] || return 1
  done; done
  return 0
}

run_audit () { # name gram_prefix lime_prefix comparison
  local name=$1 gp=$2 lp=$3 cmp=$4
  if have_all "$gp" && have_all "$lp"; then
    python -m scripts.evaluation.recovery_audit_gram \
      --predictions-dir "$P" --gram-prefix "$gp" --lime-prefix "$lp" \
      --comparison "$cmp" --out-json "$R/recovery_audit_${name}.json" && \
      echo "[audit-ok] $name"
  else
    echo "[audit-skip] $name (missing predictions)"
  fi
}

run_audit gram_full        gram            lime             "LIME-Rec-20 minus retrained GRAM (per-seed paired)"
run_audit gram_nocal       gram            lime_nocal       "LIME-Rec-20 no-calibration minus retrained GRAM"
run_audit gram_nosem       gram            lime_nosem       "LIME-Rec-20 no-semantic minus retrained GRAM"
run_audit gram_minilm      gram            lime_minilm      "LIME-Rec-20 MiniLM-prior minus retrained GRAM"
run_audit native_mask_init   gram_nativemask lime_maskinit   "LIME-Rec-20 init-gate repeat-masked minus GRAM native repeat-masked"
run_audit native_mask_refit  gram_nativemask lime_masked     "LIME-Rec-20 refit-gate repeat-masked minus GRAM native repeat-masked"
run_audit native_mask_frozen gram_nativemask lime_maskfrozen "LIME-Rec-20 frozen-fitted-gate repeat-masked minus GRAM native repeat-masked"
run_audit tiger            tiger           lime             "LIME-Rec-20 minus TIGER-LETTER-BGE (per-seed paired)"

python -m scripts.evaluation.summarize_recovery_battery
