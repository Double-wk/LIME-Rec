"""Formal recovery audit for serving-time language-model control ($m_{\\mathrm{agent}}$).

Companion to scripts.evaluation.recovery_audit_gram: same criterion, same paired
user-level percentile bootstrap (10,000 resamples, seed 2027), same sign
convention, applied to the controller conditions evaluated over a shared
candidate pool.

Witness = the deterministic pool recovery path (`recovery`). Each controller
condition is audited as a target A; the witness R attains recovery on that
condition's user set iff LCB95(U(R) - U(A)) > -delta, superiority iff > 0.

Unlike the generative audit there is a single serving run per condition (no
training-seed replicates), so the audit reports no cross-seed robustness
statement; the bootstrap bounds prediction-sample uncertainty conditional on the
frozen run.

Each (scope, condition) cell derives its own RNG stream, so adding a condition
later does not perturb the intervals of conditions already reported.

Regime stratification: `--strata` repeats the audit inside terciles of an
observable regime variable z (history length, expert disagreement, pool size)
computed from the candidate cache. Stratum decisions are descriptive; they do
not carry a separate significance claim.

Usage:
  python3 -m scripts.agentic.recovery_audit_agent \
    --witness recovery=output_final/results/agentic/beauty/recovery.jsonl \
    --condition "Qwen3-8B adaptive=output_final/results/agentic/beauty/adaptive_agent.jsonl" \
    --candidate-cache outputs/agentic/candidate_cache/amazon_beauty_test_1000.jsonl \
    --out-json output_final/results/recovery_battery/table_agent_audit.json
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from lime_rec.agentic.metrics import evaluate_agentic

GRID = (0.01, 0.025, 0.05, 0.10)
STRATA = ("history", "ambiguity", "pool")


def _rows(path: str) -> dict[str, dict]:
    return {json.loads(line)["user_id"]: json.loads(line)
            for line in Path(path).read_text().splitlines() if line}


def _hits(rows: dict[str, dict], users: list[str], k: int) -> np.ndarray:
    return np.asarray([1.0 if (rows[u]["target_rank"] is not None
                               and rows[u]["target_rank"] < k) else 0.0 for u in users])


def _cell_rng(seed: int, scope: str, label: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{scope}|{label}".encode())])


def _audit(witness: np.ndarray, target: np.ndarray, rng: np.random.Generator,
           n_resamples: int) -> dict:
    diff = witness - target
    boot = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
    lcb = float(np.percentile(boot, 5.0))
    u_a = float(target.mean())
    r_min = max(0.0, -lcb) / u_a if u_a > 0 else float("nan")
    return {"U_A": u_a, "U_R": float(witness.mean()), "delta": float(diff.mean()),
            "lcb95": lcb, "superiority": bool(lcb > 0.0), "r_min_relative": r_min,
            "recovery_at_margin": {f"{eta:g}": bool(lcb > -eta * u_a) for eta in GRID}}


def _regime_variables(cache_rows: list[dict]) -> dict[str, dict[str, float]]:
    """Observable per-user regime variables computable without labels."""
    out: dict[str, dict[str, float]] = {}
    for row in cache_rows:
        scores = np.asarray(row["expert_scores"], dtype=np.float64)
        tops = [set(np.argsort(-s, kind="stable")[:10].tolist()) for s in scores]
        overlaps = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                union = len(tops[i] | tops[j])
                overlaps.append(len(tops[i] & tops[j]) / union if union else 0.0)
        out[row["user_id"]] = {
            "history": float(row["history_length"]),
            "pool": float(len(row["candidate_ids"])),
            "ambiguity": 1.0 - float(np.mean(overlaps)),
        }
    return out


def _buckets(values: np.ndarray, fixed_edges: list[float] | None = None, n_strata: int = 3):
    edges = (list(fixed_edges) if fixed_edges
             else np.percentile(values, [100.0 * i / n_strata for i in range(1, n_strata)]).tolist())
    masks = []
    for index in range(len(edges) + 1):
        lower = -np.inf if index == 0 else edges[index - 1]
        upper = np.inf if index == len(edges) else edges[index]
        masks.append((values > lower) & (values <= upper))
    return edges, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness", required=True, metavar="LABEL=PATH",
                        help="Deterministic recovery path (witness R).")
    parser.add_argument("--condition", action="append", required=True, metavar="LABEL=PATH",
                        help="Controller condition to audit as target A; repeatable.")
    parser.add_argument("--candidate-cache", required=True,
                        help="Frozen test candidate cache used for regime variables.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tex")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--strata", default=",".join(STRATA),
                        help="Comma-separated regime variables; empty to skip.")
    parser.add_argument("--z-bins", default="",
                        help="Prespecified bucket edges as VAR=e1,e2;VAR=e1,e2. Variables "
                             "without an entry use quantile terciles. Prespecified edges "
                             "are preferred for the primary regime axis because the "
                             "bucketing then does not depend on the observed distribution "
                             "and is comparable across mechanisms and datasets.")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    fixed_edges: dict[str, list[float]] = {}
    for spec in [item for item in args.z_bins.split(";") if item]:
        variable, _, raw = spec.partition("=")
        fixed_edges[variable] = [float(value) for value in raw.split(",") if value]

    def split(spec: str) -> tuple[str, str]:
        label, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"expected LABEL=PATH, got {spec!r}")
        return label, path

    witness_label, witness_path = split(args.witness)
    conditions = [split(spec) for spec in args.condition]

    witness_rows = _rows(witness_path)
    cache = [json.loads(line) for line in Path(args.candidate_cache).read_text().splitlines() if line]
    regime = _regime_variables(cache)
    users = sorted(witness_rows)
    missing = [u for u in users if u not in regime]
    if missing:
        raise RuntimeError(f"{len(missing)} witness users absent from the candidate cache")

    result = {
        "criterion": ("recovery witness iff LCB95(U(R)-U(A)) > -delta; "
                      "superiority iff LCB95 > 0"),
        "audited_mechanism": "m_agent: serving-time language-model control over fixed expert evidence",
        "witness": {"label": witness_label, "path": witness_path, "R@10": float(
            _hits(witness_rows, users, args.k).mean())},
        "bootstrap": {"unit": "user", "resamples": args.n_resamples, "seed": args.seed,
                      "method": "percentile, one-sided lower 95%", "k": args.k},
        "margins_relative_to_U_A": list(GRID),
        "n_users": len(users),
        "seed_scope": ("single frozen serving run per condition; no cross-seed "
                       "robustness statement"),
        "note": args.note,
        "z_bins": fixed_edges,
        "regime_variable_definitions": {
            "history": "interactions visible before the test recommendation "
                       "(training history plus the validation interaction)",
            "ambiguity": "1 - mean pairwise Jaccard overlap of the three experts' "
                         "within-pool top-10 candidate sets",
            "pool": "number of candidates in the shared pool",
        },
        "conditions": {},
        "strata": {},
    }

    witness_hits = _hits(witness_rows, users, args.k)
    for label, path in conditions:
        rows = _rows(path)
        if sorted(rows) != users:
            raise RuntimeError(f"user set of {label} differs from the witness user set")
        hits = _hits(rows, users, args.k)
        cell = _audit(witness_hits, hits, _cell_rng(args.seed, "overall", label), args.n_resamples)
        cell["path"] = path
        cell["metrics"] = evaluate_agentic(list(rows.values()))
        result["conditions"][label] = cell
        print(f"[{label}] U_A={cell['U_A']:.3f} U_R={cell['U_R']:.3f} "
              f"d={cell['delta']:+.4f} lcb={cell['lcb95']:+.4f} "
              f"r_min={100*cell['r_min_relative']:.1f}% sup={cell['superiority']}")

    for variable in [v for v in args.strata.split(",") if v]:
        if variable not in STRATA:
            raise ValueError(f"unknown regime variable {variable!r}; expected one of {STRATA}")
        values = np.asarray([regime[u][variable] for u in users])
        edges, masks = _buckets(values, fixed_edges.get(variable))
        blocks = []
        for index, mask in enumerate(masks):
            subset = [u for u, keep in zip(users, mask) if keep]
            if not subset:
                continue
            w = _hits(witness_rows, subset, args.k)
            block = {"stratum": index, "n_users": len(subset),
                     "z_range": [float(values[mask].min()), float(values[mask].max())],
                     "conditions": {}}
            for label, path in conditions:
                rows = _rows(path)
                cell = _audit(w, _hits(rows, subset, args.k),
                              _cell_rng(args.seed, f"{variable}:{index}", label), args.n_resamples)
                block["conditions"][label] = cell
            blocks.append(block)
            print(f"[{variable} q{index}] n={len(subset)} "
                  f"z=[{block['z_range'][0]:.3f},{block['z_range'][1]:.3f}]")
        result["strata"][variable] = {"edges": edges, "buckets": blocks}

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] {out}")

    if args.out_tex:
        def tex(text: str) -> str:
            return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Target (serving-time LM control) & $U_A$ & $U_R$ & $\Delta$ "
                 r"& $\mathrm{LCB}_{95}$ & $\eta^{\star}$ \\", r"\midrule"]
        for label, cell in result["conditions"].items():
            grid = next((eta for eta in GRID if cell["recovery_at_margin"][f"{eta:g}"]), None)
            grid_s = r"$0\%$ (sup.)" if cell["superiority"] else (
                f"${100*grid:g}\\%$" if grid is not None else "none")
            lines.append(f"{tex(label)} & {cell['U_A']:.3f} & {cell['U_R']:.3f} & "
                         f"{cell['delta']:+.4f} & {cell['lcb95']:+.4f} & {grid_s} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        Path(args.out_tex).write_text("\n".join(lines) + "\n")
        print(f"[saved] {args.out_tex}")


if __name__ == "__main__":
    main()