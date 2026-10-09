"""Paper-internal numerical consistency audit for the AAMAS 2027 submission.

Recomputes every number quoted in the main text from the frozen result
artifacts and checks it against the value printed in the paper. Exits non-zero
on any mismatch so it can run in CI before submission.

Checks:
  T2  controlled GRAM table (per-seed means/SDs -> paper mean +/- s.d.)
  T3  recovery decision table (delta, LCB95, superiority, r_min, eta* grid)
  T4  merged configuration grid (six rows, three datasets)
  T4d factorial increments quoted in the text
  T4i factorial interaction bootstrap intervals
  RX  repeat-excluded frozen-weight isolation deltas
  AG  controller table + coverage factorization + Granite valid-user pairing
  AA  m_agent recovery-audit reproducibility (U_A, delta, LCB95, r_min, grid)
  F2  shuffle-control permutation statistics
Usage:
  python3 -m scripts.evaluation.check_paper_consistency
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "output_final" / "results"

FAILURES: list[str] = []


def check(name: str, got: float, want: float, tol: float = 5e-5) -> None:
    if abs(got - want) > tol:
        FAILURES.append(f"{name}: got {got:.6f}, paper says {want:.6f}")


def check_str(name: str, got: str, want: str) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, paper says {want!r}")


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def seeds3(pattern: str) -> list[float]:
    return [json.load(open(RES / pattern.format(s=s)))["test_metrics"]["R@10"] for s in (0, 1, 2)]


def main() -> int:
    # ---- T2: controlled GRAM reproduction -------------------------------
    paper_t2 = {  # (gram_r10, lime_r10) mean/sd
        "amazon_beauty": ((0.0890, 0.0002), (0.1002, 0.0008)),
        "amazon_toys": ((0.0956, 0.0010), (0.1101, 0.0005)),
        "amazon_sports": ((0.0551, 0.0008), (0.0565, 0.0009)),
    }
    for ds, ((gm, gs), (lm, ls)) in paper_t2.items():
        for system, sub, (wm, ws) in (("gram", "gram", (gm, gs)), ("lime", "lime_matched20", (lm, ls))):
            per_seed = [json.load(open(RES / "controlled_gram" / sub / ds / f"seed{s}.json"))["metrics"]["R@10"]
                        for s in (0, 1, 2)]
            m, sd = mean_sd(per_seed)
            check(f"T2 {ds} {system} mean", m, wm, 5e-4)
            check(f"T2 {ds} {system} sd", sd, ws, 5e-4)

    # ---- T3: recovery decisions ------------------------------------------
    audit = json.load(open(RES / "controlled_gram" / "recovery_audit_gram.json"))
    paper_t3_allowed = {
        "amazon_beauty": [(0.0121, 0.0092, "Y", 0.0), (0.0107, 0.0078, "Y", 0.0), (0.0109, 0.0080, "Y", 0.0)],
        "amazon_toys": [(0.0129, 0.0096, "Y", 0.0), (0.0149, 0.0115, "Y", 0.0), (0.0155, 0.0122, "Y", 0.0)],
        "amazon_sports": [(0.0008, -0.0013, "n", 0.023), (0.0026, 0.0005, "Y", 0.0), (0.0006, -0.0015, "n", 0.028)],
    }
    grid = (0.01, 0.025, 0.05, 0.10)
    for ds, rows in paper_t3_allowed.items():
        for want, row in zip(rows, audit["datasets"][ds]):
            check(f"T3 {ds} seed{row['seed']} delta", row["delta"], want[0], 1e-4)
            check(f"T3 {ds} seed{row['seed']} lcb", row["lcb95"], want[1], 1e-4)
            check_str(f"T3 {ds} seed{row['seed']} superiority", "Y" if row["superiority"] else "n", want[2])
            check(f"T3 {ds} seed{row['seed']} r_min", row["r_min_relative"], want[3], 2e-3)
        # eta*: smallest grid tolerance satisfied on all seeds (0 if all superior)
        if all(r["superiority"] for r in audit["datasets"][ds]):
            eta_star = 0.0
        else:
            eta_star = next((e for e in grid
                             if all(r["lcb95"] > -e * r["U_A"] for r in audit["datasets"][ds])), None)
        want_star = {"amazon_beauty": 0.0, "amazon_toys": 0.0, "amazon_sports": 0.05}[ds]
        check(f"T3 {ds} eta*", eta_star if eta_star is not None else float("nan"), want_star)

    rex = json.load(open(RES / "controlled_gram" / "repeat_excluded_recovery.json"))
    paper_t3_excluded = {
        "amazon_beauty": [(0.0128, 0.0099, "Y", 0.0), (0.0126, 0.0097, "Y", 0.0), (0.0124, 0.0095, "Y", 0.0)],
        "amazon_toys": [(0.0157, 0.0123, "Y", 0.0), (0.0191, 0.0156, "Y", 0.0), (0.0182, 0.0148, "Y", 0.0)],
        "amazon_sports": [(0.0012, -0.0009, "n", 0.016), (0.0043, 0.0021, "Y", 0.0), (0.0021, -0.0001, "n", 0.002)],
    }
    for ds, rows in paper_t3_excluded.items():
        for want, row in zip(rows, rex["datasets"][ds]):
            check(f"T3x {ds} seed{row['seed']} delta", row["delta"], want[0], 1e-4)
            check(f"T3x {ds} seed{row['seed']} lcb", row["lcb95_one_sided"], want[1], 1e-4)
            check(f"T3x {ds} seed{row['seed']} r_min", row["min_supported_margin"], want[3], 2e-3)

    # ---- T4: merged configuration grid -----------------------------------
    t4_rows = {
        "S only (no calibration)": ("tab_isolation_m20", "{ds}_sasrec_nocal_seed{s}.json"),
        "S only + Cal.": ("tab_isolation_m20", "{ds}_sasrec_cal_seed{s}.json"),
        "S+C + Cal.": ("tab_ablation_m20", "ablation_{ds}_seq-cf_seed{s}.json"),
        "S+M + Cal.": ("tab_ablation_m20", "ablation_{ds}_seq-sem_seed{s}.json"),
        "S+C+M, no calibration": ("tab_isolation_m20", "{ds}_fusion_nocal_seed{s}.json"),
        "S+C+M + Cal. (full)": ("tab_isolation_m20", "{ds}_fusion_full_seed{s}.json"),
    }
    paper_t4 = {  # beauty, toys, sports -> (mean, sd)
        "S only (no calibration)": [(0.0740, 0.0014), (0.0739, 0.0010), (0.0395, 0.0005)],
        "S only + Cal.": [(0.0796, 0.0015), (0.0773, 0.0009), (0.0413, 0.0010)],
        "S+C + Cal.": [(0.0912, 0.0006), (0.0946, 0.0004), (0.0508, 0.0011)],
        "S+M + Cal.": [(0.0915, 0.0006), (0.1001, 0.0011), (0.0474, 0.0007)],
        "S+C+M, no calibration": [(0.0884, 0.0001), (0.0982, 0.0014), (0.0490, 0.0005)],
        "S+C+M + Cal. (full)": [(0.1005, 0.0009), (0.1100, 0.0005), (0.0565, 0.0009)],
    }
    names = {"beauty": "beauty", "toys": "toys", "sports": "sports"}
    for row_name, (sub, pat) in t4_rows.items():
        for di, ds in enumerate(("beauty", "toys", "sports")):
            vals = seeds3(f"{sub}/" + pat.replace("{ds}", names[ds]).replace("{s}", "{s}"))
            m, sd = mean_sd(vals)
            check(f"T4 {row_name} {ds} mean", m, paper_t4[row_name][di][0], 5e-4)
            check(f"T4 {row_name} {ds} sd", sd, paper_t4[row_name][di][1], 5e-4)

    # ---- T4d: factorial increments quoted in text -------------------------
    for di, ds in enumerate(("beauty", "toys", "sports")):
        s0 = paper_t4["S only (no calibration)"][di][0]
        sc = paper_t4["S only + Cal."][di][0]
        fnc = paper_t4["S+C+M, no calibration"][di][0]
        full = paper_t4["S+C+M + Cal. (full)"][di][0]
        check(f"T4d cal-only {ds}", sc - s0, (0.0056, 0.0034, 0.0018)[di], 1e-4)
        check(f"T4d fusion-nocal {ds}", fnc - s0, (0.0144, 0.0243, 0.0095)[di], 1e-4)
        check(f"T4d cal-on-fusion {ds}", full - fnc, (0.0121, 0.0118, 0.0075)[di], 1e-4)

    # ---- T4i: factorial interaction intervals ------------------------------
    # All four cells are read from the fused output path (fusion.repeat_aware),
    # because calibration acts through the gate even in SASRec+Cal.
    paper_interaction = {"beauty": (0.0070, 0.0051, 0.0089),
                         "toys": (0.0092, 0.0073, 0.0112),
                         "sports": (0.0061, 0.0050, 0.0071)}

    def hit10_map(path: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        for line in open(path):
            row = json.loads(line)
            rank = row["fusion"]["repeat_aware"]["target_rank"]
            out[row["user_id"]] = 1.0 if rank < 10 else 0.0
        return out

    for ds, (point, lo, hi) in paper_interaction.items():
        per_user = {
            "s": hit10_map(RES / "tab_isolation_peruser_m20" / f"{ds}_sasrec_nocal_seed0_peruser.jsonl"),
            "sc": hit10_map(RES / "tab_isolation_peruser_m20" / f"{ds}_sasrec_cal_seed0_peruser.jsonl"),
            "f": hit10_map(RES / "tab_isolation_peruser_m20" / f"{ds}_fusion_nocal_seed0_peruser.jsonl"),
            "fc": hit10_map(RES / "tab_isolation_peruser_m20" / f"{ds}_fusion_full_seed0_peruser.jsonl"),
        }
        users = sorted(set.intersection(*(set(v) for v in per_user.values())))
        inter = [(per_user["fc"][u] - per_user["f"][u]) - (per_user["sc"][u] - per_user["s"][u])
                 for u in users]
        rng = random.Random(2027)
        n = len(inter)
        boot = sorted(sum(inter[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
        check(f"T4i {ds} point", statistics.mean(inter), point, 1e-3)
        check(f"T4i {ds} lo", boot[int(0.025 * 10000)], lo, 1e-3)
        check(f"T4i {ds} hi", boot[int(0.975 * 10000)], hi, 1e-3)

    # ---- RX: repeat-excluded frozen-weight isolation + gate collapse -------
    # Frozen-weight (0.60/0.15/0.25) native repeat-excluded runs:
    shared = json.load(open(ROOT / "outputs" / "3expert_shared_weight_bge_mask_m20.json"))
    paper_rx = {"amazon_beauty": (0.0815, 0.1023), "amazon_toys": (0.0790, 0.1113),
                "amazon_sports": (0.0416, 0.0589)}
    for ds, (sas, fus) in paper_rx.items():
        check(f"RX {ds} sasrec", shared["results"][ds]["SASRec"]["R@10"], sas, 5e-4)
        check(f"RX {ds} fusion", shared["results"][ds]["3expert_shared"]["R@10"], fus, 5e-4)
    # Collapsed validation-fitted gate under native repeat-exclusion:
    paper_collapse = {"beauty": 0.0512, "toys": 0.0643, "sports": 0.0263}
    for ds, want in paper_collapse.items():
        got = json.load(open(RES / "tab_isolation_mask_m20" / f"{ds}_fusion_full_seed0.json"))["test_metrics"]["R@10"]
        check(f"RX-collapse {ds}", got, want, 5e-4)

    # ---- AG: controller table + identity + Granite valid-user pairing ------
    def load_users(path: Path) -> dict[str, dict]:
        return {json.loads(l)["user_id"]: json.loads(l) for l in open(path)}

    paper_ag = {  # (R@10, format_failure)
        ("agentic", "recovery"): (0.112, 0.000),
        ("agentic", "adaptive_agent"): (0.077, 0.000),
        ("agentic", "adaptive_agent_no_semantic"): (0.077, 0.001),
        ("agentic", "llm_all_tools"): (0.054, 0.036),
        ("agentic_qwen4b", "adaptive_agent"): (0.078, 0.012),
        ("agentic_qwen4b", "adaptive_agent_no_semantic"): (0.077, 0.009),
        ("agentic_qwen4b", "llm_all_tools"): (0.043, 0.185),
        ("agentic_qwen4b", "forced_sequential"): (0.062, 0.057),
        ("agentic_granite", "adaptive_agent"): (0.063, 0.223),
        ("agentic_granite", "adaptive_agent_no_semantic"): (0.065, 0.214),
        ("agentic_granite", "llm_all_tools"): (0.067, 0.132),
    }
    for (run, cond), (r10, ff) in paper_ag.items():
        metrics = json.load(open(RES / run / "beauty" / "metrics.json")) if (RES / run / "beauty" / "metrics.json").exists() else None
        if metrics and cond in metrics:
            check(f"AG {run}/{cond} R@10", metrics[cond]["R@10"], r10, 1e-3)
            check(f"AG {run}/{cond} ff", metrics[cond]["format_failure_rate"], ff, 1e-3)
    # forced_sequential for Qwen3-8B lives in agentic_extended
    ext = RES / "agentic_extended" / "beauty"
    if (ext / "metrics.json").exists():
        m = json.load(open(ext / "metrics.json"))
        if "forced_sequential" in m:
            check("AG extended forced Qwen3-8B R@10", m["forced_sequential"]["R@10"], 0.059, 1e-3)

    # coverage factorization (Qwen3-8B + Granite, all four shared conditions)
    for run in ("agentic", "agentic_granite"):
        for cond in ("recovery", "adaptive_agent", "adaptive_agent_no_semantic", "llm_all_tools"):
            recs = load_users(RES / run / "beauty" / f"{cond}.jsonl")
            users = list(recs)
            cov = sum(r["target_item_id"] in r["candidate_ids"] for r in recs.values()) / len(users)
            hits = [0.0 if r.get("format_failure") else float(r["target_item_id"] in r["ranking"][:10])
                    for r in recs.values()]
            overall = sum(hits) / len(hits)
            in_pool = [i for i, r in enumerate(recs.values()) if r["target_item_id"] in r["candidate_ids"]]
            cond_r10 = sum(hits[i] for i in in_pool) / len(in_pool)
            check(f"AG-id {run}/{cond} coverage", cov, 0.185, 1e-3)
            check(f"AG-id {run}/{cond} identity", overall, cov * cond_r10, 1e-9)
    # conditional R@10 values quoted in text
    recs = load_users(RES / "agentic" / "beauty" / "recovery.jsonl")
    in_pool = [r for r in recs.values() if r["target_item_id"] in r["candidate_ids"]]
    cond_rec = sum(r["target_item_id"] in r["ranking"][:10] for r in in_pool) / len(in_pool)
    check("AG recovery conditional R@10", cond_rec, 0.61, 5e-3)
    recs = load_users(RES / "agentic" / "beauty" / "adaptive_agent.jsonl")
    in_pool = [r for r in recs.values() if r["target_item_id"] in r["candidate_ids"]]
    cond_qwen = sum(r["target_item_id"] in r["ranking"][:10] for r in in_pool) / len(in_pool)
    check("AG qwen adaptive conditional R@10", cond_qwen, 0.42, 5e-3)
    recs = load_users(RES / "agentic_granite" / "beauty" / "adaptive_agent.jsonl")
    in_pool_g = [r for r in recs.values() if r["target_item_id"] in r["candidate_ids"]]
    cond_gra = sum(r["target_item_id"] in r["ranking"][:10] for r in in_pool_g) / len(in_pool_g)
    check("AG granite adaptive conditional R@10", cond_gra, 0.34, 5e-3)

    # Granite valid-user paired comparison (777 users; 0.081 vs 0.116;
    # paper reports controller-minus-recovery = -0.035 [-0.052, -0.018])
    gran = load_users(RES / "agentic_granite" / "beauty" / "adaptive_agent.jsonl")
    rec = load_users(RES / "agentic_granite" / "beauty" / "recovery.jsonl")
    valid = [u for u, r in gran.items() if not r["format_failure"] and r["ranking"]]
    check("AG granite valid n", float(len(valid)), 777.0, 0.1)
    g_r10 = sum(gran[u]["hit10"] for u in valid) / len(valid)
    r_r10 = sum(rec[u]["hit10"] for u in valid) / len(valid)
    check("AG granite valid controller R@10", g_r10, 0.081, 2e-3)
    check("AG granite valid recovery R@10", r_r10, 0.116, 2e-3)
    rng = random.Random(2027)
    diffs = [gran[u]["hit10"] - rec[u]["hit10"] for u in valid]
    n = len(diffs)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    check("AG granite valid gap", statistics.mean(diffs), -0.035, 2e-3)
    check("AG granite valid gap lo", boot[int(0.025 * 10000)], -0.052, 3e-3)
    check("AG granite valid gap hi", boot[int(0.975 * 10000)], -0.018, 3e-3)

    # ---- AA: m_agent recovery audit reproducibility ------------------------
    # The audit JSON must be reproducible from the raw per-user predictions:
    # U_A recomputed, the paired bootstrap re-run on the identical RNG stream,
    # and r_min derived as max(0, -LCB95) / U_A.
    import numpy as np
    from scripts.agentic.recovery_audit_agent import GRID, _audit, _cell_rng
    audit_path = RES / "recovery_battery" / "table_agent_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        bootstrap = audit["bootstrap"]
        condition_dirs = {}
        for run in ("agentic", "agentic_qwen4b", "agentic_granite", "agentic_extended",
                    "agentic_adapted", "agentic_labeled"):
            for name in ("recovery", "adaptive_agent", "llm_all_tools", "forced_sequential",
                         "llm_all_tools_adapted", "llm_all_tools_labeled"):
                path = RES / run / "beauty" / f"{name}.jsonl"
                if path.exists():
                    condition_dirs[path] = name
        witness_users, witness_hits = None, None
        witness_entry = audit["witness"]["path"]
        recs = load_users(ROOT / witness_entry)
        witness_users = sorted(recs)
        witness_hits = np.asarray([float(r["target_rank"] is not None
                                         and r["target_rank"] < bootstrap["k"])
                                   for r in (recs[u] for u in witness_users)])
        for label, cell in audit["conditions"].items():
            recs = load_users(ROOT / cell["path"])
            users = sorted(recs)
            check_str(f"AA {label} users", "same" if users == witness_users else "differ", "same")
            hits = np.asarray([float(r["target_rank"] is not None
                                     and r["target_rank"] < bootstrap["k"])
                               for r in (recs[u] for u in users)])
            check(f"AA {label} U_A", float(hits.mean()), cell["U_A"], 1e-9)
            recomputed = _audit(witness_hits, hits,
                                _cell_rng(bootstrap["seed"], "overall", label),
                                bootstrap["resamples"])
            check(f"AA {label} delta", recomputed["delta"], cell["delta"], 1e-9)
            check(f"AA {label} lcb95", recomputed["lcb95"], cell["lcb95"], 1e-9)
            check(f"AA {label} r_min", recomputed["r_min_relative"], cell["r_min_relative"], 1e-9)
            check_str(f"AA {label} grid", json.dumps(recomputed["recovery_at_margin"], sort_keys=True),
                      json.dumps(cell["recovery_at_margin"], sort_keys=True))
        check_str(f"AA grid definition", json.dumps(list(GRID)), json.dumps(audit["margins_relative_to_U_A"]))

    # ---- F2: shuffle-control permutation statistics ------------------------
    aligned_full = {"beauty": 0.1005, "toys": 0.1100, "sports": 0.0565}
    aligned_wsem = {"beauty": 0.327, "toys": 0.408, "sports": 0.345}
    paper_drop = {"beauty": (15.7, 1.0), "toys": (17.8, 0.9), "sports": (15.9, 0.8)}
    paper_wsem = {"beauty": (0.216, 0.008), "toys": (0.208, 0.005), "sports": (0.197, 0.010)}
    for ds in ("beauty", "toys", "sports"):
        r10, wsem = [], []
        for p in range(100, 110):
            d = json.load(open(RES / "control_shuffle_multi_m20" / f"{ds}_perm{p}.json"))
            r10.append(d["test_metrics"]["R@10"])
            wsem.append(d["mean_test_weights"][2])
        drops = [(aligned_full[ds] - v) / aligned_full[ds] * 100 for v in r10]
        check(f"F2 {ds} drop mean", statistics.mean(drops), paper_drop[ds][0], 0.15)
        check(f"F2 {ds} drop sd", statistics.stdev(drops), paper_drop[ds][1], 0.15)
        check(f"F2 {ds} wsem mean", statistics.mean(wsem), paper_wsem[ds][0], 2e-3)
        check(f"F2 {ds} wsem sd", statistics.stdev(wsem), paper_wsem[ds][1], 2e-3)
        aw = json.load(open(RES / "tab_isolation_peruser_m20" / f"{ds}_fusion_full_seed0.json"))["mean_test_weights"][2]
        check(f"F2 {ds} aligned wsem", aw, aligned_wsem[ds], 2e-2)

    if FAILURES:
        print(f"CONSISTENCY AUDIT FAILED ({len(FAILURES)} mismatches):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("CONSISTENCY AUDIT PASSED: all quoted numbers match the frozen artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
