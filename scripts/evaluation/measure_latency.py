"""Measure real per-query inference latency of LIME-Rec on this machine.

For each of N sampled users, measure wall-clock time of:
  - SASRec forward pass only (scoring all items)
  - ItemCF lookup
  - Semantic cosine over all items
  - Score normalization + 3-expert fusion + top-K selection

Reports mean / p50 / p95 / p99 across users. Single-query (batch=1) to mimic
online serving.

GRAM latency is NOT measured here (no GPU). Paper claims must be confined to
LIME-Rec's own absolute numbers + parameter-count comparison (architectural fact).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from lime_rec.evaluation import (
    load_configured_dataset,
    load_semantic_embeddings,
    normalize_scores,
)
from lime_rec.experts import ItemCFExpert, LLMSemanticExpert
from lime_rec.models import SASRec


def load_sasrec(path, item_ids):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SASRec(num_items=ck["num_items"], hidden=ck["hidden"],
                   maxlen=ck["maxlen"], num_layers=ck["num_layers"],
                   num_heads=ck["num_heads"], dropout=0.0,
                   architecture=ck.get("architecture", "legacy"))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    item_to_idx = ck.get("item_to_idx")
    if item_to_idx is None:
        item_to_idx = {iid: i + 1 for i, iid in enumerate(item_ids)}
    return model, ck, item_to_idx


def percentile(values, p):
    return float(np.percentile(values, p))


def cpu_model_name():
    """Return a useful CPU identifier on Linux, with portable fallbacks."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def configure_cpu_threads(n_threads: int):
    """Pin PyTorch and native BLAS/OpenMP pools for a batch-1 CPU benchmark."""
    torch.set_num_threads(n_threads)
    # The inter-op pool must be configured before parallel work starts.  It is
    # usually not on the serving hot path, but recording it prevents a hidden
    # source of host-to-host variation.
    try:
        torch.set_num_interop_threads(n_threads)
    except RuntimeError:
        # This only occurs if a caller has already begun inter-op work.
        pass
    # NumPy's semantic matrix-vector product is backed by OpenBLAS on the
    # benchmark host, so torch.set_num_threads alone is insufficient.
    return threadpool_limits(limits=n_threads)


def summarize(times_ms, name):
    return {
        "name": name,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": percentile(times_ms, 50),
        "p95_ms": percentile(times_ms, 95),
        "p99_ms": percentile(times_ms, 99),
        "n": len(times_ms),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sasrec-model", required=True)
    p.add_argument("--itemcf-model", required=True)
    p.add_argument("--semantic-emb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-users", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20,
                   help="Untimed warmup iterations")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--recency-decay", type=float, default=0.1,
                   help="Exponential semantic recency decay (paper default: 0.1)")
    p.add_argument("--w-seq", type=float, default=0.50)
    p.add_argument("--w-cf", type=float, default=0.20)
    p.add_argument("--w-sem", type=float, default=0.30)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                   help="Force device. Online serving usually = CPU.")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="PyTorch CPU threads (default: 1 for batch-1 reproducibility)")
    args = p.parse_args()

    weights = np.asarray([args.w_seq, args.w_cf, args.w_sem], dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        p.error("--w-seq, --w-cf, and --w-sem must be non-negative and sum to 1")
    if args.recency_decay < 0:
        p.error("--recency-decay must be non-negative")
    if args.torch_threads < 1:
        p.error("--torch-threads must be at least 1")
    thread_limit = None
    if args.device == "cpu":
        thread_limit = configure_cpu_threads(args.torch_threads)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ds = load_configured_dataset(args.config)
    print(f"[data] {cfg['name']}: users={ds.num_users} items={ds.num_items}", flush=True)

    item_index = {iid: i for i, iid in enumerate(ds.item_ids)}
    n_items = len(ds.item_ids)
    icf = ItemCFExpert(args.itemcf_model, ds.item_ids)
    sas_model, sas_ck, item_to_idx = load_sasrec(args.sasrec_model, ds.item_ids)
    sem_emb = load_semantic_embeddings(args.semantic_emb, ds.item_ids)
    semantic = LLMSemanticExpert(
        ds.item_ids,
        ds.item_text,
        embeddings=sem_emb,
        recency_decay=args.recency_decay,
    )
    print(f"[loaded] SAS hidden={sas_ck['hidden']} maxlen={sas_ck['maxlen']} "
          f"sem dim={sem_emb.shape[1]}", flush=True)

    sas_model.to(args.device)
    # Sample users
    rng = np.random.default_rng(42)
    eligible = [u for u in ds.user_ids if u in ds.test_by_user]
    sample = list(rng.choice(eligible, size=min(args.n_users + args.warmup, len(eligible)),
                              replace=False))

    # Pre-build histories (we don't time this; it's data prep, not inference)
    user_inputs = []
    for u in sample:
        hist = ds.history_by_user.get(u, []) + [ds.valid_by_user.get(u, "")]
        hist = [h for h in hist if h]
        if not hist:
            continue
        # SASRec input ids (already padded)
        maxlen = sas_ck["maxlen"]
        seq = [item_to_idx.get(i, 0) for i in hist]
        seq = [x for x in seq if x != 0]
        seq = seq[-maxlen:]
        pad_len = maxlen - len(seq)
        input_ids = [0] * pad_len + seq
        # Pre-build expert mappings
        history_ei = [item_index[h] for h in hist if h in item_index]
        user_inputs.append({
            "u": u,
            "history": hist,
            "history_ei": history_ei,
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=args.device),
        })

    print(f"[setup] {len(user_inputs)} users prepared, warmup={args.warmup}", flush=True)

    # Per-stage timing
    t_sas, t_icf, t_sem, t_norm_fuse_topk = [], [], [], []
    t_lime_total, t_sasrec_only = [], []

    K = args.top_k
    for i, ui in enumerate(user_inputs):
        is_warmup = i < args.warmup
        history = ui["history"]
        history_ei = ui["history_ei"]
        inp = ui["input_ids"]

        # === SAS ===
        torch.mps.synchronize() if args.device == "mps" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = sas_model.predict(inp)[0]
        torch.mps.synchronize() if args.device == "mps" else None
        t1 = time.perf_counter()

        # Map SAS logits to ds.item_ids order
        sa = np.full(n_items, -1e9, dtype=np.float32)
        logits_np = logits.cpu().numpy() if args.device != "cpu" else logits.numpy()
        for iid, mi in item_to_idx.items():
            ei = item_index.get(iid)
            if ei is not None and 0 <= mi < logits_np.shape[0]:
                sa[ei] = logits_np[mi]
        t2 = time.perf_counter()

        # === ItemCF ===
        ci = icf.score(ui["u"], history)
        t3 = time.perf_counter()

        # === Semantic (exponential-recency user vector + precomputed cosine) ===
        se = semantic.score(ui["u"], history)
        t4 = time.perf_counter()

        # === Normalize + fuse + topK ===
        for ei in history_ei:
            sa[ei] = -1e9
            ci[ei] = -1e9
            se[ei] = -1e9
        sa_n = normalize_scores(sa)
        ci_n = normalize_scores(ci)
        se_n = normalize_scores(se)
        fused = args.w_seq * sa_n + args.w_cf * ci_n + args.w_sem * se_n
        topk = np.argpartition(-fused, K)[:K]
        topk = topk[np.argsort(-fused[topk])]
        t5 = time.perf_counter()

        if not is_warmup:
            t_sas.append((t2 - t0) * 1000)       # SAS forward + numpy map
            t_icf.append((t3 - t2) * 1000)
            t_sem.append((t4 - t3) * 1000)
            t_norm_fuse_topk.append((t5 - t4) * 1000)
            t_lime_total.append((t5 - t0) * 1000)
            t_sasrec_only.append((t2 - t0) * 1000)  # SAS forward only

    result = {
        "dataset": cfg["name"],
        "n_items": n_items,
        "device": args.device,
        "top_k": K,
        "method": {
            "semantic_embedding": args.semantic_emb,
            "semantic_embedding_dim": int(sem_emb.shape[1]),
            "semantic_user_vector": "exponential_recency_weighted_mean",
            "recency_decay": args.recency_decay,
            "fixed_gate": {
                "w_seq": args.w_seq,
                "w_cf": args.w_cf,
                "w_sem": args.w_sem,
            },
        },
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": cpu_model_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "native_threadpools": threadpool_info(),
        },
        "stages": {
            "SAS_forward": summarize(t_sas, "SAS forward + scatter to item-space"),
            "ItemCF_lookup": summarize(t_icf, "ItemCF lookup"),
            "Semantic_cosine": summarize(t_sem, "Pre-computed semantic cosine"),
            "Fusion_topK": summarize(t_norm_fuse_topk, "Normalize + fuse + top-K"),
        },
        "total": {
            "SASRec_only": summarize(t_sasrec_only, "SASRec-only (baseline)"),
            "LIME_Rec_3expert": summarize(t_lime_total, "Full 3-expert LIME-Rec"),
        },
    }

    print(f"\n=== Latency on {cfg['name']} ({args.device}, {n_items} items, K={K}) ===")
    print(f"  {'Stage':<32}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}")
    for s_name, s in result["stages"].items():
        print(f"  {s_name:<32}{s['mean_ms']:>9.2f}ms{s['p50_ms']:>9.2f}ms"
              f"{s['p95_ms']:>9.2f}ms{s['p99_ms']:>9.2f}ms")
    print(f"  {'-'*72}")
    for t_name, t in result["total"].items():
        print(f"  {t_name:<32}{t['mean_ms']:>9.2f}ms{t['p50_ms']:>9.2f}ms"
              f"{t['p95_ms']:>9.2f}ms{t['p99_ms']:>9.2f}ms")
    print(f"\n  n={result['total']['LIME_Rec_3expert']['n']} timed users "
          f"({args.warmup} warmup discarded)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[saved] {args.out}")
    # Keep the controller alive through all timed calls, then restore the
    # caller's native threadpool settings as the process exits.
    if thread_limit is not None:
        thread_limit.restore_original_limits()


if __name__ == "__main__":
    main()
