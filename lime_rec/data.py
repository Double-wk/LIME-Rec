"""Data loading: iterative 5-core filtering + leave-two-out split."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: int


@dataclass
class Dataset:
    """A bundle containing 5-core filtered interactions and leave-two-out splits."""

    name: str
    user_ids: List[str]
    item_ids: List[str]
    history_by_user: Dict[str, List[str]]   # train history, ordered by time
    valid_by_user: Dict[str, str]           # second-last item per user
    test_by_user: Dict[str, str]            # last item per user
    item_popularity: Dict[str, float]       # normalised popularity from train
    item_text: Dict[str, str] = field(default_factory=dict)

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)


def load_dataset(
    name: str,
    interactions_path: str,
    metadata_path: Optional[str] = None,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    split_manifest: Optional[str] = None,
) -> Dataset:
    if split_manifest is not None:
        spec = json.loads(Path(split_manifest).read_text())
        digest = hashlib.sha256()
        with Path(interactions_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if spec["dataset"] != name or digest.hexdigest() != spec["source_sha256"]:
            raise ValueError("temporal split dataset/source hash mismatch")
        history, valid, test = spec["history_by_user"], spec["valid_by_user"], spec["test_by_user"]
        items = spec["item_ids"]
        if set(valid) != set(test) or not set(valid) <= set(history):
            raise ValueError("temporal split users do not align")
        item_set = set(items)
        if len(items) != len(item_set) or items != sorted(items):
            raise ValueError("temporal catalog must be unique and sorted")
        if any(i not in item_set for i in list(valid.values()) + list(test.values())):
            raise ValueError("temporal target outside training catalog")
        if any(i not in item_set for h in history.values() for i in h):
            raise ValueError("temporal training item outside catalog")
        counts = Counter(i for h in history.values() for i in h)
        maximum = max(counts.values(), default=1)
        return Dataset(name=name, user_ids=sorted(history), item_ids=items,
                       history_by_user=history, valid_by_user=valid, test_by_user=test,
                       item_popularity={i: counts[i] / maximum for i in items},
                       item_text=_read_metadata(Path(metadata_path)) if metadata_path else {})
    rows = list(_read_interactions(Path(interactions_path)))
    rows = _iterative_kcore(rows, min_user_interactions, min_item_interactions)
    history_by_user, valid_by_user, test_by_user = _leave_two_out(rows)

    train_counts = Counter()
    for items in history_by_user.values():
        train_counts.update(items)
    max_count = max(train_counts.values()) if train_counts else 1
    item_popularity = {iid: count / max_count for iid, count in train_counts.items()}

    user_ids = sorted(history_by_user.keys())
    item_ids = sorted({iid for items in history_by_user.values() for iid in items}
                      | set(valid_by_user.values()) | set(test_by_user.values()))

    item_text = _read_metadata(Path(metadata_path)) if metadata_path else {}

    return Dataset(
        name=name,
        user_ids=user_ids,
        item_ids=item_ids,
        history_by_user=history_by_user,
        valid_by_user=valid_by_user,
        test_by_user=test_by_user,
        item_popularity={iid: item_popularity.get(iid, 0.0) for iid in item_ids},
        item_text=item_text,
    )


def _read_interactions(path: Path) -> Iterable[Interaction]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = _parse_json(line)
            user_id = str(row.get("reviewerID") or row.get("user_id") or row.get("user") or "")
            item_id = str(row.get("asin") or row.get("item_id") or row.get("item") or "")
            timestamp = int(row.get("unixReviewTime") or row.get("timestamp") or 0)
            if user_id and item_id:
                yield Interaction(user_id, item_id, timestamp)


def _read_metadata(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    text_by_item: Dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = _parse_json(line)
            item_id = str(row.get("asin") or row.get("item_id") or "")
            if not item_id:
                continue
            parts = [
                str(row.get("title") or "").strip(),
                str(row.get("brand") or "").strip(),
                _flatten_category(row.get("category") or row.get("categories")),
                str(row.get("description") or "").strip(),
            ]
            text_by_item[item_id] = " | ".join(p for p in parts if p)
    return text_by_item


def _flatten_category(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        flat = []
        for entry in value:
            if isinstance(entry, list):
                flat.extend(str(x) for x in entry if x)
            elif entry:
                flat.append(str(entry))
        return " > ".join(flat)
    return str(value)


def _iterative_kcore(
    rows: List[Interaction],
    min_user: int,
    min_item: int,
) -> List[Interaction]:
    current = rows
    while True:
        u_counts = Counter(r.user_id for r in current)
        i_counts = Counter(r.item_id for r in current)
        nxt = [
            r for r in current
            if u_counts[r.user_id] >= min_user and i_counts[r.item_id] >= min_item
        ]
        if len(nxt) == len(current):
            return nxt
        current = nxt


def _leave_two_out(rows: List[Interaction]):
    by_user: Dict[str, List[Interaction]] = defaultdict(list)
    for r in rows:
        by_user[r.user_id].append(r)

    history: Dict[str, List[str]] = {}
    valid: Dict[str, str] = {}
    test: Dict[str, str] = {}
    for user_id, user_rows in by_user.items():
        user_rows.sort(key=lambda r: r.timestamp)
        if len(user_rows) < 3:
            continue
        history[user_id] = [r.item_id for r in user_rows[:-2]]
        valid[user_id] = user_rows[-2].item_id
        test[user_id] = user_rows[-1].item_id
    return history, valid, test


def _parse_json(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)
