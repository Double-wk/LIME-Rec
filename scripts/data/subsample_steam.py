"""Create a deterministic user subsample of a Steam interaction file.

The command keeps every Nth user after sorting user IDs and then reapplies an
iterative user/item k-core filter. It writes to a separate file by default, so
the source interaction file is never modified accidentally.

Example:
    python -m scripts.data.subsample_steam \
      --input data/steam/train.jsonl \
      --output data/steam/train_liger_every7.jsonl \
      --stride 7 \
      --min-interactions 5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/steam/train.jsonl")
DEFAULT_OUTPUT = Path("data/steam/train_liger_every7.jsonl")


def iterative_kcore(
    records: list[dict[str, Any]], min_interactions: int
) -> list[dict[str, Any]]:
    """Repeatedly remove users and items below the requested interaction count."""
    iteration = 0
    while True:
        iteration += 1
        user_count = Counter(record["user_id"] for record in records)
        item_count = Counter(record["item_id"] for record in records)
        before = len(records)
        filtered = [
            record
            for record in records
            if user_count[record["user_id"]] >= min_interactions
            and item_count[record["item_id"]] >= min_interactions
        ]
        users = {record["user_id"] for record in filtered}
        items = {record["item_id"] for record in filtered}
        print(
            f"  pass {iteration}: {before:,} -> {len(filtered):,}  "
            f"users={len(users):,}  items={len(items):,}",
            flush=True,
        )
        if len(filtered) == before:
            return filtered
        records = filtered


def read_records(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate the interaction fields needed by the filter."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    required_fields = {"user_id", "item_id", "timestamp"}
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_path} at line {line_number}."
                )
            missing = required_fields.difference(record)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing required field(s) {names} in {input_path} at line {line_number}."
                )
            records.append(record)
    return records


def timestamp_key(record: dict[str, Any]) -> tuple[int, float | str]:
    """Provide a stable ordering for numeric and string timestamps."""
    timestamp = record["timestamp"]
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        return (0, float(timestamp))
    return (1, str(timestamp))


def write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records grouped by user and ordered by timestamp, atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_user: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_user[record["user_id"]].append(record)

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for user_id in sorted(by_user, key=str):
                for record in sorted(by_user[user_id], key=timestamp_key):
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source JSONL interaction file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination JSONL file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7,
        help="Keep every Nth sorted user ID (default: 7).",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum user and item frequency in the iterative k-core filter (default: 5).",
    )
    parser.add_argument(
        "--allow-overwrite-input",
        action="store_true",
        help="Allow --output to be the same file as --input. Disabled by default.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1.")
    if args.min_interactions < 1:
        parser.error("--min-interactions must be at least 1.")
    if args.input.resolve() == args.output.resolve() and not args.allow_overwrite_input:
        parser.error(
            "--output must differ from --input by default. "
            "Use --allow-overwrite-input only when replacement is intentional."
        )
    return args


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    print(f"[input] {args.input}: {len(records):,} interactions", flush=True)

    users = sorted({record["user_id"] for record in records}, key=str)
    selected_users = set(users[:: args.stride])
    print(
        f"[subsample] keeping {len(selected_users):,} / {len(users):,} users "
        f"(every {args.stride}th sorted user)",
        flush=True,
    )
    records = [record for record in records if record["user_id"] in selected_users]
    print(f"[subsample] {len(records):,} interactions before k-core", flush=True)

    records = iterative_kcore(records, args.min_interactions)
    write_records(records, args.output)

    users_kept = {record["user_id"] for record in records}
    items_kept = {record["item_id"] for record in records}
    print(f"[written] {args.output}", flush=True)
    print(
        f"  users={len(users_kept):,}  items={len(items_kept):,}  "
        f"interactions={len(records):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
