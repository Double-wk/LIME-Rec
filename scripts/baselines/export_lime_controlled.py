"""Validate/copy matched-20 predictions into the shared controlled format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    required = {"user_id", "target_item_id", "ranking", "seed", "dataset", "model"}
    for row in rows:
        if not required.issubset(row) or row["model"] != "LIME-Rec-matched20" or len(row["ranking"]) < 20:
            raise RuntimeError("invalid matched-20 controlled prediction row")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
