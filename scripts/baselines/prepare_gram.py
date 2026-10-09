"""Audit the official GRAM checkout without mutating the LIME environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lime_rec.baselines.gram import sha256_file


DATASETS = {
    "Beauty": "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt",
    "Toys": "item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt",
    "Sports": "item_generative_indexing_hierarchy_v1_c32_l7_len32768_split.txt",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram-root", default="external/GRAM")
    parser.add_argument("--out", default="output_final/results/controlled_gram/provenance.json")
    args = parser.parse_args()
    root = Path(args.gram_root)
    if not root.is_dir():
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"status": "BLOCKED_BY_MISSING_GRAM_CHECKOUT",
                                   "clone_command": "git clone https://github.com/skleee/GRAM.git external/GRAM"},
                                  indent=2) + "\n", encoding="utf-8")
        print("BLOCKED_BY_MISSING_GRAM_CHECKOUT")
        print("git clone https://github.com/skleee/GRAM.git external/GRAM")
        raise SystemExit(2)

    required = [root / "src/main_generative_gram.py", root / "src/utils/utils.py",
                root / "src/processor/DistMultiDataTaskSampler.py", root / "requirements.txt"]
    for dataset, mapping in DATASETS.items():
        required.extend([root / f"rec_datasets/{dataset}/user_sequence.txt",
                         root / f"rec_datasets/{dataset}/{mapping}",
                         root / f"rec_datasets/{dataset}/item_plain_text.txt"])
    missing = [str(path) for path in required if not path.is_file()]
    status = "READY" if not missing else "BLOCKED_BY_UPSTREAM_ARTIFACT"
    payload = {"status": status, "gram_root": str(root),
               "gram_remote": _git(root, "remote", "-v"),
               "gram_commit": _git(root, "rev-parse", "HEAD"),
               "gram_status": _git(root, "status", "--short"),
               "missing_required_artifacts": missing,
               "key_file_sha256": {str(path.relative_to(root)): sha256_file(path)
                                      for path in required if path.is_file()},
               "environment_policy": "GRAM_PYTHON or conda run -n gram; LIME torch unchanged"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
