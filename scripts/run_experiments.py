"""Run a suite of experiments and aggregate the results into one table.

    python scripts/run_experiments.py --suite configs/ablations.yaml
    python scripts/run_experiments.py --suite configs/seeds.yaml --aggregate-only

Each suite entry overrides one axis of the base config. Results land in
``runs/<suite>/<name>/`` and a combined ``summary.json`` / ``summary.md`` is
written at the suite root.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from stkseg.config import Config
from stkseg.constants import DEVICE


def load_suite(path: Path) -> list[dict]:
    entries = yaml.safe_load(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a list of experiment entries")
    for entry in entries:
        if "name" not in entry:
            raise ValueError(f"every entry needs a 'name': {entry}")
    return entries


def aggregate(root: Path) -> list[dict]:
    """Collect whatever has finished so far, so partial suites still report."""
    rows = []
    for history_path in sorted(root.glob("*/history.json")):
        summary = json.loads(history_path.read_text())
        row = {"name": history_path.parent.name, "val_iou": summary.get("best_val_iou")}

        metrics_path = history_path.parent / "metrics_val.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            row["mIoU"] = metrics.get("mIoU")
            row["boundary_mIoU"] = metrics.get("boundary_mIoU")
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_no finished runs yet_\n"

    columns = ["name", "val_iou", "mIoU", "boundary_mIoU"]
    present = [c for c in columns if any(r.get(c) is not None for r in rows)]
    lines = ["| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
    for row in rows:
        cells = [
            f"{row[c]:.4f}" if isinstance(row.get(c), float) else str(row.get(c, ""))
            for c in present
        ]
        lines.append("| " + " | ".join(cells) + " |")

    scores = [r["val_iou"] for r in rows if isinstance(r.get("val_iou"), float)]
    if len(scores) > 1:
        lines.append("")
        lines.append(
            f"mean val mIoU {statistics.mean(scores):.4f} "
            f"+/- {statistics.stdev(scores):.4f} over {len(scores)} runs"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/ablations.yaml")
    parser.add_argument("--base", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--only", nargs="*", default=None, help="run a subset by name")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    entries = load_suite(suite_path)
    root = Path("runs") / suite_path.stem
    root.mkdir(parents=True, exist_ok=True)

    if args.only:
        entries = [e for e in entries if e["name"] in set(args.only)]

    if not args.aggregate_only:
        base = Config.from_yaml(args.base)
        for position, entry in enumerate(entries, start=1):
            name = entry["name"]
            overrides = {k: v for k, v in entry.items() if k != "name"}
            overrides.setdefault("train", {})["out_dir"] = str(root / name)
            if args.data_root:
                overrides.setdefault("data", {})["root"] = args.data_root
            if args.epochs:
                overrides["train"]["epochs"] = args.epochs

            cfg = base.with_overrides(**overrides)
            print(f"\n[{position}/{len(entries)}] {name}")
            if args.dry_run:
                print(json.dumps(overrides, indent=2))
                continue

            from stkseg.train import run

            run(cfg, device=args.device, quiet=True)

    rows = aggregate(root)
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    (root / "summary.md").write_text(to_markdown(rows))
    print("\n" + to_markdown(rows))


if __name__ == "__main__":
    main()
