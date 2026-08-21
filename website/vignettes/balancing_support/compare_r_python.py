"""Build the matched R/Python RMSE comparison used by the report."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
R_RESULTS = ROOT / "reference" / "r_slide_rmse.csv"
PYTHON_RESULTS = ROOT / "results" / "python_rmse.csv"
OUTPUT = ROOT / "results" / "r_python_rmse_comparison.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    r_rows = read_rows(R_RESULTS)
    python_rows = read_rows(PYTHON_RESULTS)
    r_index = {
        (row["family"], row["setting"], row["frame"], row["outcome"]): row
        for row in r_rows
    }
    python_keys = [
        tuple(row[field] for field in ("family", "setting", "frame", "outcome"))
        for row in python_rows
    ]
    if len(python_keys) != 398 or len(set(python_keys)) != 398:
        raise RuntimeError(
            "Python results must contain the 398 unique cells in the "
            "displayed slide grid"
        )
    missing_keys = sorted(set(python_keys) - set(r_index))
    if missing_keys:
        raise RuntimeError(
            f"R reference is missing Python result keys: {missing_keys[:5]}"
        )
    comparison = []
    for python_row in python_rows:
        key = tuple(
            python_row[field] for field in ("family", "setting", "frame", "outcome")
        )
        r_row = r_index[key]
        r_rmse = float(r_row["rmse"])
        python_rmse = float(python_row["rmse"])
        combined_mcse = math.hypot(
            float(r_row["rmse_mcse"]), float(python_row["rmse_mcse"])
        )
        difference = python_rmse - r_rmse
        comparison.append(
            {
                "family": key[0],
                "setting": key[1],
                "frame": key[2],
                "outcome": key[3],
                "r_rmse": r_rmse,
                "r_rmse_mcse": float(r_row["rmse_mcse"]),
                "python_rmse": python_rmse,
                "python_rmse_mcse": float(python_row["rmse_mcse"]),
                "difference": difference,
                "relative_difference": difference / r_rmse,
                "combined_mcse": combined_mcse,
                "standardized_difference": difference / combined_mcse,
                "within_three_mcse": abs(difference) <= 3.0 * combined_mcse,
                "n_r": int(float(r_row["n"])),
                "n_python": int(float(python_row["n"])),
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparison[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparison)
    maximum = max(abs(float(row["standardized_difference"])) for row in comparison)
    within_three = sum(row["within_three_mcse"] for row in comparison)
    print(f"Wrote {len(comparison)} matched RMSE cells to {OUTPUT}")
    print(f"Cells within three combined Monte Carlo standard errors: {within_three}")
    print(f"Maximum absolute standardized RMSE difference: {maximum:.3f}")


if __name__ == "__main__":
    main()
