"""Evaluate training mazes, average seeds, and subtract validation means."""

import csv
import logging
from pathlib import Path

import torch

if __package__:
    from .analysis import performance_analysis
    from .summarize import METRICS, summarize
else:
    from analysis import performance_analysis
    from summarize import METRICS, summarize


DIRECTORY = Path(__file__).resolve().parent
TRAIN_RESULTS = DIRECTORY / "train_results.csv"
TRAIN_SUMMARY = DIRECTORY / "train_summary.csv"
VALIDATION_SUMMARY = DIRECTORY / "summary.csv"
GAP = DIRECTORY / "gap.csv"


def write_gap():
    with TRAIN_SUMMARY.open(newline="") as file:
        training = {(row["maze"], row["policy"]): row for row in csv.DictReader(file)}
    with VALIDATION_SUMMARY.open(newline="") as file:
        validation = {(row["maze"], row["policy"]): row for row in csv.DictReader(file)}

    columns = ["maze", "policy", *(f"{metric}_gap" for metric in METRICS)]
    with GAP.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for maze, policy in sorted(training):
            train_row = training[(maze, policy)]
            validation_row = validation[(maze, policy)]
            writer.writerow({
                "maze": maze,
                "policy": policy,
                **{f"{metric}_gap": float(train_row[metric]) - float(validation_row[metric])
                   for metric in METRICS},
            })
    print(f"Saved {len(training)} rows to {GAP}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(1)
    performance_analysis(is_evaluation=False, output_path=TRAIN_RESULTS)
    summarize(TRAIN_RESULTS, TRAIN_SUMMARY)
    write_gap()
