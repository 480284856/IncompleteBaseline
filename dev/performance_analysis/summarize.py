"""Average evaluation metrics across seeds for each maze and policy."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DIRECTORY = Path(__file__).resolve().parent
INPUT = DIRECTORY / "results.csv"
OUTPUT = DIRECTORY / "summary.csv"

POLICY_NAMES = {
    ("ActorDQN", "normal"): "ActorDQN",
    ("DQN", "normal"): "Basic DQN",
    ("ActorDQN", "random_baseline_checkpoint"): "Random Actor",
    ("Random", "uniform"): "Uniform Random",
}
METRICS = ("mean_steps_all", "mean_steps_success", "success_rate")


def summarize(input_path: Path = INPUT, output_path: Path = OUTPUT):
    groups = defaultdict(list)
    with input_path.open(newline="") as file:
        for row in csv.DictReader(file):
            policy = POLICY_NAMES[(row["model"], row["variant"])]
            groups[(row["maze"], policy)].append(row)

    columns = ["maze", "policy", "seeds", *METRICS,
               *(f"{metric}_std" for metric in METRICS)]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for (maze, policy), rows in sorted(groups.items()):
            result = {"maze": maze, "policy": policy, "seeds": len(rows)}
            for metric in METRICS:
                values = [float(row[metric]) for row in rows]
                result[metric] = mean(values)
                result[f"{metric}_std"] = stdev(values)
            writer.writerow(result)

    print(f"Saved {len(groups)} rows to {output_path}")


if __name__ == "__main__":
    summarize()
