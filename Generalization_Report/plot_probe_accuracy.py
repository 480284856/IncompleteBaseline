"""Plot held-out action agreement from the classification/regression probe CSV."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_DIR = Path(__file__).resolve().parent
DATA = (REPORT_DIR.parent / "dev/maze4x4/cls_fatsterthan_reg"
        / "probe_results_greedy/metrics.csv")
OUTPUT = REPORT_DIR / "img/probe-action-agreement.png"


def main():
    series = defaultdict(list)
    with DATA.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["teacher"], row["task"])
            series[key].append((int(row["step"]) / 1000, float(row["action_accuracy"])))

    colors = {"classification": "#0072B2", "regression": "#D55E00"}
    styles = {"classification": "-", "regression": "--"}
    labels = {"classification": "Classification", "regression": "Q-value regression"}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, teacher, title in zip(
        axes,
        ("dqn", "actordqn"),
        ("DQN target teacher", "ActorDQN target teacher"),
    ):
        for task in ("classification", "regression"):
            points = sorted(series[(teacher, task)])
            updates, accuracy = zip(*points)
            ax.plot(updates, accuracy, color=colors[task], linestyle=styles[task],
                    marker="o", markersize=3.5, linewidth=1.8, label=labels[task])
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-2, 110)
        ax.set_ylim(0.15, 1.02)
        ax.set_xticks((0, 20, 40, 60, 80, 102.4),
                      ("0", "20", "40", "60", "80", "102.4"))
        ax.set_xlabel("Minibatch updates (thousands)")
        ax.grid(alpha=0.22)

    axes[0].set_ylabel("Held-out action agreement")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.23, wspace=0.08)
    fig.savefig(OUTPUT, dpi=240)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
