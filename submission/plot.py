# This is based on https://github.com/matsui528/annbench/blob/main/plot.py
import argparse
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from itertools import cycle

from datasets import get_query_count


marker = cycle(('p', '^', 'h', 'x', 'o', 's', '*', '+', 'D', '1', 'X')) 
linestyle = cycle((':', '-', '--'))

def draw(lines, xlabel, ylabel, title, filename, with_ctrl, width, height):
    """
    Visualize search results and save them as an image
    Args:
        lines (list): search results. list of dict.
        xlabel (str): label of x-axis, usually "recall"
        ylabel (str): label of y-axis, usually "query per sec"
        title (str): title of the result_img
        filename (str): output file name of image
        with_ctrl (bool): show control parameters or not
        width (int): width of the figure
        height (int): height of the figure
    """
    plt.figure(figsize=(width, height))

    for line in lines:
        for key in ["xs", "ys", "label", "ctrls"]:
            assert key in line

    for line in lines:
        plt.plot(line["xs"], line["ys"], label=line["label"], marker=next(marker), linestyle=next(linestyle))
        if with_ctrl:
            for x, y, ctrl in zip(line["xs"], line["ys"], line["ctrls"]):
                plt.annotate(text=str(ctrl), xy=(x, y),
                             xytext=(x, y+50))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which="both")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.cla()

def get_pareto_frontier(line):
    data = sorted(zip(line["ys"], line["xs"], line["ctrls"]),reverse=True)
    line["xs"] = []
    line["ys"] = []
    line["ctrls"] = []

    cur = 0
    for y, x, label in data:
        if x > cur:
            cur = x
            line["xs"].append(x)
            line["ys"].append(y)
            line["ctrls"].append(label)

    return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=['task1', 'task2', 'task3'],
        help="task type to plot",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="dataset name to plot (e.g. gooaq-small); inferred from the CSV if only one dataset is present",
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()

    with open(args.csvfile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    if args.dataset is None:
        datasets = {row["dataset"] for row in data if row["task"] == args.task}
        if len(datasets) == 0:
            print(f"No results found for task={args.task!r}")
            raise SystemExit(1)
        if len(datasets) > 1:
            print(f"Multiple datasets found for task={args.task!r}: {sorted(datasets)}")
            print("Please specify --dataset explicitly.")
            raise SystemExit(1)
        args.dataset = datasets.pop()
        print(f"Inferred dataset: {args.dataset}")

    lines = {}
    for res in data:
        if res["task"] != args.task or res["dataset"] != args.dataset:
            continue
        algo = res["algo"]
        if algo not in lines:
            lines[algo] = {
                "xs": [],
                "ys": [],
                "ctrls": [],
                "label": algo,
            }
        lines[algo]["xs"].append(float(res["recall"]))
        lines[algo]["ys"].append(get_query_count(args.dataset, args.task) / float(res["querytime"]))
        try:
            run_identifier = res["params"].split("query=")[1]
        except IndexError:
            run_identifier = res["params"]
        lines[algo]["ctrls"].append(run_identifier)

    if not lines:
        print(f"No results found for dataset={args.dataset!r} task={args.task!r}")
        raise SystemExit(1)

    draw([get_pareto_frontier(line) for line in lines.values()],
         "Recall", "QPS (1/s)", f"{args.dataset} / {args.task}",
         f"result_{args.dataset}_{args.task}.png", True, 10, 8)