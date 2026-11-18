"""Regenerate report tables and plots from saved experiment data."""

import csv
import json
from pathlib import Path

from experiment import METHODS, SIMULATIONS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
COLORS = dict(CE="#15638F", RobustCE="#A53D28", Smoothing="#39754D")
MARKERS = dict(CE="o", RobustCE="s", Smoothing="^")
TITLES = dict(gaussian="Gaussian logistic regression", semicircles="Semicircle network")
LABELS = dict(accuracy="Accuracy", ce="Ordinary CE", robust_ce="RobustCE (eval. radius 0.10)",
              confidence="Mean confidence")
METRICS = ("accuracy", "ce", "robust_ce")


def read_csv(name):
    categorical = {"simulation", "method"}
    with (ROOT / "results" / name).open(encoding="utf-8", newline="") as stream:
        return [{key: value if key in categorical else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def subset(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def stats(rows, metric):
    values = np.array([row[metric] for row in rows])
    if len(values) < 2:
        raise ValueError(f"Insufficient runs for {metric}: {len(values)}")
    return values.mean(), values.std(ddof=1)


def write_table(name, columns, header, rows):
    path = ROOT / "tables" / f"{name}.tex"
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(r"\addlinespace" if row is None else " & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def tables(testing, protocol):
    summary, differences, chosen = [], [], []
    for sim, settings in protocol["settings"].items():
        for name, setting in settings.items():
            chosen.append([sim.capitalize(), name, f'{setting["rate"]:g}',
                           "--" if name == "CE" else f'{setting["strength"]:g}'])
        for eta in protocol["corruption"]:
            condition = subset(testing, simulation=sim, eta=eta)
            for method in METHODS:
                rows = subset(condition, method=method)
                formatted = []
                for key in (*METRICS, "confidence", "seconds"):
                    mean, sd = stats(rows, key)
                    factor, digits = (100, 2) if key == "accuracy" else (1, 3)
                    formatted.append(rf"${factor * mean:.{digits}f}\pm{factor * sd:.{digits}f}$")
                summary.append([sim.capitalize(), f"{eta:.2f}", method, *formatted])
            pairs = (("RobustCE", "CE", "R--CE"), ("Smoothing", "CE", "S--CE"),
                     ("RobustCE", "Smoothing", "R--S"))
            for method, comparison, label in pairs:
                baseline = {r["seed"]: r for r in subset(condition, method=comparison)}
                paired = [r for r in subset(condition, method=method) if r["seed"] in baseline]
                if len(paired) < 2:
                    raise ValueError("Insufficient matched runs for paired intervals")
                intervals = []
                for metric in METRICS:
                    d = np.array([row[metric] - baseline[row["seed"]][metric] for row in paired])
                    if metric == "accuracy":
                        d *= 100
                    center = d.mean()
                    margin = t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    digits = 2 if metric == "accuracy" else 3
                    intervals.append(f"{center:+.{digits}f} [{center - margin:+.{digits}f}, "
                                     f"{center + margin:+.{digits}f}]")
                differences.append([sim.capitalize(), f"{eta:.2f}", label,
                                    str(len(paired)), *intervals])
            summary.append(None)
            differences.append(None)
        chosen.append(None)
    write_table("settings", "@{}llrr@{}", ["Simulation", "Method", "Initial rate",
                                            r"$\delta$ or $\varepsilon$"], chosen[:-1])
    write_table("test", "@{}lrlrrrrr@{}", ["Simulation", r"$\eta$", "Method", r"Acc. (\%)",
                "CE", "RobustCE", "Conf.", "Time (s)"], summary[:-1])
    write_table("paired", "@{}lrlrrrr@{}", ["Simulation", r"$\eta$", "Contrast", "$n$",
                "Acc. (pp)", "CE", "RobustCE"], differences[:-1])


def show_band(axis, xs, rows_by_x, metric, method):
    values = np.array([stats(rows, metric) for rows in rows_by_x])
    mean, sd = values.T
    color = COLORS[method]
    axis.plot(xs, mean, label=method, color=color, marker=MARKERS[method],
              markevery=max(1, len(xs) // 6), markersize=4, linewidth=1.4)
    axis.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
    axis.grid(alpha=0.2)


def save(fig, name):
    fig.savefig(ROOT / "figures" / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def selection_plot(tuning, protocol):
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.5), sharex=True, layout="constrained")
    radii = protocol["radii"]
    for col, sim in enumerate(SIMULATIONS):
        axes[0, col].set_title(TITLES[sim])
        rows_by_radius = []
        for radius in radii:
            rows = subset(tuning, simulation=sim, method="RobustCE", strength=radius)
            candidates = [subset(rows, rate=rate) for rate in protocol["learning_rates"]]
            complete = [candidate for candidate in candidates
                        if len(candidate) == len(protocol["tuning_seeds"])]
            rows_by_radius.append(min(complete, key=lambda group: stats(group, "robust_ce")[0]))
        chosen = protocol["settings"][sim]["RobustCE"]["strength"]
        for row, metric in enumerate(("robust_ce", "ce", "confidence")):
            axis = axes[row, col]
            show_band(axis, radii, rows_by_radius, metric, "RobustCE")
            axis.axvline(chosen, color="#333333", linestyle="--", linewidth=1)
            if col == 0:
                axis.set_ylabel(LABELS[metric])
        axes[-1, col].set_xlabel(r"Training radius $\delta$")
    save(fig, "radius")


def corruption_plot(testing, protocol):
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.5), sharex=True, layout="constrained")
    rates = protocol["corruption"]
    for col, sim in enumerate(SIMULATIONS):
        axes[0, col].set_title(TITLES[sim])
        for row, metric in enumerate(METRICS):
            axis = axes[row, col]
            for method in METHODS:
                groups = [subset(testing, simulation=sim, method=method, eta=eta) for eta in rates]
                show_band(axis, rates, groups, metric, method)
            if col == 0:
                axis.set_ylabel(LABELS[metric])
        axes[-1, col].set_xticks(rates, ["0%", "15%", "30%"])
        axes[-1, col].set_xlabel("Training-label flip probability")
    axes[0, 1].legend(frameon=False, fontsize=9)
    save(fig, "corruption")


def learning_plot(learning, protocol):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.7), sharex=True, layout="constrained")
    updates = sorted({row["update"] for row in learning})
    for col, sim in enumerate(SIMULATIONS):
        axes[0, col].set_title(TITLES[sim])
        for row, eta in enumerate((0.0, protocol["corruption"][-1])):
            axis = axes[row, col]
            for method in METHODS:
                series = subset(learning, simulation=sim, method=method, eta=eta)
                groups = [subset(series, update=update) for update in updates]
                show_band(axis, updates, groups, "ce", method)
            if col == 0:
                axis.set_ylabel(f"Validation CE\n{eta:.0%} training flips")
        axes[-1, col].set_xlabel("Adam update")
    axes[0, 1].legend(frameon=False, fontsize=9)
    save(fig, "learning")


def boundary_plot(simulation, protocol, data):
    model = SIMULATIONS[simulation]
    x, y = data[f"{simulation}_x"], data[f"{simulation}_y"]
    axes_limits = [np.linspace(low - 0.2, high + 0.2, 180)
                   for low, high in zip(x.min(axis=0), x.max(axis=0))]
    mx, my = np.meshgrid(*axes_limits)
    grid = np.column_stack((mx.ravel(), my.ravel()))
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 4.9), sharex=True, sharey=True,
                             layout="constrained")
    for row, eta in enumerate((0.0, protocol["corruption"][-1])):
        for col, method in enumerate(METHODS):
            axis = axes[row, col]
            n_arrays = 2 if simulation == "gaussian" else 4
            params = tuple(data[f"{simulation}_{eta}_{method}_{i}"] for i in range(n_arrays))
            logits, _ = model.forward(params, grid)
            probability = expit(logits).reshape(mx.shape)
            color = axis.contourf(mx, my, probability, levels=np.linspace(0, 1, 21),
                                  cmap="RdBu_r", vmin=0, vmax=1)
            if probability.min() < 0.5 < probability.max():
                axis.contour(mx, my, probability, levels=[0.5], colors="#232323", linewidths=1)
            for label, fill, marker in [(0, "#2B6DA6", "o"), (1, "#C24D3C", "^")]:
                points = x[y == label]
                axis.scatter(*points.T, marker=marker, c=fill, s=12, linewidth=0.35,
                             edgecolors="white", alpha=0.83)
            if row == 0:
                axis.set_title(method)
            if col == 0:
                axis.set_ylabel(f"{eta:.0%} flips\nFeature 2")
            if row == 1:
                axis.set_xlabel("Feature 1")
    fig.colorbar(color, ax=axes, label="Predicted class-1 probability", shrink=0.94,
                 ticks=np.linspace(0, 1, 5))
    save(fig, simulation)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for name in ("figures", "tables"):
        (ROOT / name).mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "results" / "protocol.json").read_text(encoding="utf-8"))
    testing, tuning, learning = map(read_csv, ("evaluation.csv", "tuning.csv", "learning.csv"))
    tables(testing, protocol)
    selection_plot(tuning, protocol)
    learning_plot(learning, protocol)
    corruption_plot(testing, protocol)
    with np.load(ROOT / "results" / "representative.npz") as data:
        for simulation in SIMULATIONS:
            boundary_plot(simulation, protocol, data)


if __name__ == "__main__":
    main()
