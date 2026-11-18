"""Run clean validation tuning and the matched corruption study."""

import os

# These small CPU matrix products benefit from a fixed, single-thread BLAS.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import csv
import json
import platform
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.special import expit

from losses import cross_entropy, logit_gradient, robust_cross_entropy, smoothed_target, worst_target
from models import Logistic, Split, TwoLayer, sample

METHODS = ("CE", "RobustCE", "Smoothing")
SIMULATIONS = {"gaussian": Logistic(), "semicircles": TwoLayer()}
TUNING_SEEDS = tuple(range(101, 106))
EVALUATION_SEEDS = tuple(range(201, 211))
CORRUPTION = (0.0, 0.15, 0.30)
EVAL_RADIUS = 0.10
RADII = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
SMOOTHING = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 0.60, 0.90, 1.20, 1.80)
UPDATES = 300
SPLIT_SIZES = (512, 512, 2048)
METRICS = ("accuracy", "ce", "robust_ce", "confidence")
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Setting:
    method: str
    rate: float
    strength: float


def prepare(simulation, seed, with_test=False):
    children = np.random.SeedSequence([seed, list(SIMULATIONS).index(simulation)]).spawn(5)
    train_rng, valid_rng, test_rng, init_rng, flip_rng = map(np.random.default_rng, children)
    train = sample(simulation, train_rng, SPLIT_SIZES[0])
    valid = sample(simulation, valid_rng, SPLIT_SIZES[1])
    test = sample(simulation, test_rng, SPLIT_SIZES[2]) if with_test else None
    initial = SIMULATIONS[simulation].initialize(init_rng)
    return train, valid, test, initial, flip_rng.random(len(train.y))


def evaluate(model, params, split):
    logits, _ = model.forward(params, split.x)
    return dict(accuracy=np.mean((logits >= 0) == split.y),
                ce=np.mean(cross_entropy(logits, split.y)),
                robust_ce=np.mean(robust_cross_entropy(logits, split.y, EVAL_RADIUS)),
                confidence=np.mean(expit(np.abs(logits))))


def fit(model, initial, train, labels, valid, setting, record=False):
    params = tuple(array.copy() for array in initial)
    moment = [np.zeros_like(array) for array in params]
    square = [np.zeros_like(array) for array in params]
    fixed = (smoothed_target(labels, setting.strength) if setting.method == "Smoothing"
             else labels)
    history = []
    if record:
        history.append({"update": 0, **evaluate(model, params, valid)})
    elapsed = 0.0
    started = perf_counter()
    for step in range(1, UPDATES + 1):
        logits, cache = model.forward(params, train.x)
        target = (worst_target(logits, labels, setting.strength)
                  if setting.method == "RobustCE" else fixed)
        residual = logit_gradient(logits, target) / len(labels)
        gradient = model.backward(params, train.x, cache, residual)
        progress = (step - 1) / (UPDATES - 1)
        rate = setting.rate * (0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2)
        for weight, first, second, derivative in zip(params, moment, square, gradient):
            first *= 0.9
            first += 0.1 * derivative
            second *= 0.999
            second += 0.001 * derivative**2
            weight -= rate * (first / (1 - 0.9**step)) / (
                np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        if record and step % 10 == 0:
            elapsed += perf_counter() - started
            history.append({"update": step, **evaluate(model, params, valid)})
            started = perf_counter()
    elapsed += perf_counter() - started
    if not all(np.isfinite(array).all() for array in params):
        raise FloatingPointError("Nonfinite fitted parameters")
    return params, elapsed, history


def tune(simulation, rows, failures):
    model = SIMULATIONS[simulation]
    datasets = {seed: prepare(simulation, seed) for seed in TUNING_SEEDS}
    strengths = dict(CE=(0.0,), RobustCE=RADII, Smoothing=SMOOTHING)
    best_settings = {}
    for method in METHODS:
        by_setting = {}
        for rate, strength in product(LEARNING_RATES, strengths[method]):
            setting = Setting(method, rate, strength)
            scores = []
            for seed, (train, valid, _, init, _) in datasets.items():
                details = {"simulation": simulation, "seed": seed, **asdict(setting)}
                try:
                    params, seconds, _ = fit(model, init, train, train.y, valid, setting)
                    score = evaluate(model, params, valid)
                except FloatingPointError as exc:
                    failures.append({"phase": "tuning", **details, "error": str(exc)})
                    continue
                scores.append(score["robust_ce"])
                rows.append({**details, **score, "seconds": seconds})
            if len(scores) == len(datasets):
                by_setting[setting] = np.mean(scores)
        best_settings[method] = min(by_setting, key=lambda s: (by_setting[s], s.strength, s.rate))
    return best_settings


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    tuning, testing, histories, failures, panels = [], [], [], [], {}
    settings = {sim: tune(sim, tuning, failures) for sim in SIMULATIONS}
    write_csv(results / "tuning.csv", tuning)
    metadata = dict(tuning_seeds=TUNING_SEEDS, evaluation_seeds=EVALUATION_SEEDS,
                    corruption=CORRUPTION, evaluation_radius=EVAL_RADIUS, radii=RADII,
                    smoothing=SMOOTHING, learning_rates=LEARNING_RATES, updates=UPDATES,
                    sizes=SPLIT_SIZES, representative_seed=EVALUATION_SEEDS[0],
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    machine=f"{platform.system()} {platform.machine()}",
                    settings={sim: {name: asdict(setting) for name, setting in chosen.items()}
                              for sim, chosen in settings.items()})
    (results / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["settings"], indent=2), flush=True)

    for simulation, model in SIMULATIONS.items():
        for seed in EVALUATION_SEEDS:
            train, valid, test, initial, flip = prepare(simulation, seed, with_test=True)
            representative = seed == EVALUATION_SEEDS[0]
            if representative:
                panels[f"{simulation}_x"] = test.x[:300]
                panels[f"{simulation}_y"] = test.y[:300]
            for eta in CORRUPTION:
                flipped = flip < eta
                labels = np.where(flipped, 1 - train.y, train.y)
                for method, setting in settings[simulation].items():
                    details = {"simulation": simulation, "seed": seed, "eta": eta,
                               **asdict(setting), "flips": int(flipped.sum())}
                    try:
                        params, seconds, history = fit(model, initial, train, labels, valid,
                                                       setting, record=True)
                        score = evaluate(model, params, test)
                    except FloatingPointError as exc:
                        failures.append({"phase": "evaluation", **details, "error": str(exc)})
                        continue
                    testing.append({**details, **score, "seconds": seconds})
                    histories.extend({**details, **row} for row in history)
                    if representative:
                        for index, array in enumerate(params):
                            panels[f"{simulation}_{eta}_{method}_{index}"] = array
    write_csv(results / "evaluation.csv", testing)
    write_csv(results / "learning.csv", histories)
    (results / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    np.savez_compressed(results / "representative.npz", **panels)
    print(f"Completed {len(tuning)} tuning runs and {len(testing)} evaluation runs. "
          f"Failures: {len(failures)}.")


if __name__ == "__main__":
    main()
