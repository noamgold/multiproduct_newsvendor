"""Rebuild Figures 2 and 3 of Section 6.2 (predictive performance on the
Online Retail data), the companions of rebuild_figure1.py.

Both figures plot the average test loss (24) of three models over the same
three rolling training schemes as Figure 1:

* the uniform-demand model (12), with a = min and b = max of the training demands;
* the data-driven model (23); and
* the distribution-free model (19) with the portfolio variance bound,
  weights ((p-s)*sigma/mu)**2 and lambda = 1e-4 (``--lambda``).

Figure 2 uses k_i = 1 and varies the fixed-cost budget K, i.e. the number of
products stocked.  Figure 3 uses fixed costs correlated with profit as in
(22): k_i = pi_i + epsilon, where pi_i is the expected profit at the
unconstrained optimum of the distribution-free model with lambda = 0
(Gallego and Moon), estimated from the training window's mean and standard
deviation, and K = kappa/101 * sum_i k_i for varying kappa.  The same k_i
are used for all three models in a training window.

Costs, salvage values, capacity, schemes, and the loss follow
rebuild_figure1.py exactly (``--cost-model random`` by default, Section 6.2).
The uniform and distribution-free models are solved with the Lagrangian
(dual) cut method of submodular_cuts.py, and the data-driven model as a MILP
with Gurobi.  Products with no training-window sales (a = b = 0) never order
and are left out of the cut masters; a product with constant positive
training demand (a = b > 0) gets b = a + 1e-6 * max(1, a) so that the
uniform model is defined.

Typical commands::

    python rebuild_figures2_3.py --figure 2
    python rebuild_figures2_3.py --figure 3
    python rebuild_figures2_3.py --figure 2 --schemes 20 --levels 40 --max-windows 1

Results are checkpointed per (level, training window, model) and resumed on
re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rebuild_figure1 as f1

DEFAULT_LAMBDA = 1.0e-4
FIXED_COST_EPSILON = 1.0e-9
UNIFORM_WIDENING = 1.0e-6
DEFAULT_TIME_LIMIT_SECONDS = 600.0

FIGURE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "2": {
        "levels": tuple(range(20, 141, 10)),
        "xlabel": r"$K$",
        "level_name": "K",
    },
    "3": {
        "levels": tuple(range(20, 61, 5)),
        "xlabel": r"$\kappa$",
        "level_name": "kappa",
    },
}

SCHEME_TITLES: Mapping[str, str] = {
    "increasing": "(a) Using an increasing training set size",
    "20": r"(b) Using a fixed training set size $|T_{\mathrm{train}}|=20$",
    "10": r"(c) Using a fixed training set size $|T_{\mathrm{train}}|=10$",
}

MODELS = ("uniform", "data_driven", "distribution_free")
MODEL_LABELS = {
    "uniform": "Uniform",
    "data_driven": "Data-driven",
    "distribution_free": "Distribution-free",
}
MODEL_COLORS = {
    "uniform": "#1f77b4",
    "data_driven": "#e76f51",
    "distribution_free": "#2a9d8f",
}

RECORD_FIELDS = (
    "scheme",
    "level",
    "train_begin_week",
    "train_end_week",
    "test_week",
    "model",
    "budget",
    "solver_status",
    "solve_time_seconds",
    "selected_products",
    "sum_x",
    "test_loss",
)

SUMMARY_FIELDS = (
    "Scheme",
    "Level",
    "Uniform",
    "Uniform_Std",
    "Data_driven",
    "DD_Std",
    "Distribution_free",
    "DF_Std",
    "Solved_Windows",
)

require = f1.require


def uniform_inputs(
    demand_train: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, b, x_u) of the uniform model for one training window."""
    a = np.min(demand_train, axis=1)
    b = np.max(demand_train, axis=1)
    constant = (b <= a) & (a > 0.0)
    b = np.where(constant, a + UNIFORM_WIDENING * np.maximum(1.0, a), b)
    x_u = np.where(b > a, a + (b - a) * (price - cost) / (price - salvage), 0.0)
    return a, b, x_u


def gallego_moon_profits(
    mu: np.ndarray,
    sigma: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> np.ndarray:
    """pi_i = theta_i(0) - theta_i(x_i^0), x_i^0 the lambda = 0 minimizer.

    theta_i'(x) = 0 gives (x - mu)/sqrt(sigma^2 + (x - mu)^2) = t with
    t = 1 - 2(c - s)/(p - s), which lies in (-1, 1) because 0 < c - s < p - s.
    """
    t = 1.0 - 2.0 * (cost - salvage) / (price - salvage)
    x0 = np.maximum(0.0, mu + sigma * t / np.sqrt(1.0 - t**2))
    theta = f1.distribution_free_theta
    return theta(np.zeros_like(mu), mu, sigma, price, cost, salvage) - theta(
        x0, mu, sigma, price, cost, salvage
    )


def fixed_cost_data(
    figure: str,
    level: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> tuple[np.ndarray, float]:
    """(k, K) for one figure level: K = level with k = 1 (Figure 2), or
    k_i = pi_i + epsilon and K = level/101 * sum k (Figure 3, equation (22))."""
    if figure == "2":
        return np.ones_like(mu), float(level)
    k = gallego_moon_profits(mu, sigma, price, cost, salvage) + FIXED_COST_EPSILON
    return k, level / 101.0 * float(np.sum(k))


def solve_uniform_cuts(
    a: np.ndarray,
    b: np.ndarray,
    x_u: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    capacity: float,
    fixed_costs: np.ndarray,
    budget: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Solve the uniform model (12) with the Lagrangian (dual) cut method."""
    import submodular_cuts

    quantities = np.zeros(len(a), dtype=float)
    active = x_u > 0.0
    status, solve_time = "optimal", 0.0
    if np.any(active):
        result = submodular_cuts.solve_uniform_fixed_cost_submodular_cuts(
            a[active],
            b[active],
            price[active],
            cost[active],
            salvage[active],
            x_u[active],
            fixed_costs[active],
            capacity,
            budget,
            time_limit_seconds=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            verbose=args.verbose_solver,
        )
        require(
            result.status in ("optimal", "time_limit") and result.quantities is not None,
            f"uniform cut solve ended with status {result.status}",
        )
        status, solve_time = result.status, result.solve_seconds
        quantities[active] = result.quantities
    return {"status": status, "x": quantities, "solve_time": solve_time}


def run_scheme(
    figure: str,
    scheme: str,
    data: f1.RetailData,
    levels: Sequence[int],
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    train_size, initial_window, _title = f1.SCHEME_SPECS[scheme]
    scheme_directory = output_root / scheme
    records_csv = scheme_directory / "solutions.csv"
    summary_csv = scheme_directory / "summary.csv"
    training_ends = f1.selected_training_ends(initial_window, data.week_count, args.max_windows)

    price = data.unit_prices
    cost, salvage = f1.economic_parameters(price, args.cost_model, args.cost_seed)
    capacity = f1.CAPACITY_FRACTION * float(np.sum(np.mean(data.demand, axis=1)))
    done = f1.load_csv_records(records_csv, ("level", "train_end_week", "model")) if args.resume else {}

    for train_end in training_ends:
        train_begin = max(1, train_end - train_size + 1)
        demand_train = data.demand[:, train_begin - 1 : train_end]
        demand_test = data.demand[:, train_end]
        b_max, mu, sigma = f1.training_parameters(demand_train)
        a, b, uniform_x_u = uniform_inputs(demand_train, price, cost, salvage)
        weights = f1.variance_weights(mu, sigma, price, salvage, "portfolio")
        df_x_u = f1.distribution_free_unconstrained_quantities(
            mu, sigma, args.lambda_, weights, price, cost, salvage
        )["x"]

        for level in levels:
            fixed_costs, budget = fixed_cost_data(figure, level, mu, sigma, price, cost, salvage)
            for model in MODELS:
                key = (str(level), str(train_end), model)
                if key in done:
                    continue
                print(f"[figure {figure} {scheme}] level={level} train_end={train_end} {model}", flush=True)
                if model == "uniform":
                    solved = solve_uniform_cuts(
                        a, b, uniform_x_u, price, cost, salvage, capacity, fixed_costs, budget, args
                    )
                elif model == "data_driven":
                    solved = f1.solve_data_driven(
                        demand_train, b_max, price, cost, salvage, capacity, args,
                        fixed_costs=fixed_costs, budget=budget,
                    )
                else:
                    solved = f1.solve_distribution_free_cuts(
                        mu, sigma, args.lambda_, weights, price, cost, salvage, capacity, args,
                        x_u=df_x_u, fixed_costs=fixed_costs, budget=budget,
                    )
                record = {
                    "scheme": scheme,
                    "level": level,
                    "train_begin_week": train_begin,
                    "train_end_week": train_end,
                    "test_week": train_end + 1,
                    "model": model,
                    "budget": budget,
                    "solver_status": solved["status"],
                    "solve_time_seconds": solved["solve_time"],
                    "selected_products": int(np.sum(solved["x"] > 0.0)),
                    "sum_x": float(np.sum(solved["x"])),
                    "test_loss": f1.loss_function(demand_test, solved["x"], price, cost, salvage),
                }
                f1.append_csv_record(records_csv, RECORD_FIELDS, record)
                done[key] = {field: str(record[field]) for field in RECORD_FIELDS}

    summaries = []
    for level in levels:
        statistics = {}
        for model in MODELS:
            losses = [float(done[(str(level), str(end), model)]["test_loss"]) for end in training_ends]
            statistics[model] = f1.padded_statistics(losses, data.week_count)
        summaries.append(
            {
                "Scheme": scheme,
                "Level": level,
                "Uniform": statistics["uniform"][0],
                "Uniform_Std": statistics["uniform"][1],
                "Data_driven": statistics["data_driven"][0],
                "DD_Std": statistics["data_driven"][1],
                "Distribution_free": statistics["distribution_free"][0],
                "DF_Std": statistics["distribution_free"][1],
                "Solved_Windows": len(training_ends),
            }
        )
    f1.write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    return summary_csv


def read_summary(path: Path) -> list[dict[str, float]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "Level": float(row["Level"]),
                "uniform": float(row["Uniform"]),
                "data_driven": float(row["Data_driven"]),
                "distribution_free": float(row["Distribution_free"]),
            }
            for row in csv.DictReader(handle)
        ]


def plot_figure(
    figure: str, summaries: Mapping[str, Sequence[Mapping[str, float]]], output_root: Path
) -> None:
    _, plt = f1.require_plot_dependencies()
    schemes = [name for name in f1.SCHEME_SPECS if name in summaries]
    require(schemes, "there are no summaries to plot")
    fig, axes = plt.subplots(1, len(schemes), figsize=(11.5, 3.25), squeeze=False)
    for axis, scheme in zip(axes[0], schemes):
        rows = sorted(summaries[scheme], key=lambda row: row["Level"])
        levels = [row["Level"] for row in rows]
        for model in MODELS:
            axis.plot(
                levels,
                [row[model] for row in rows],
                color=MODEL_COLORS[model],
                marker="o",
                markersize=2.7,
                linewidth=1.25,
                label=MODEL_LABELS[model],
            )
        axis.set_xlabel(FIGURE_SPECS[figure]["xlabel"], labelpad=0)
        axis.set_title(SCHEME_TITLES[scheme], y=-0.34, fontsize=10)
        axis.grid(True, which="major", alpha=0.18, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=7)
        axis.legend(loc="upper right", fontsize=6.5, frameon=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.98, bottom=0.31, wspace=0.24)
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / f"figure{figure}_rebuilt.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"figure{figure}_rebuilt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_levels(text: str, figure: str) -> tuple[int, ...]:
    if text.strip().lower() == "all":
        return tuple(FIGURE_SPECS[figure]["levels"])
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise f1.Figure1Error(f"invalid level selection: {text!r}") from error
    require(values and all(value > 0 for value in values), "levels must be positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", choices=tuple(FIGURE_SPECS), required=True)
    parser.add_argument("--input", type=Path, default=f1.default_retail_workbook())
    parser.add_argument("--output-dir", type=Path, help="default: figure<N>_<cost-model>_results next to this script")
    parser.add_argument("--schemes", default="all", help="all or a comma list from: increasing,20,10")
    parser.add_argument("--levels", default="all", help="all or a comma list of K (Figure 2) or kappa (Figure 3)")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--cost-model", choices=f1.COST_MODELS, default="random")
    parser.add_argument("--cost-seed", type=int, default=f1.DEFAULT_COST_SEED)
    parser.add_argument("--mip-gap", type=float, default=f1.MIP_GAP)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-windows", type=int, help="limit windows per panel for verification runs")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.lambda_ >= 0.0, "lambda must be nonnegative")
    require(args.mip_gap >= 0.0, "mip gap must be nonnegative")
    require(args.time_limit > 0.0, "time limit must be positive")
    require(args.threads >= 1, "threads must be positive")
    if args.max_windows is not None:
        require(args.max_windows >= 1, "max-windows must be positive")

    figure = args.figure
    schemes = f1.parse_schemes(args.schemes)
    levels = parse_levels(args.levels, figure)
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / f"figure{figure}_{args.cost_model}_results"
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_settings = {
        "figure": figure,
        "lambda": args.lambda_,
        "variance_bound": "portfolio",
        "cost_model": args.cost_model,
        "cost_seed": args.cost_seed if args.cost_model == "random" else None,
    }
    metadata_path = output_root / "run_metadata.json"
    if args.resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_settings = {key: previous.get(key) for key in run_settings}
        require(
            previous_settings == run_settings,
            f"{output_root} holds results for {previous_settings}; use another --output-dir or --no-resume",
        )

    data = f1.load_retail_data(args.input.resolve())
    print(f"Loaded {data.product_count} products and {data.week_count} weeks from {args.input.resolve()}")
    metadata = {
        **run_settings,
        "input": str(args.input.resolve()),
        "product_count": data.product_count,
        "week_count": data.week_count,
        "schemes": list(schemes),
        "levels": list(levels),
        "mip_gap": args.mip_gap,
        "time_limit_seconds": args.time_limit,
        "threads": args.threads,
        "capacity_fraction": f1.CAPACITY_FRACTION,
        "fixed_cost_epsilon": FIXED_COST_EPSILON,
        "uniform_widening": UNIFORM_WIDENING,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    summary_paths = {
        scheme: run_scheme(figure, scheme, data, levels, output_root, args) for scheme in schemes
    }
    if not args.no_plot:
        plot_figure(figure, {scheme: read_summary(path) for scheme, path in summary_paths.items()}, output_root)
    print(f"Results saved under {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except f1.Figure1Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
