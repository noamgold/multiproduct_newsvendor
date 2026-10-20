"""Rebuild Figure 1 in Section 9.2 of ``NewsvendorR1.pdf``.

This is a Python translation of the parts of
``Loss Function - sens_ anal_lambda_T_train_k=1.jl`` used by Figure 1.
The paper plots the data-driven and distribution-free models only; the Julia
driver disables its triangular solve, and the uniform results are not displayed
in Figure 1.  Accordingly, this script solves exactly the two displayed models.

The three panels use:

* an incrementally increasing training window, beginning with 10 weeks;
* a rolling 20-week training window; and
* a rolling 10-week training window.

For every distribution-free solve, the script stores scalar CSV summaries only:

* ``sum_theta = sum(theta)``;
* ``lambda_coefficient = sum(((p-s)*sigma)**2 * u)``; and
* the objective decomposition
  ``sum_theta + lambda * lambda_coefficient``.

The full ``x``, ``theta``, ``u``, ``z``, and ``x_u`` vectors exist only in
memory while a solve is processed and are never written to disk.

The experiment is intentionally checkpointed because the full reproduction
requires thousands of conic optimization solves. Re-running the command
resumes from completed scalar CSV records.

Typical commands
----------------

Full reproduction from Processed_Online_Retail.xlsx located next to this file
or in the sibling manuscript folder::

    python rebuild_figure1.py

Short verification run (one panel, one lambda, one rolling window)::

    python rebuild_figure1.py --schemes increasing --lambdas 5e-10 --max-windows 1

Recreate the plot directly from the original Julia summary workbooks::

    python rebuild_figure1.py --plot-only-from "C:\\path\\to\\Loss Functions - comparison"

Required packages are NumPy, gurobipy, openpyxl, matplotlib, and a licensed
GUROBI installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LAMBDA_VALUES = (
    1e-13,
    2.5e-13,
    5e-13,
    7.5e-13,
    1e-12,
    2.5e-12,
    5e-12,
    7.5e-12,
    1e-11,
    2.5e-11,
    5e-11,
    7.5e-11,
    1e-10,
    2.5e-10,
    5e-10,
    7.5e-10,
    1e-9,
    2.5e-9,
    5e-9,
    7.5e-9,
    1e-8,
    2.5e-8,
    5e-8,
    7.5e-8,
    1e-7,
    2.5e-7,
    5e-7,
    7.5e-7,
    1e-6,
)

MIP_GAP = 0.0001
TIME_LIMIT_SECONDS = 3.0 * 3600.0
CAPACITY_FRACTION = 0.1
FIXED_COST = 1.0
FIXED_COST_BUDGET = 40.0

SCHEME_SPECS: Mapping[str, tuple[int, int, str]] = {
    "increasing": (10**10, 10, "(a) Using an increasing training set size"),
    "20": (20, 20, r"(b) $|T_{\mathrm{train}}|=20$"),
    "10": (10, 10, r"(c) $|T_{\mathrm{train}}|=10$"),
}


def default_retail_workbook() -> Path:
    """Find the processed retail workbook in the code or manuscript folder."""
    script_dir = Path(__file__).resolve().parent
    candidates = (
        script_dir / "Processed_Online_Retail.xlsx",
        script_dir.parent / "data" / "Processed_Online_Retail.xlsx",
        script_dir.parent / "Newsvendor Noam New 23_Sep" / "Processed_Online_Retail.xlsx",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])

ORIGINAL_SUMMARY_NAMES: Mapping[str, str] = {
    "increasing": "Loss_Func_Lambda_k=1_T=inc_K=40_gap=0.0001_TL=3.xlsx",
    "20": "Loss_Func_Lambda_k=1_T=20_K=40_gap=0.0001_TL=3.xlsx",
    "10": "Loss_Func_Lambda_k=1_T=10_K=40_gap=0.0001_TL=3.xlsx",
}

DF_RECORD_FIELDS = (
    "scheme",
    "lambda",
    "train_begin_week",
    "train_end_week",
    "test_week",
    "solver_status",
    "unconstrained_solver_status",
    "solve_time_seconds",
    "unconstrained_solve_time_seconds",
    "sum_theta",
    "lambda_coefficient",
    "lambda_penalty",
    "objective_reconstructed",
    "objective_solver",
    "test_loss",
)

DD_RECORD_FIELDS = (
    "scheme",
    "train_begin_week",
    "train_end_week",
    "test_week",
    "solver_status",
    "solve_time_seconds",
    "selected_products",
    "sum_x",
    "objective_solver",
    "test_loss",
)

LAMBDA_SUMMARY_FIELDS = (
    "Scheme",
    "Lambda",
    "Data_driven",
    "DD_Std",
    "Distribution_free",
    "DF_Std",
    "Solved_Windows",
    "Mean_Sum_Theta",
    "Mean_Lambda_Coefficient",
    "Mean_Lambda_Penalty",
    "Mean_Objective",
)


class Figure1Error(RuntimeError):
    """Raised when the Figure 1 reproduction cannot be validated."""


@dataclass(frozen=True)
class RetailData:
    stock_codes: tuple[str, ...]
    unit_prices: np.ndarray
    demand: np.ndarray  # products x weeks

    @property
    def product_count(self) -> int:
        return len(self.stock_codes)

    @property
    def week_count(self) -> int:
        return int(self.demand.shape[1])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Figure1Error(message)


def require_plot_dependencies() -> tuple[Any, Any]:
    try:
        from openpyxl import load_workbook
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Figure1Error(
            "Plotting and XLSX input require openpyxl and matplotlib. "
            "Install them with: python -m pip install openpyxl matplotlib"
        ) from error
    return load_workbook, plt


def load_retail_data(path: Path) -> RetailData:
    """Reproduce the ordered StockCode construction in the Julia driver."""
    load_workbook, _ = require_plot_dependencies()
    require(path.is_file(), f"input workbook does not exist: {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    require("Sheet1" in workbook.sheetnames, f"{path} has no Sheet1")
    rows = workbook["Sheet1"].iter_rows(values_only=True)
    try:
        header = tuple(next(rows))
    except StopIteration as error:
        raise Figure1Error(f"{path} is empty") from error
    expected = ("WeekNo", "StockCode", "UnitPrice", "WeekQuantity")
    require(header[:4] == expected, f"unexpected {path.name} header: {header!r}")

    observations: list[tuple[int, str, float, float]] = []
    prices_by_stock: OrderedDict[str, float] = OrderedDict()
    maximum_week = 0
    for row_number, row in enumerate(rows, start=2):
        if not row or all(value is None for value in row[:4]):
            continue
        try:
            week = int(row[0])
            stock_code = str(row[1])
            unit_price = float(row[2])
            quantity = float(row[3])
        except (TypeError, ValueError) as error:
            raise Figure1Error(f"invalid row {row_number} in {path}: {row!r}") from error
        require(week >= 1, f"invalid week {week} at row {row_number}")
        if stock_code in prices_by_stock:
            require(
                math.isclose(prices_by_stock[stock_code], unit_price, rel_tol=0.0, abs_tol=1e-12),
                "The Julia source assumes one UnitPrice per StockCode, but "
                f"{stock_code!r} has multiple prices.",
            )
        else:
            prices_by_stock[stock_code] = unit_price
        observations.append((week, stock_code, unit_price, quantity))
        maximum_week = max(maximum_week, week)

    require(observations, f"{path} has no data rows")
    stock_codes = tuple(prices_by_stock)
    product_index = {stock_code: index for index, stock_code in enumerate(stock_codes)}
    demand = np.zeros((len(stock_codes), maximum_week), dtype=float)
    for week, stock_code, _unit_price, quantity in observations:
        demand[product_index[stock_code], week - 1] += quantity
    prices = np.asarray([prices_by_stock[code] for code in stock_codes], dtype=float)
    require(np.all(np.isfinite(demand)), "demand matrix contains non-finite values")
    require(np.all(prices > 0.0), "unit prices must be positive")
    return RetailData(stock_codes=stock_codes, unit_prices=prices, demand=demand)


def training_parameters(demand_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return b, mu, sigma using Julia's maximum/mean/sample-std definitions."""
    require(demand_train.ndim == 2 and demand_train.shape[1] >= 2, "training window is too short")
    b = np.max(demand_train, axis=1)
    mu = np.mean(demand_train, axis=1)
    sigma = np.std(demand_train, axis=1, ddof=1)
    return b, mu, sigma


def loss_function(
    demand: np.ndarray,
    x: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> float:
    """Equation (21), matching Compute_Loss_Function in the Julia source."""
    underage = np.maximum(0.0, demand - x)
    overage = np.maximum(0.0, x - demand)
    return float((price - cost) @ underage + (cost - salvage) @ overage)


def solve_data_driven(
    demand_train: np.ndarray,
    b: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    capacity: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Solve empirical formulation (20), translated from mpnv_DD.jl."""
    import gurobipy as gp

    n, periods = demand_train.shape
    model = gp.Model("figure1_data_driven")
    model.Params.OutputFlag = 1 if args.verbose_solver else 0
    model.Params.MIPGap = args.mip_gap
    model.Params.TimeLimit = args.time_limit
    if args.threads is not None:
        model.Params.Threads = args.threads
    x = model.addMVar(n, lb=0.0, name="x")
    underage = model.addMVar((n, periods), lb=0.0, name="underage")
    overage = model.addMVar((n, periods), lb=0.0, name="overage")
    z = model.addMVar(n, vtype=gp.GRB.BINARY, name="z")
    model.setObjective(
        gp.quicksum(
            (price[index] - cost[index]) * underage[index, period]
            + (cost[index] - salvage[index]) * overage[index, period]
            for index in range(n)
            for period in range(periods)
        ),
        gp.GRB.MINIMIZE,
    )
    model.addConstr(x.sum() <= capacity, name="capacity")
    model.addConstr(FIXED_COST * z.sum() <= FIXED_COST_BUDGET, name="fixed_cost_budget")
    model.addConstr(x <= b * z, name="selection_link")
    for period in range(periods):
        model.addConstr(underage[:, period] >= demand_train[:, period] - x)
        model.addConstr(overage[:, period] >= x - demand_train[:, period])
    model.optimize()
    require(model.Status == gp.GRB.OPTIMAL, f"data-driven solve ended with status {model.Status}")
    return {
        "status": "optimal",
        "x": np.asarray(x.X, dtype=float),
        "z": np.asarray(z.X, dtype=float),
        "objective": float(model.ObjVal),
        "solve_time": float(model.Runtime),
    }


def solve_distribution_free(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_value: float,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    capacity: float,
    args: argparse.Namespace,
    *,
    x_u: np.ndarray,
) -> dict[str, Any]:
    """Solve the fixed-cost version of formulation (17) with Gurobi."""
    import gurobipy as gp

    n = len(mu)
    model = gp.Model("figure1_distribution_free")
    model.Params.OutputFlag = 1 if args.verbose_solver else 0
    model.Params.MIPGap = args.mip_gap
    model.Params.TimeLimit = args.time_limit
    if args.threads is not None:
        model.Params.Threads = args.threads
    x = model.addMVar(n, lb=0.0, name="x")
    theta = model.addMVar(n, lb=-gp.GRB.INFINITY, name="theta")
    radius = model.addMVar(n, lb=0.0, name="radius")
    u = model.addMVar(n, lb=0.0, name="u")
    z = model.addMVar(n, vtype=gp.GRB.BINARY, name="z")
    model.addConstr(x <= x_u * z, name="selection_link")
    model.addConstr(FIXED_COST * z.sum() <= FIXED_COST_BUDGET, name="fixed_cost_budget")

    spread = price - salvage
    lambda_weights = (spread * sigma) ** 2
    model.setObjective(theta.sum() + lambda_value * (lambda_weights @ u), gp.GRB.MINIMIZE)
    model.addConstr(x.sum() <= capacity, name="capacity")
    model.addConstr(
        (cost - salvage) * x + spread * (radius - x + mu) / 2.0 <= theta,
        name="theta_epigraph",
    )
    for index in range(n):
        x_i = x[index].item()
        u_i = u[index].item()
        z_i = z[index].item()
        radius_i = radius[index].item()
        model.addQConstr(x_i * x_i <= u_i * z_i)
        centered = x_i - float(mu[index])
        model.addQConstr(float(sigma[index]) ** 2 + centered * centered <= radius_i * radius_i)
    model.optimize()
    require(model.Status == gp.GRB.OPTIMAL, f"distribution-free solve ended with status {model.Status}")

    x_value = np.asarray(x.X, dtype=float)
    theta_value = np.asarray(theta.X, dtype=float)
    u_value = np.asarray(u.X, dtype=float)
    z_value = np.asarray(z.X, dtype=float)
    sum_theta = float(np.sum(theta_value))
    lambda_coefficient = float(lambda_weights @ u_value)
    reconstructed = sum_theta + lambda_value * lambda_coefficient
    tolerance = 1e-5 * max(1.0, abs(float(model.ObjVal)), abs(reconstructed))
    require(
        abs(float(model.ObjVal) - reconstructed) <= tolerance,
        "distribution-free objective decomposition failed: "
        f"solver={model.ObjVal}, reconstructed={reconstructed}",
    )
    return {
        "status": "optimal",
        "x": x_value,
        "theta": theta_value,
        "u": u_value,
        "z": z_value,
        "sum_theta": sum_theta,
        "lambda_coefficient": lambda_coefficient,
        "objective": float(model.ObjVal),
        "solve_time": float(model.Runtime),
    }


def distribution_free_unconstrained_quantities(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_value: float,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> dict[str, Any]:
    """Compute x^U from the exact separable first-order condition.

    The Julia code obtains the same bound by solving a continuous conic model
    with capacity ``sum(mu + 3*sigma)``. That capacity is checked below and is
    nonbinding for these data. Bisection avoids Gurobi barrier status 13 caused
    by objective coefficients as small as 1e-13.
    """
    started = time.perf_counter()
    spread = price - salvage
    lambda_weights = (spread * sigma) ** 2
    quantities = np.zeros_like(mu, dtype=float)

    for index in range(len(mu)):
        mu_i = float(mu[index])
        sigma_i = float(sigma[index])
        if sigma_i <= 1e-14:
            # With zero variance the ambiguity-set loss has its kink/minimum
            # at deterministic demand mu_i; the lambda weight is also zero.
            quantities[index] = max(0.0, mu_i)
            continue

        p_i = float(price[index])
        c_i = float(cost[index])
        s_i = float(salvage[index])
        weight_i = float(lambda_weights[index])

        def derivative(x_value: float) -> float:
            centered = x_value - mu_i
            radius = math.sqrt(sigma_i * sigma_i + centered * centered)
            return (
                c_i
                - s_i
                + (p_i - s_i) * (centered / radius - 1.0) / 2.0
                + 2.0 * lambda_value * weight_i * x_value
            )

        if derivative(0.0) >= 0.0:
            quantities[index] = 0.0
            continue
        lower = 0.0
        upper = max(1.0, mu_i + 3.0 * sigma_i)
        while derivative(upper) < 0.0:
            upper *= 2.0
            require(upper < 1e15, "could not bracket a distribution-free unconstrained minimizer")
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            if derivative(midpoint) < 0.0:
                lower = midpoint
            else:
                upper = midpoint
        quantities[index] = (lower + upper) / 2.0

    capacity = float(np.sum(mu + 3.0 * sigma))
    require(
        float(np.sum(quantities)) <= capacity + 1e-8 * max(1.0, capacity),
        "the Julia unconstrained-bound capacity is binding; separable x^U is not valid",
    )
    return {
        "status": "analytic_optimal",
        "x": quantities,
        "solve_time": time.perf_counter() - started,
    }


def lambda_text(value: float) -> str:
    return format(value, ".17g")


def load_csv_records(path: Path, key_fields: Sequence[str]) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            tuple(row[field] for field in key_fields): row
            for row in csv.DictReader(handle)
        }


def append_csv_record(path: Path, fields: Sequence[str], record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({field: record[field] for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: Sequence[str], records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
    os.replace(temporary, path)


def selected_training_ends(initial_window: int, week_count: int, max_windows: int | None) -> tuple[int, ...]:
    values = tuple(range(initial_window, week_count))
    if max_windows is not None:
        values = values[:max_windows]
    require(values, "no rolling test periods were selected")
    return values


def run_scheme(
    scheme: str,
    data: RetailData,
    lambda_values: Sequence[float],
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    train_size, initial_window, _title = SCHEME_SPECS[scheme]
    scheme_directory = output_root / scheme
    df_csv = scheme_directory / "distribution_free_solutions.csv"
    dd_csv = scheme_directory / "data_driven_solutions.csv"
    summary_csv = scheme_directory / "lambda_summary.csv"
    training_ends = selected_training_ends(initial_window, data.week_count, args.max_windows)

    price = data.unit_prices
    cost = price / 1.9
    salvage = cost * 0.4
    capacity = CAPACITY_FRACTION * float(np.sum(np.mean(data.demand, axis=1)))

    df_done = load_csv_records(df_csv, ("lambda", "train_end_week")) if args.resume else {}
    dd_done = load_csv_records(dd_csv, ("train_end_week",)) if args.resume else {}

    for train_end in training_ends:
        train_begin = max(1, train_end - train_size + 1)
        demand_train = data.demand[:, train_begin - 1 : train_end]
        demand_test = data.demand[:, train_end]
        b, mu, sigma = training_parameters(demand_train)

        dd_key = (str(train_end),)
        if dd_key not in dd_done:
            print(f"[{scheme}] data-driven train_end={train_end}", flush=True)
            solved_dd = solve_data_driven(demand_train, b, price, cost, salvage, capacity, args)
            dd_record = {
                "scheme": scheme,
                "train_begin_week": train_begin,
                "train_end_week": train_end,
                "test_week": train_end + 1,
                "solver_status": solved_dd["status"],
                "solve_time_seconds": solved_dd["solve_time"],
                "selected_products": int(np.rint(solved_dd["z"]).sum()),
                "sum_x": float(np.sum(solved_dd["x"])),
                "objective_solver": solved_dd["objective"],
                "test_loss": loss_function(demand_test, solved_dd["x"], price, cost, salvage),
            }
            append_csv_record(dd_csv, DD_RECORD_FIELDS, dd_record)
            dd_done[dd_key] = {key: str(value) for key, value in dd_record.items()}

        for lambda_value in lambda_values:
            key = (lambda_text(lambda_value), str(train_end))
            existing = df_done.get(key)
            if existing is not None:
                continue
            print(f"[{scheme}] lambda={lambda_value:.4g} train_end={train_end}", flush=True)
            unconstrained = distribution_free_unconstrained_quantities(
                mu,
                sigma,
                lambda_value,
                price,
                cost,
                salvage,
            )
            x_u = unconstrained["x"]
            solved = solve_distribution_free(
                mu,
                sigma,
                lambda_value,
                price,
                cost,
                salvage,
                capacity,
                args,
                x_u=x_u,
            )
            record = {
                "scheme": scheme,
                "lambda": lambda_text(lambda_value),
                "train_begin_week": train_begin,
                "train_end_week": train_end,
                "test_week": train_end + 1,
                "solver_status": solved["status"],
                "unconstrained_solver_status": unconstrained["status"],
                "solve_time_seconds": solved["solve_time"],
                "unconstrained_solve_time_seconds": unconstrained["solve_time"],
                "sum_theta": solved["sum_theta"],
                "lambda_coefficient": solved["lambda_coefficient"],
                "lambda_penalty": lambda_value * solved["lambda_coefficient"],
                "objective_reconstructed": (
                    solved["sum_theta"] + lambda_value * solved["lambda_coefficient"]
                ),
                "objective_solver": solved["objective"],
                "test_loss": loss_function(demand_test, solved["x"], price, cost, salvage),
            }
            append_csv_record(df_csv, DF_RECORD_FIELDS, record)
            df_done[key] = {field: str(record[field]) for field in DF_RECORD_FIELDS}

    aggregate_scheme(
        scheme,
        lambda_values,
        data.week_count,
        training_ends,
        df_csv,
        dd_csv,
        summary_csv,
    )
    return summary_csv


def padded_statistics(values: Sequence[float], total_periods: int) -> tuple[float, float]:
    """Match Julia's mean/std over WeekNum-length arrays with leading zeros."""
    require(len(values) <= total_periods, "more observations than periods")
    padded = np.zeros(total_periods, dtype=float)
    padded[: len(values)] = np.asarray(values, dtype=float)
    return float(np.mean(padded)), float(np.std(padded, ddof=1))


def aggregate_scheme(
    scheme: str,
    lambda_values: Sequence[float],
    week_count: int,
    training_ends: Sequence[int],
    df_csv: Path,
    dd_csv: Path,
    output_csv: Path,
) -> None:
    df_rows = load_csv_records(df_csv, ("lambda", "train_end_week"))
    dd_rows = load_csv_records(dd_csv, ("train_end_week",))
    dd_losses = [float(dd_rows[(str(end),)]["test_loss"]) for end in training_ends]
    dd_mean, dd_std = padded_statistics(dd_losses, week_count)
    summaries: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        rows = [df_rows[(lambda_text(lambda_value), str(end))] for end in training_ends]
        df_mean, df_std = padded_statistics([float(row["test_loss"]) for row in rows], week_count)
        summaries.append(
            {
                "Scheme": scheme,
                "Lambda": lambda_text(lambda_value),
                "Data_driven": dd_mean,
                "DD_Std": dd_std,
                "Distribution_free": df_mean,
                "DF_Std": df_std,
                "Solved_Windows": len(rows),
                "Mean_Sum_Theta": np.mean([float(row["sum_theta"]) for row in rows]),
                "Mean_Lambda_Coefficient": np.mean(
                    [float(row["lambda_coefficient"]) for row in rows]
                ),
                "Mean_Lambda_Penalty": np.mean([float(row["lambda_penalty"]) for row in rows]),
                "Mean_Objective": np.mean([float(row["objective_reconstructed"]) for row in rows]),
            }
        )
    write_csv(output_csv, LAMBDA_SUMMARY_FIELDS, summaries)


def read_csv_summary(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "Lambda": float(row["Lambda"]),
                "Data_driven": float(row["Data_driven"]),
                "Distribution_free": float(row["Distribution_free"]),
            }
            for row in csv.DictReader(handle)
        ]


def read_original_summaries(directory: Path) -> dict[str, list[dict[str, float]]]:
    load_workbook, _ = require_plot_dependencies()
    summaries: dict[str, list[dict[str, float]]] = {}
    for scheme, filename in ORIGINAL_SUMMARY_NAMES.items():
        path = directory / filename
        require(path.is_file(), f"missing original summary workbook: {path}")
        worksheet = load_workbook(path, read_only=True, data_only=True)["Sheet1"]
        rows = worksheet.iter_rows(values_only=True)
        header = tuple(next(rows))
        indices = {name: header.index(name) for name in ("Lambda", "Data_driven", "Distribution_free")}
        summaries[scheme] = [
            {
                name: float(row[index])
                for name, index in indices.items()
            }
            for row in rows
            if row[indices["Lambda"]] is not None
        ]
    return summaries


def plot_figure(summaries: Mapping[str, Sequence[Mapping[str, float]]], output_root: Path) -> None:
    _, plt = require_plot_dependencies()
    schemes = [name for name in SCHEME_SPECS if name in summaries]
    require(schemes, "there are no summaries to plot")
    figure, axes = plt.subplots(1, len(schemes), figsize=(11.5, 3.25), squeeze=False)
    for axis, scheme in zip(axes[0], schemes):
        rows = sorted(summaries[scheme], key=lambda row: row["Lambda"])
        lambdas = [row["Lambda"] for row in rows]
        data_driven = [row["Data_driven"] for row in rows]
        distribution_free = [row["Distribution_free"] for row in rows]
        axis.plot(
            lambdas,
            data_driven,
            color="#1f77b4",
            marker="o",
            markersize=2.7,
            linewidth=1.25,
            label="Data driven",
        )
        axis.plot(
            lambdas,
            distribution_free,
            color="#e76f51",
            marker="o",
            markersize=2.7,
            linewidth=1.25,
            label="Distribution free",
        )
        axis.set_xscale("log")
        axis.set_xlim(1e-13, 1e-6)
        axis.set_xlabel(r"$\lambda$", labelpad=0)
        axis.set_title(SCHEME_SPECS[scheme][2], y=-0.34, fontsize=10)
        axis.grid(True, which="major", alpha=0.18, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=7)
        axis.legend(loc="upper right", fontsize=6.5, frameon=True)
    figure.subplots_adjust(left=0.065, right=0.995, top=0.98, bottom=0.31, wspace=0.24)
    output_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_root / "figure1_rebuilt.pdf", bbox_inches="tight")
    figure.savefig(output_root / "figure1_rebuilt.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_schemes(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "all":
        return tuple(SCHEME_SPECS)
    values = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    require(values and all(value in SCHEME_SPECS for value in values), "schemes must be all, increasing, 20, or 10")
    return values


def parse_lambdas(text: str) -> tuple[float, ...]:
    if text.strip().lower() == "all":
        return LAMBDA_VALUES
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise Figure1Error(f"invalid lambda selection: {text!r}") from error
    require(values and all(value > 0.0 for value in values), "lambda values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_retail_workbook())
    parser.add_argument("--output-dir", type=Path, default=script_directory / "figure1_python_results")
    parser.add_argument("--plot-only-from", type=Path, help="directory containing the three original Julia summary XLSX files")
    parser.add_argument("--schemes", default="all", help="all or a comma list from: increasing,20,10")
    parser.add_argument("--lambdas", default="all", help="all or a comma-separated list")
    parser.add_argument("--solver", default="GUROBI")
    parser.add_argument("--mip-gap", type=float, default=MIP_GAP)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--threads", type=int, help="omit to retain the Julia/Gurobi automatic thread setting")
    parser.add_argument("--max-windows", type=int, help="limit windows per panel for verification runs")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.mip_gap >= 0.0, "mip gap must be nonnegative")
    require(args.time_limit > 0.0, "time limit must be positive")
    if args.threads is not None:
        require(args.threads >= 1, "threads must be positive")
    if args.max_windows is not None:
        require(args.max_windows >= 1, "max-windows must be positive")

    if args.plot_only_from is not None:
        plot_figure(read_original_summaries(args.plot_only_from.resolve()), args.output_dir.resolve())
        print(f"Wrote {args.output_dir.resolve() / 'figure1_rebuilt.pdf'}")
        return 0

    schemes = parse_schemes(args.schemes)
    lambda_values = parse_lambdas(args.lambdas)
    require(args.solver.upper() == "GUROBI", "this Julia-equivalent implementation requires GUROBI")
    try:
        import gurobipy  # noqa: F401
    except ImportError as error:
        raise Figure1Error("gurobipy is not installed") from error
    data = load_retail_data(args.input.resolve())
    print(f"Loaded {data.product_count} products and {data.week_count} weeks from {args.input.resolve()}")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_julia": "Loss Function - sens_ anal_lambda_T_train_k=1.jl",
        "input": str(args.input.resolve()),
        "product_count": data.product_count,
        "week_count": data.week_count,
        "schemes": list(schemes),
        "lambda_values": list(lambda_values),
        "mip_gap": args.mip_gap,
        "time_limit_seconds": args.time_limit,
        "solver": args.solver,
        "threads": args.threads,
        "capacity_fraction": CAPACITY_FRACTION,
        "fixed_cost": FIXED_COST,
        "fixed_cost_budget": FIXED_COST_BUDGET,
        "julia_zero_padding_preserved": True,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    summary_paths = {
        scheme: run_scheme(scheme, data, lambda_values, output_root, args)
        for scheme in schemes
    }
    if not args.no_plot:
        summaries = {scheme: read_csv_summary(path) for scheme, path in summary_paths.items()}
        plot_figure(summaries, output_root)
    print(f"Results saved under {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Figure1Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
