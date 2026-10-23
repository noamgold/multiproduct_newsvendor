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

Variance penalty.  ``--variance-bound portfolio`` (the default) weights the
penalty by ``((p-s)*sigma/mu)**2``.  For nonnegative demand with mean mu > 0,
Var(min(D, x)) <= (sigma/mu)**2 * x**2, so with independent demands the
penalty sum_i ((p_i-s_i)*sigma_i/mu_i)**2 * x_i**2 is the variance of the
portfolio sum_i (p_i-s_i) x_i D_i/mu_i and an upper bound on the profit
variance.  ``--variance-bound sigma2`` restores the original weights
``((p-s)*sigma)**2`` of the submitted manuscript's Figure 1.

Costs and salvage values.  ``--cost-model fixed`` (the default) uses the
Julia driver's c_i = p_i/1.9 and s_i = 0.4 c_i.  ``--cost-model random``
instead draws c_i = p_i * U[10/14, 10/11] and s_i = c_i * U[-0.9, 0.9] once per
product from ``--cost-seed``, shared by every scheme, window and model.

Distribution-free solves use the Lagrangian (dual) cut method of
submodular_cuts.py by default (``--df-method cuts``); ``--df-method misocp``
solves the perspective MISOCP (20) directly with Gurobi, which can stall on
numerical issues for intermediate lambda.

For every distribution-free solve, the script stores scalar CSV summaries only:

* ``sum_theta = sum(theta)``;
* ``lambda_coefficient = sum(weights * u)`` with the weights above; and
* the objective decomposition
  ``sum_theta + lambda * lambda_coefficient``.

Each panel also solves lambda = 0 (the Gallego-Moon model), drawn as a dashed
reference line, and records the data-dependent upper bound lambda_max of
Section 5.4 (with epsilon = C/n) for every training window; the median over
windows is drawn as a vertical line.

The full ``x``, ``theta``, ``u``, ``z``, and ``x_u`` vectors exist only in
memory while a solve is processed and are never written to disk.

The experiment is intentionally checkpointed because the full reproduction
requires thousands of conic optimization solves. Re-running the command
resumes from completed scalar CSV records.

Typical commands
----------------

Full run with the portfolio variance bound, from Processed_Online_Retail.xlsx
located next to this file or in the sibling data folder::

    python rebuild_figure1.py

Original (submitted) Figure 1 weights::

    python rebuild_figure1.py --variance-bound sigma2

Short verification run (one panel, one lambda, one rolling window)::

    python rebuild_figure1.py --schemes increasing --lambdas 1e-6 --max-windows 1

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


SIGMA2_LAMBDA_VALUES = (
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

# The portfolio weights are the sigma2 weights divided by mu**2 (median mu is
# about 20 units/week), and lambda_max is O(1) for these data, so the grid is
# shifted up and extended to cover [0, lambda_max].
PORTFOLIO_LAMBDA_VALUES = tuple(
    mantissa * 10.0**exponent
    for exponent in range(-9, 1)
    for mantissa in (1.0, 2.5, 5.0)
) + (10.0,)

VARIANCE_BOUNDS = ("portfolio", "sigma2")
DEFAULT_LAMBDA_VALUES: Mapping[str, tuple[float, ...]] = {
    "portfolio": PORTFOLIO_LAMBDA_VALUES,
    "sigma2": SIGMA2_LAMBDA_VALUES,
}

COST_MODELS = ("random", "fixed")
DEFAULT_COST_SEED = 20261022
COST_FACTOR_RANGE = (10.0 / 14.0, 10.0 / 11.0)
SALVAGE_FACTOR_RANGE = (-0.9, 0.9)


def default_output_directory(variance_bound: str, cost_model: str) -> str:
    if (variance_bound, cost_model) == ("sigma2", "fixed"):
        return "figure1_python_results"
    return f"figure1_{variance_bound}_{cost_model}_results"

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

WINDOW_RECORD_FIELDS = (
    "scheme",
    "train_begin_week",
    "train_end_week",
    "lambda_max",
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


def economic_parameters(
    price: np.ndarray, cost_model: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cost, salvage) per product.

    ``random`` (Section 6.2): c_i = p_i * U[10/14, 10/11], s_i = c_i * U[-0.9, 0.9],
    drawn once per product in product order from ``seed``.
    ``fixed``: the Julia driver's c_i = p_i / 1.9, s_i = 0.4 c_i.
    """
    if cost_model == "fixed":
        cost = price / 1.9
        return cost, cost * 0.4
    require(cost_model == "random", f"unknown cost model {cost_model!r}")
    rng = np.random.default_rng(seed)
    cost = price * rng.uniform(*COST_FACTOR_RANGE, size=price.shape)
    salvage = cost * rng.uniform(*SALVAGE_FACTOR_RANGE, size=price.shape)
    return cost, salvage


def variance_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    price: np.ndarray,
    salvage: np.ndarray,
    variance_bound: str,
) -> np.ndarray:
    """Penalty weights w_i of the term lambda * sum_i w_i * x_i**2.

    ``portfolio``: w_i = ((p_i-s_i)*sigma_i/mu_i)**2, the variance of the
    mean-normalized portfolio, which bounds the newsvendor profit variance.
    A product with mu_i = 0 never sold in the window (demand is nonnegative),
    so sigma_i = 0 and its weight is 0.
    ``sigma2``: w_i = ((p_i-s_i)*sigma_i)**2, the submitted manuscript's weights.
    """
    spread = price - salvage
    if variance_bound == "sigma2":
        return (spread * sigma) ** 2
    require(variance_bound == "portfolio", f"unknown variance bound {variance_bound!r}")
    require(bool(np.all(sigma[mu <= 0.0] <= 1e-14)), "positive sigma with nonpositive mean")
    safe_mu = np.where(mu > 0.0, mu, 1.0)
    return np.where(mu > 0.0, (spread * sigma / safe_mu) ** 2, 0.0)


def lambda_upper_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    lambda_weights: np.ndarray,
    epsilon: float,
) -> float:
    """lambda_max of Section 5.4: for lambda >= lambda_max every product's
    marginal cost at x = epsilon is nonnegative, so x* <= epsilon.

    Products with zero weight cannot be driven to zero by the penalty and are
    excluded, matching the restriction to sigma_i > 0 in the manuscript.
    """
    spread = price - salvage
    radius = np.sqrt(sigma**2 + (epsilon - mu) ** 2)
    radius = np.where(radius > 0.0, radius, 1.0)
    numerator = np.maximum(spread / 2.0 * (1.0 + (mu - epsilon) / radius) - (cost - salvage), 0.0)
    penalized = lambda_weights > 0.0
    if not np.any(penalized):
        return math.inf
    return float(np.max(numerator[penalized] / (2.0 * lambda_weights[penalized] * epsilon)))


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
    *,
    fixed_costs: np.ndarray | None = None,
    budget: float = FIXED_COST_BUDGET,
) -> dict[str, Any]:
    """Solve empirical formulation (23), translated from mpnv_DD.jl.

    ``fixed_costs`` defaults to k_i = FIXED_COST for every product.
    """
    import gurobipy as gp

    n, periods = demand_train.shape
    k = np.full(n, FIXED_COST) if fixed_costs is None else fixed_costs
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
    model.addConstr(k @ z <= budget, name="fixed_cost_budget")
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
    lambda_weights: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    capacity: float,
    args: argparse.Namespace,
    *,
    x_u: np.ndarray,
    fixed_costs: np.ndarray | None = None,
    budget: float = FIXED_COST_BUDGET,
) -> dict[str, Any]:
    """Solve the perspective formulation (20) with Gurobi."""
    import gurobipy as gp

    n = len(mu)
    k = np.full(n, FIXED_COST) if fixed_costs is None else fixed_costs
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
    model.addConstr(k @ z <= budget, name="fixed_cost_budget")

    spread = price - salvage
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


def distribution_free_theta(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
) -> np.ndarray:
    """theta_i(x_i) of (19b), the Gallego-Moon expected-cost bound."""
    radius = np.sqrt(sigma**2 + (x - mu) ** 2)
    return (cost - salvage) * x + (price - salvage) * (radius - x + mu) / 2.0


def solve_distribution_free_cuts(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_value: float,
    lambda_weights: np.ndarray,
    price: np.ndarray,
    cost: np.ndarray,
    salvage: np.ndarray,
    capacity: float,
    args: argparse.Namespace,
    *,
    x_u: np.ndarray,
    fixed_costs: np.ndarray | None = None,
    budget: float = FIXED_COST_BUDGET,
) -> dict[str, Any]:
    """Solve (19) with the Lagrangian (dual) cut method of submodular_cuts.

    Products with x^U_i = 0 never order anything and are left out of the cut
    master; in these data they are exactly the products with no sales in the
    training window (mu_i = sigma_i = 0), which the oracle cannot represent.
    The cut solver maximizes the profit sum_i g_i(0) - g_i(x_i); the cost
    objective of (19) is recovered by subtracting it from sum_i g_i(0).
    """
    import submodular_cuts

    n = len(mu)
    k = np.full(n, FIXED_COST) if fixed_costs is None else fixed_costs
    active = x_u > 0.0
    require(
        bool(np.all((sigma[active] > 0.0) & (mu[active] > 0.0))),
        "a product with positive x^U has zero mean or zero standard deviation",
    )
    quantities = np.zeros(n, dtype=float)
    status = "optimal"
    solve_time = 0.0
    profit = 0.0
    if np.any(active):
        result = submodular_cuts.solve_distribution_free_fixed_cost_submodular_cuts(
            mu[active],
            sigma[active],
            lambda_value,
            price[active],
            cost[active],
            salvage[active],
            x_u[active],
            k[active],
            capacity,
            budget,
            variance_weight=lambda_weights[active],
            time_limit_seconds=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads if args.threads is not None else 1,
            verbose=args.verbose_solver,
        )
        require(
            result.status in ("optimal", "time_limit") and result.quantities is not None,
            f"distribution-free cut solve ended with status {result.status}",
        )
        status = result.status
        solve_time = result.solve_seconds
        profit = float(result.objective)
        quantities[active] = result.quantities

    theta = distribution_free_theta(quantities, mu, sigma, price, cost, salvage)
    sum_theta = float(np.sum(theta))
    lambda_coefficient = float(lambda_weights @ quantities**2)
    zero_cost = float(np.sum(distribution_free_theta(np.zeros(n), mu, sigma, price, cost, salvage)))
    objective = zero_cost - profit
    reconstructed = sum_theta + lambda_value * lambda_coefficient
    require(
        abs(objective - reconstructed) <= 1e-6 * max(1.0, abs(objective)),
        f"cut-method objective decomposition failed: solver={objective}, reconstructed={reconstructed}",
    )
    return {
        "status": status,
        "x": quantities,
        "sum_theta": sum_theta,
        "lambda_coefficient": lambda_coefficient,
        "objective": objective,
        "solve_time": solve_time,
    }


def distribution_free_unconstrained_quantities(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_value: float,
    lambda_weights: np.ndarray,
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
    window_csv = scheme_directory / "windows.csv"
    summary_csv = scheme_directory / "lambda_summary.csv"
    training_ends = selected_training_ends(initial_window, data.week_count, args.max_windows)

    price = data.unit_prices
    cost, salvage = economic_parameters(price, args.cost_model, args.cost_seed)
    capacity = CAPACITY_FRACTION * float(np.sum(np.mean(data.demand, axis=1)))
    epsilon = capacity / data.product_count

    df_done = load_csv_records(df_csv, ("lambda", "train_end_week")) if args.resume else {}
    dd_done = load_csv_records(dd_csv, ("train_end_week",)) if args.resume else {}
    window_done = load_csv_records(window_csv, ("train_end_week",)) if args.resume else {}

    for train_end in training_ends:
        train_begin = max(1, train_end - train_size + 1)
        demand_train = data.demand[:, train_begin - 1 : train_end]
        demand_test = data.demand[:, train_end]
        b, mu, sigma = training_parameters(demand_train)
        lambda_weights = variance_weights(mu, sigma, price, salvage, args.variance_bound)

        if (str(train_end),) not in window_done:
            window_record = {
                "scheme": scheme,
                "train_begin_week": train_begin,
                "train_end_week": train_end,
                "lambda_max": lambda_upper_bound(
                    mu, sigma, price, cost, salvage, lambda_weights, epsilon
                ),
            }
            append_csv_record(window_csv, WINDOW_RECORD_FIELDS, window_record)
            window_done[(str(train_end),)] = {key: str(value) for key, value in window_record.items()}

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
                lambda_weights,
                price,
                cost,
                salvage,
            )
            x_u = unconstrained["x"]
            solve = solve_distribution_free_cuts if args.df_method == "cuts" else solve_distribution_free
            solved = solve(
                mu,
                sigma,
                lambda_value,
                lambda_weights,
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
    lambda_max_values = [float(window_done[(str(end),)]["lambda_max"]) for end in training_ends]
    write_csv(
        scheme_directory / "lambda_max_summary.csv",
        ("Scheme", "Median_Lambda_Max", "Min_Lambda_Max", "Max_Lambda_Max"),
        [
            {
                "Scheme": scheme,
                "Median_Lambda_Max": float(np.median(lambda_max_values)),
                "Min_Lambda_Max": min(lambda_max_values),
                "Max_Lambda_Max": max(lambda_max_values),
            }
        ],
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


def read_lambda_max(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[0]["Median_Lambda_Max"]) if rows else None


def plot_figure(
    summaries: Mapping[str, Sequence[Mapping[str, float]]],
    output_root: Path,
    lambda_max: Mapping[str, float | None] | None = None,
) -> None:
    _, plt = require_plot_dependencies()
    schemes = [name for name in SCHEME_SPECS if name in summaries]
    require(schemes, "there are no summaries to plot")
    figure, axes = plt.subplots(1, len(schemes), figsize=(11.5, 3.25), squeeze=False)
    for axis, scheme in zip(axes[0], schemes):
        rows = sorted(
            (row for row in summaries[scheme] if row["Lambda"] > 0.0),
            key=lambda row: row["Lambda"],
        )
        zero_rows = [row for row in summaries[scheme] if row["Lambda"] == 0.0]
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
            label="Data-driven",
        )
        axis.plot(
            lambdas,
            distribution_free,
            color="#e76f51",
            marker="o",
            markersize=2.7,
            linewidth=1.25,
            label="Distribution-free",
        )
        if zero_rows:
            axis.axhline(
                zero_rows[0]["Distribution_free"],
                color="#e76f51",
                linestyle="--",
                linewidth=0.9,
                label=r"Distribution-free, $\lambda=0$",
            )
        scheme_lambda_max = (lambda_max or {}).get(scheme)
        if scheme_lambda_max is not None and lambdas and scheme_lambda_max <= 10.0 * lambdas[-1]:
            axis.axvline(
                scheme_lambda_max,
                color="0.35",
                linestyle=":",
                linewidth=1.0,
                label=r"median $\lambda_{\max}$",
            )
        axis.set_xscale("log")
        axis.set_xlim(lambdas[0], lambdas[-1])
        axis.set_xlabel(r"$\lambda$", labelpad=0)
        axis.set_title(SCHEME_SPECS[scheme][2], y=-0.34, fontsize=10)
        axis.grid(True, which="major", alpha=0.18, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=7)
        axis.legend(loc="upper left", fontsize=6.5, frameon=True)
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


def parse_lambdas(text: str, variance_bound: str) -> tuple[float, ...]:
    if text.strip().lower() == "all":
        return DEFAULT_LAMBDA_VALUES[variance_bound]
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
    parser.add_argument(
        "--variance-bound",
        choices=VARIANCE_BOUNDS,
        default="portfolio",
        help="portfolio: weights ((p-s)*sigma/mu)**2 (default); sigma2: the submitted manuscript's ((p-s)*sigma)**2",
    )
    parser.add_argument(
        "--cost-model",
        choices=COST_MODELS,
        default="fixed",
        help="fixed: c = p/1.9, s = 0.4c (default); random: c = p*U[10/14,10/11], s = c*U[-0.9,0.9]",
    )
    parser.add_argument("--cost-seed", type=int, default=DEFAULT_COST_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: figure1_<variance-bound>_<cost-model>_results next to this script",
    )
    parser.add_argument(
        "--df-method",
        choices=("cuts", "misocp"),
        default="cuts",
        help="cuts: Lagrangian (dual) cut method (default); misocp: perspective MISOCP (20) solved by Gurobi",
    )
    parser.add_argument(
        "--no-zero-lambda",
        dest="zero_lambda",
        action="store_false",
        help="skip the lambda = 0 (Gallego-Moon) reference solves",
    )
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
    parser.set_defaults(resume=True, zero_lambda=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.mip_gap >= 0.0, "mip gap must be nonnegative")
    require(args.time_limit > 0.0, "time limit must be positive")
    if args.threads is not None:
        require(args.threads >= 1, "threads must be positive")
    if args.max_windows is not None:
        require(args.max_windows >= 1, "max-windows must be positive")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / default_output_directory(
            args.variance_bound, args.cost_model
        )

    if args.plot_only_from is not None:
        plot_figure(read_original_summaries(args.plot_only_from.resolve()), args.output_dir.resolve())
        print(f"Wrote {args.output_dir.resolve() / 'figure1_rebuilt.pdf'}")
        return 0

    schemes = parse_schemes(args.schemes)
    lambda_values = parse_lambdas(args.lambdas, args.variance_bound)
    if args.zero_lambda:
        lambda_values = (0.0,) + lambda_values
    require(args.solver.upper() == "GUROBI", "this Julia-equivalent implementation requires GUROBI")
    try:
        import gurobipy  # noqa: F401
    except ImportError as error:
        raise Figure1Error("gurobipy is not installed") from error
    data = load_retail_data(args.input.resolve())
    print(f"Loaded {data.product_count} products and {data.week_count} weeks from {args.input.resolve()}")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "run_metadata.json"
    run_settings = {
        "variance_bound": args.variance_bound,
        "cost_model": args.cost_model,
        "cost_seed": args.cost_seed if args.cost_model == "random" else None,
    }
    if args.resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_settings = {
            "variance_bound": previous.get("variance_bound", "sigma2"),
            "cost_model": previous.get("cost_model", "fixed"),
            "cost_seed": previous.get("cost_seed"),
        }
        require(
            previous_settings == run_settings,
            f"{output_root} holds results for {previous_settings}; "
            "use another --output-dir or --no-resume",
        )
    metadata = {
        **run_settings,
        "df_method": args.df_method,
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
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    summary_paths = {
        scheme: run_scheme(scheme, data, lambda_values, output_root, args)
        for scheme in schemes
    }
    if not args.no_plot:
        summaries = {scheme: read_csv_summary(path) for scheme, path in summary_paths.items()}
        lambda_max = {
            scheme: read_lambda_max(path.parent / "lambda_max_summary.csv")
            for scheme, path in summary_paths.items()
        }
        plot_figure(summaries, output_root, lambda_max)
    print(f"Results saved under {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Figure1Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
