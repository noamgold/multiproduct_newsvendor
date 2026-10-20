"""Generate the corrected random instances for manuscript Tables 1--3.

The global parameters below define the complete experiment grid. The program
uses only the Python standard library and writes one CSV dataset per table; it
does not construct an optimization model or invoke a solver.

Mathematical sources (read-only):
* specification_v1_2_1_review/COMPUTATIONAL_SPECIFICATION_V1_2_1.md
* python_replication/src/newsvendor_replication/{uniform,triangular,
  distribution_free,scalar_oracles,fixed_costs,generation}.py
* python_replication/ADR_023_PAIRED_FROZEN_INSTANCE_REQUIREMENT.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


# ---------------------------------------------------------------------------
# REQUIRED CONFIGURATION BLOCK
# ---------------------------------------------------------------------------
N_VALUES = [100, 200, 300, 400, 500, 600, 700]
REPLICATIONS = 30
MASTER_SEED = 20260829
KAPPA = 50
DF_LAMBDA = 1e-9
DF_SCORE_TOLERANCE = 1e-10
MAX_DF_GENERATION_ATTEMPTS = 100
GENERATION_VERSION = "TABLES_1_3_CORRECTED_STAGE1_V1"
# ---------------------------------------------------------------------------

FAMILIES = ("UNIFORM", "TRIANGULAR", "DISTRIBUTION_FREE")
RNG_ALGORITHM = "Python random.Random (MT19937)"
DF_REJECTION_REASON = "MAX_DF_SCORE_LE_SCORE_ZERO_TOLERANCE"
REJECTION_FIELDS = (
    "candidate_index",
    "reason_code",
    "observed_max_df_score",
    "score_zero_tolerance",
)

# Numerical tolerances are validation tolerances only.  Generated values are
# never clipped, shifted, rounded, or silently repaired.
FORMULA_ABS_TOLERANCE = 1e-10
FORMULA_REL_TOLERANCE = 2e-12
TRI_OBJECTIVE_SCALED_TOLERANCE = 1e-11
DF_GOLDEN_ABS_TOLERANCE = 1e-11
DF_DERIVATIVE_TOLERANCE = 1e-7
DF_NEARBY_VALUE_TOLERANCE = 1e-10

BASE_DIRECTORY = Path(__file__).resolve().parent
CSV_FILENAMES = {
    "UNIFORM": "uniform_table1_instances.csv",
    "TRIANGULAR": "triangular_table2_instances.csv",
    "DISTRIBUTION_FREE": "distribution_free_table3_instances.csv",
}

CSV_HEADERS = {
    "UNIFORM": (
        "family",
        "n",
        "replication",
        "instance_id",
        "product_id",
        "master_seed",
        "instance_seed",
        "a",
        "b",
        "cost_c",
        "price_p",
        "salvage_s",
        "price_markup_draw",
        "salvage_multiplier",
        "x_u",
        "fixed_cost_score",
        "k",
        "kappa",
        "K",
        "capacity_fraction",
        "C",
        "generation_version",
    ),
    "TRIANGULAR": (
        "family",
        "n",
        "replication",
        "instance_id",
        "product_id",
        "master_seed",
        "instance_seed",
        "a",
        "m",
        "b",
        "cost_c",
        "price_p",
        "salvage_s",
        "price_markup_draw",
        "salvage_multiplier",
        "x_u",
        "fixed_cost_score",
        "k",
        "kappa",
        "K",
        "capacity_fraction",
        "C",
        "generation_version",
    ),
    "DISTRIBUTION_FREE": (
        "family",
        "n",
        "replication",
        "instance_id",
        "product_id",
        "master_seed",
        "instance_seed",
        "mu",
        "sigma",
        "lambda",
        "cost_c",
        "price_p",
        "salvage_s",
        "price_markup_draw",
        "salvage_multiplier",
        "x_u",
        "fixed_cost_score",
        "k",
        "kappa",
        "K",
        "capacity_fraction",
        "C",
        "candidate_attempt",
        "candidate_rejection_count",
        "accepted_max_df_score",
        "rejection_records",
        "generation_version",
    ),
}

FORBIDDEN_IMPORT_ROOTS = {
    "cplex",
    "cvxpy",
    "docplex",
    "gurobi",
    "gurobipy",
    "mip",
    "mosek",
    "numpy",
    "ortools",
    "pandas",
    "pyomo",
    "scipy",
}


class ValidationError(ValueError):
    """Raised when generated or CSV data violates the Stage 1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _require_finite(label: str, value: float) -> None:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value),
        f"{label} must be a finite real number; observed {value!r}",
    )


def _close(actual: float, expected: float, label: str) -> None:
    _require_finite(label, actual)
    _require_finite(f"expected {label}", expected)
    difference = abs(actual - expected)
    allowance = FORMULA_ABS_TOLERANCE + FORMULA_REL_TOLERANCE * max(
        1.0, abs(actual), abs(expected)
    )
    _require(
        difference <= allowance,
        f"{label} mismatch: actual={actual!r}, expected={expected!r}, "
        f"difference={difference!r}, allowance={allowance!r}",
    )


def _scaled_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _validate_economic_order(price_p: float, cost_c: float, salvage_s: float) -> None:
    for label, value in (("price_p", price_p), ("cost_c", cost_c), ("salvage_s", salvage_s)):
        _require_finite(label, value)
    _require(price_p > cost_c > 0.0, f"expected price_p > cost_c > 0; got {price_p}, {cost_c}")
    _require(cost_c > salvage_s, f"expected cost_c > salvage_s; got {cost_c}, {salvage_s}")


def derive_instance_seed(family: str, n: int, replication: int) -> int:
    _require(family in FAMILIES, f"unknown family {family!r}")
    _require(isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be positive")
    _require(
        isinstance(replication, int) and not isinstance(replication, bool) and replication > 0,
        "replication must be positive",
    )
    family_number = FAMILIES.index(family) + 1
    return MASTER_SEED * 10_000_000 + family_number * 1_000_000 + n * 100 + replication


def stable_instance_id(family: str, n: int, replication: int) -> str:
    _require(family in FAMILIES, f"unknown family {family!r}")
    return f"{family}-N{n:04d}-R{replication:03d}"


def _draw_until(
    rng: Any,
    lower: float,
    upper: float,
    predicate: Callable[[float], bool],
    label: str,
) -> float:
    """Redraw deterministic MT19937 uniforms until the strict domain holds.

    This handles the finite-precision possibility of an excluded endpoint.  It
    never clips, shifts, rounds, or substitutes for the invalid draw.
    """
    for _ in range(1_000_000):
        value = rng.uniform(lower, upper)
        if predicate(value):
            return value
    raise RuntimeError(f"could not draw a valid {label} after 1,000,000 redraws")


def _draw_positive_cost(rng: Any) -> float:
    # The upper endpoint 10 is permitted; only the excluded zero is redrawn.
    return _draw_until(rng, 0.0, 10.0, lambda value: value > 0.0, "positive cost")


def _draw_open_mode_fraction(rng: Any) -> float:
    return _draw_until(
        rng,
        0.0,
        1.0,
        lambda value: 0.0 < value < 1.0,
        "strict triangular mode fraction",
    )


# Uniform functions are the complete certified functions from uniform.py.
def uniform_unconstrained_quantity(a: float, b: float, price_p: float, cost_c: float, salvage_s: float) -> float:
    return a + (b - a) * (price_p - cost_c) / (price_p - salvage_s)


def uniform_expected_profit(
    x: float,
    a: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    _require(x >= 0.0, f"uniform quantity must be nonnegative; observed {x!r}")
    if x <= a:
        return (price_p - cost_c) * x
    if x <= b:
        return (price_p - cost_c) * x + (salvage_s - price_p) * (x - a) ** 2 / (
            2.0 * (b - a)
        )
    return (salvage_s - cost_c) * x + (price_p - salvage_s) * (a + b) / 2.0


def triangular_segment_quantities(x: float, a: float, m: float, b: float) -> tuple[float, float, float]:
    _require(0.0 <= x <= b, f"triangular segment quantity must lie in [0,b]; observed {x!r}")
    x_bar = min(x, a)
    x_hat = min(max(x - a, 0.0), m - a)
    x_tilde = min(max(x - m, 0.0), b - m)
    return x_bar, x_hat, x_tilde


# The current direct expression is copied faithfully from certified triangular.py.
def triangular_current_direct_profit(
    x: float,
    a: float,
    m: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    _require(x >= 0.0, f"triangular quantity must be nonnegative; observed {x!r}")
    if x > b:
        return (salvage_s - cost_c) * x + (price_p - salvage_s) * (a + m + b) / 3.0
    x_bar, x_hat, x_tilde = triangular_segment_quantities(x, a, m, b)
    return (
        (price_p - cost_c) * (x_bar + x_hat + x_tilde)
        + (salvage_s - price_p) * (m - a) * x_tilde / (b - a)
        + (salvage_s - price_p) * x_tilde**2 / (b - a)
        + (salvage_s - price_p)
        * (x_hat**3 / (m - a) - x_tilde**3 / (b - m))
        / (3.0 * (b - a))
    )


# This remains a separate implementation of the corrected 2018 polynomial.
def triangular_corrected_2018_raw_objective(
    x: float,
    a: float,
    m: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    _require(x >= 0.0, f"triangular quantity must be nonnegative; observed {x!r}")
    if x > b:
        normalized_tail = (salvage_s - cost_c) * x + (price_p - salvage_s) * (a + m + b) / 3.0
        return normalized_tail - price_p * (a + m + b) / 3.0
    x_bar, x_hat, x_tilde = triangular_segment_quantities(x, a, m, b)
    holding_h = -salvage_s
    return (
        (price_p - cost_c) * (x_bar + x_hat + x_tilde)
        - (price_p + holding_h) * (m - a) * x_tilde / (b - a)
        - (price_p + holding_h) * x_tilde**2 / (b - a)
        + (price_p + holding_h)
        * (x_tilde**3 / (b - m) - x_hat**3 / (m - a))
        / (3.0 * (b - a))
        - price_p * (a + m + b) / 3.0
    )


def triangular_corrected_2018_normalized_objective(
    x: float,
    a: float,
    m: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    return triangular_corrected_2018_raw_objective(
        x, a, m, b, price_p, cost_c, salvage_s
    ) + price_p * (a + m + b) / 3.0


def triangular_cdf(x: float, a: float, m: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x <= m:
        return (x - a) ** 2 / ((b - a) * (m - a))
    if x < b:
        return 1.0 - (b - x) ** 2 / ((b - a) * (b - m))
    return 1.0


def triangular_unconstrained_quantity(
    a: float,
    m: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    alpha = (price_p - cost_c) / (price_p - salvage_s)
    mode_cdf = (m - a) / (b - a)
    if alpha <= mode_cdf:
        return a + math.sqrt(alpha * (b - a) * (m - a))
    return b - math.sqrt((1.0 - alpha) * (b - a) * (b - m))


# Distribution-Free functions follow certified distribution_free.py; the two
# derivatives are independently implemented from the approved formulas.
def df_theta(
    x: float,
    mu: float,
    sigma: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    _require(x >= 0.0, f"Distribution-Free quantity must be nonnegative; observed {x!r}")
    return (cost_c - salvage_s) * x + (price_p - salvage_s) * (
        math.sqrt(sigma**2 + (x - mu) ** 2) - x + mu
    ) / 2.0


def df_g(
    x: float,
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    return df_theta(x, mu, sigma, price_p, cost_c, salvage_s) + lambda_ * (
        price_p - salvage_s
    ) ** 2 * sigma**2 * x**2


def df_g_prime(
    x: float,
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    radius = math.sqrt(sigma**2 + (x - mu) ** 2)
    return (
        (cost_c - salvage_s)
        + (price_p - salvage_s) / 2.0 * ((x - mu) / radius - 1.0)
        + 2.0 * lambda_ * (price_p - salvage_s) ** 2 * sigma**2 * x
    )


def df_g_second(
    x: float,
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    denominator = (sigma**2 + (x - mu) ** 2) ** 1.5
    return (
        (price_p - salvage_s) / 2.0 * sigma**2 / denominator
        + 2.0 * lambda_ * (price_p - salvage_s) ** 2 * sigma**2
    )


def _bounded_minimize(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = DF_GOLDEN_ABS_TOLERANCE,
    max_iterations: int = 500,
) -> tuple[float, float]:
    """Deterministic golden-section bounded minimizer adapted from scalar_oracles.py."""
    _require(lower < upper, "bounded minimizer requires lower < upper")
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower, upper
    c_point = right - inverse_phi * (right - left)
    d_point = left + inverse_phi * (right - left)
    c_value, d_value = objective(c_point), objective(d_point)
    for _ in range(max_iterations):
        if right - left <= tolerance:
            break
        if c_value <= d_value:
            right, d_point, d_value = d_point, c_point, c_value
            c_point = right - inverse_phi * (right - left)
            c_value = objective(c_point)
        else:
            left, c_point, c_value = c_point, d_point, d_value
            d_point = left + inverse_phi * (right - left)
            d_value = objective(d_point)
    else:
        raise RuntimeError("golden-section minimizer exceeded its iteration limit")
    candidates = (
        (left, objective(left)),
        (right, objective(right)),
        (c_point, c_value),
        (d_point, d_value),
    )
    return min(candidates, key=lambda item: (item[1], item[0]))


def distribution_free_unconstrained_quantity(
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> tuple[float, float]:
    """Expand a deterministic bracket and minimize g on the bounded interval."""
    objective = lambda x: df_g(x, mu, sigma, lambda_, price_p, cost_c, salvage_s)
    upper = max(1.0, mu + sigma)
    previous = objective(0.0)
    for _ in range(60):
        at_upper = objective(upper)
        at_half = objective(upper / 2.0)
        if at_upper >= at_half or at_upper >= previous:
            break
        previous = at_upper
        upper *= 2.0
    else:
        raise RuntimeError("could not establish a finite Distribution-Free bracket")
    return _bounded_minimize(objective, 0.0, upper)


def df_scaled_derivative_residual(
    x: float,
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    radius = math.sqrt(sigma**2 + (x - mu) ** 2)
    terms = (
        cost_c - salvage_s,
        (price_p - salvage_s) / 2.0 * ((x - mu) / radius - 1.0),
        2.0 * lambda_ * (price_p - salvage_s) ** 2 * sigma**2 * x,
    )
    return abs(sum(terms)) / max(1.0, sum(abs(term) for term in terms))


def validate_distribution_free_minimizer(
    x_u: float,
    mu: float,
    sigma: float,
    lambda_: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    _require(x_u >= 0.0, f"Distribution-Free x_u must be nonnegative; observed {x_u!r}")
    curvature = df_g_second(x_u, mu, sigma, lambda_, price_p, cost_c, salvage_s)
    _require(curvature > 0.0, f"Distribution-Free curvature must be positive; observed {curvature!r}")
    derivative = df_g_prime(x_u, mu, sigma, lambda_, price_p, cost_c, salvage_s)
    if x_u == 0.0:
        _require(
            derivative >= -DF_DERIVATIVE_TOLERANCE,
            f"boundary optimality failed: g'(0)={derivative!r}",
        )
        residual = max(0.0, -derivative) / max(1.0, abs(derivative))
    else:
        residual = df_scaled_derivative_residual(
            x_u, mu, sigma, lambda_, price_p, cost_c, salvage_s
        )
        _require(
            residual <= DF_DERIVATIVE_TOLERANCE,
            f"interior derivative residual {residual!r} exceeds {DF_DERIVATIVE_TOLERANCE!r}",
        )

    value = df_g(x_u, mu, sigma, lambda_, price_p, cost_c, salvage_s)
    perturbation = max(1e-6, 1e-6 * max(1.0, x_u))
    nearby_points = [x_u + perturbation]
    if x_u > 0.0:
        nearby_points.append(max(0.0, x_u - perturbation))
    for nearby in nearby_points:
        nearby_value = df_g(nearby, mu, sigma, lambda_, price_p, cost_c, salvage_s)
        allowance = DF_NEARBY_VALUE_TOLERANCE * max(1.0, abs(value), abs(nearby_value))
        _require(
            value <= nearby_value + allowance,
            f"g(x_u) exceeds nearby feasible g({nearby!r}) by more than tolerance",
        )
    return residual


def _family_constants(family: str) -> dict[str, Any]:
    if family == "UNIFORM":
        return {"demand_distribution": "UNIFORM"}
    if family == "TRIANGULAR":
        return {
            "demand_distribution": "TRIANGULAR",
            "objective_normalization": "TRI_2018_ADD_EXPECTED_REVENUE_V1_2_1",
        }
    if family == "DISTRIBUTION_FREE":
        return {
            "demand_distribution": "DISTRIBUTION_FREE",
            "lambda": DF_LAMBDA,
            "score_zero_tolerance": DF_SCORE_TOLERANCE,
            "max_generation_attempts": MAX_DF_GENERATION_ATTEMPTS,
        }
    raise ValidationError(f"unknown family {family!r}")


def _draw_common_economics(rng: random.Random, n: int) -> dict[str, Any]:
    """Historical draw scope/order: all costs, one markup, all salvage draws, r3."""
    costs = [_draw_positive_cost(rng) for _ in range(n)]
    price_markup_draw = rng.uniform(0.0, 1.4)
    prices = [cost * (1.1 + price_markup_draw) for cost in costs]
    salvage_multipliers = [rng.uniform(-0.9, 0.9) for _ in range(n)]
    salvages = [multiplier * cost for multiplier, cost in zip(salvage_multipliers, costs, strict=True)]
    capacity_fraction = rng.uniform(1.0 / 5.0, 1.0 / 3.0)
    for index, (price, cost, salvage) in enumerate(zip(prices, costs, salvages, strict=True), 1):
        try:
            _validate_economic_order(price, cost, salvage)
        except ValidationError as error:
            raise ValidationError(f"economic domain failed for product {index}: {error}") from error
    return {
        "costs": costs,
        "prices": prices,
        "salvages": salvages,
        "salvage_multipliers": salvage_multipliers,
        "price_markup_draw": price_markup_draw,
        "capacity_fraction": capacity_fraction,
    }


def _candidate_payload_from_instance(instance: dict[str, Any]) -> dict[str, Any]:
    products = [{key: value for key, value in product.items() if key != "k"} for product in instance["products"]]
    return {
        "family": instance["family"],
        "n": instance["n"],
        "replication": instance["replication"],
        "instance_id": instance["instance_id"],
        "master_seed": instance["master_seed"],
        "instance_seed": instance["instance_seed"],
        "products": products,
        "price_markup_draw": instance["price_markup_draw"],
        "capacity_fraction": instance["capacity_fraction"],
        "family_constants": instance["family_constants"],
        "generation_version": instance["generation_version"],
    }


def _serialize_rejection_records(records: Sequence[dict[str, Any]]) -> str:
    for record in records:
        _require(tuple(record) == REJECTION_FIELDS, f"rejection fields/order are not exact: {tuple(record)!r}")
    return "|".join(
        "^".join(str(record[field]) for field in REJECTION_FIELDS)
        for record in records
    )


def _parse_rejection_records(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    records = []
    for encoded in text.split("|"):
        values = encoded.split("^")
        _require(len(values) == len(REJECTION_FIELDS), "invalid rejection record")
        records.append(
            {
                "candidate_index": int(values[0]),
                "reason_code": values[1],
                "observed_max_df_score": float(values[2]),
                "score_zero_tolerance": float(values[3]),
            }
        )
    return records


def _whole_candidate_rejection(
    rng: random.Random,
    candidate_factory: Callable[[random.Random, int], dict[str, Any]],
    score_function: Callable[[dict[str, Any]], Sequence[float]],
    *,
    score_zero_tolerance: float,
    max_generation_attempts: int,
) -> tuple[dict[str, Any], int, float, list[dict[str, Any]]]:
    _require(score_zero_tolerance >= 0.0, "score_zero_tolerance must be nonnegative")
    _require(max_generation_attempts >= 1, "max_generation_attempts must be positive")
    rejections: list[dict[str, Any]] = []
    for attempt in range(1, max_generation_attempts + 1):
        candidate = candidate_factory(rng, attempt)
        scores = tuple(score_function(candidate))
        _require(bool(scores), "candidate score vector must be nonempty")
        for index, score in enumerate(scores, 1):
            _require_finite(f"candidate score {index}", score)
        observed_max = max(scores)
        if observed_max <= score_zero_tolerance:
            rejections.append(
                {
                    "candidate_index": attempt,
                    "reason_code": DF_REJECTION_REASON,
                    "observed_max_df_score": observed_max,
                    "score_zero_tolerance": score_zero_tolerance,
                }
            )
            continue
        return candidate, attempt, observed_max, rejections
    raise RuntimeError("maximum Distribution-Free generation attempts exhausted")


def _finalize_instance(
    family: str,
    n: int,
    replication: int,
    instance_seed: int,
    products_without_k: Sequence[dict[str, Any]],
    price_markup_draw: float,
    capacity_fraction: float,
    acceptance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = [product["fixed_cost_score"] for product in products_without_k]
    _require(len(scores) == n and n > 0, "product score vector has the wrong length")
    score_shift = max(scores) / 10.0
    products = []
    for product in products_without_k:
        enriched = dict(product)
        enriched["k"] = product["fixed_cost_score"] + score_shift
        _require(
            enriched["k"] >= 0.0,
            f"theoretically invalid negative fixed-cost value {enriched['k']!r}",
        )
        products.append(enriched)
    budget_K = KAPPA / 101.0 * sum(product["k"] for product in products)
    capacity_C = capacity_fraction * sum(product["x_u"] for product in products)
    instance: dict[str, Any] = {
        "family": family,
        "n": n,
        "replication": replication,
        "instance_id": stable_instance_id(family, n, replication),
        "master_seed": MASTER_SEED,
        "instance_seed": instance_seed,
        "products": products,
        "price_markup_draw": price_markup_draw,
        "kappa": KAPPA,
        "K": budget_K,
        "capacity_fraction": capacity_fraction,
        "C": capacity_C,
        "family_constants": _family_constants(family),
        "generation_version": GENERATION_VERSION,
    }
    if acceptance is not None:
        instance.update(acceptance)
    validate_tables_1_to_3_instance(instance)
    return instance


def _generate_uniform_instance(n: int, replication: int, rng: random.Random, seed: int) -> dict[str, Any]:
    # Historical family order: all a draws, all span draws, then common economics.
    a_values = [rng.uniform(0.0, 100.0) for _ in range(n)]
    spans = [rng.uniform(100.0, 500.0) for _ in range(n)]
    economics = _draw_common_economics(rng, n)
    products = []
    for index in range(n):
        a = a_values[index]
        b = a + spans[index]
        price = economics["prices"][index]
        cost = economics["costs"][index]
        salvage = economics["salvages"][index]
        x_u = uniform_unconstrained_quantity(a, b, price, cost, salvage)
        score = uniform_expected_profit(x_u, a, b, price, cost, salvage)
        products.append(
            {
                "product_id": index + 1,
                "a": a,
                "b": b,
                "cost_c": cost,
                "price_p": price,
                "salvage_s": salvage,
                "salvage_multiplier": economics["salvage_multipliers"][index],
                "x_u": x_u,
                "fixed_cost_score": score,
            }
        )
    return _finalize_instance(
        "UNIFORM",
        n,
        replication,
        seed,
        products,
        economics["price_markup_draw"],
        economics["capacity_fraction"],
    )


def _generate_triangular_instance(n: int, replication: int, rng: random.Random, seed: int) -> dict[str, Any]:
    # Historical family order: all a, all spans, all strict mode fractions, then economics.
    a_values = [rng.uniform(0.0, 100.0) for _ in range(n)]
    spans = [rng.uniform(100.0, 500.0) for _ in range(n)]
    mode_fractions = [_draw_open_mode_fraction(rng) for _ in range(n)]
    economics = _draw_common_economics(rng, n)
    products = []
    for index in range(n):
        a = a_values[index]
        b = a + spans[index]
        m = a + mode_fractions[index] * (b - a)
        price = economics["prices"][index]
        cost = economics["costs"][index]
        salvage = economics["salvages"][index]
        x_u = triangular_unconstrained_quantity(a, m, b, price, cost, salvage)
        current = triangular_current_direct_profit(x_u, a, m, b, price, cost, salvage)
        normalized = triangular_corrected_2018_normalized_objective(
            x_u, a, m, b, price, cost, salvage
        )
        _require(
            _scaled_difference(current, normalized) <= TRI_OBJECTIVE_SCALED_TOLERANCE,
            "current and corrected normalized triangular objectives disagree at x_u",
        )
        products.append(
            {
                "product_id": index + 1,
                "a": a,
                "m": m,
                "b": b,
                "cost_c": cost,
                "price_p": price,
                "salvage_s": salvage,
                "salvage_multiplier": economics["salvage_multipliers"][index],
                "x_u": x_u,
                "fixed_cost_score": current,
            }
        )
    return _finalize_instance(
        "TRIANGULAR",
        n,
        replication,
        seed,
        products,
        economics["price_markup_draw"],
        economics["capacity_fraction"],
    )


def _generate_df_candidate(
    n: int,
    replication: int,
    seed: int,
    rng: random.Random,
) -> dict[str, Any]:
    # Historical family order: all means, all conditional sigmas, then economics.
    mu_values = [rng.uniform(100.0, 500.0) for _ in range(n)]
    sigma_values = [rng.uniform(mu / 6.0, mu / 2.0) for mu in mu_values]
    economics = _draw_common_economics(rng, n)
    products = []
    for index in range(n):
        mu = mu_values[index]
        sigma = sigma_values[index]
        price = economics["prices"][index]
        cost = economics["costs"][index]
        salvage = economics["salvages"][index]
        x_u, _ = distribution_free_unconstrained_quantity(
            mu, sigma, DF_LAMBDA, price, cost, salvage
        )
        validate_distribution_free_minimizer(
            x_u, mu, sigma, DF_LAMBDA, price, cost, salvage
        )
        score = df_g(0.0, mu, sigma, DF_LAMBDA, price, cost, salvage) - df_g(
            x_u, mu, sigma, DF_LAMBDA, price, cost, salvage
        )
        products.append(
            {
                "product_id": index + 1,
                "mu": mu,
                "sigma": sigma,
                "lambda": DF_LAMBDA,
                "cost_c": cost,
                "price_p": price,
                "salvage_s": salvage,
                "salvage_multiplier": economics["salvage_multipliers"][index],
                "x_u": x_u,
                "fixed_cost_score": score,
            }
        )
    return {
        "family": "DISTRIBUTION_FREE",
        "n": n,
        "replication": replication,
        "instance_id": stable_instance_id("DISTRIBUTION_FREE", n, replication),
        "master_seed": MASTER_SEED,
        "instance_seed": seed,
        "products": products,
        "price_markup_draw": economics["price_markup_draw"],
        "capacity_fraction": economics["capacity_fraction"],
        "family_constants": _family_constants("DISTRIBUTION_FREE"),
        "generation_version": GENERATION_VERSION,
    }


def _generate_distribution_free_instance(
    n: int, replication: int, rng: random.Random, seed: int
) -> dict[str, Any]:
    candidate, attempt, accepted_max, rejections = _whole_candidate_rejection(
        rng,
        lambda active_rng, _attempt: _generate_df_candidate(n, replication, seed, active_rng),
        lambda value: [product["fixed_cost_score"] for product in value["products"]],
        score_zero_tolerance=DF_SCORE_TOLERANCE,
        max_generation_attempts=MAX_DF_GENERATION_ATTEMPTS,
    )
    acceptance = {
        "candidate_attempt": attempt,
        "candidate_rejection_count": len(rejections),
        "accepted_max_df_score": accepted_max,
        "rejection_records": _serialize_rejection_records(rejections),
    }
    return _finalize_instance(
        "DISTRIBUTION_FREE",
        n,
        replication,
        seed,
        candidate["products"],
        candidate["price_markup_draw"],
        candidate["capacity_fraction"],
        acceptance,
    )


def generate_tables_1_to_3_instance(
    family: str, n: int, replication: int
) -> dict[str, Any]:
    seed = derive_instance_seed(family, n, replication)
    rng = random.Random(seed)
    if family == "UNIFORM":
        return _generate_uniform_instance(n, replication, rng, seed)
    if family == "TRIANGULAR":
        return _generate_triangular_instance(n, replication, rng, seed)
    if family == "DISTRIBUTION_FREE":
        return _generate_distribution_free_instance(n, replication, rng, seed)
    raise ValidationError(f"unknown family {family!r}")


def _validate_product_common(product: dict[str, Any], instance: dict[str, Any], index: int) -> None:
    _require(product["product_id"] == index, f"product IDs must be one-based and stable; expected {index}")
    price = product["price_p"]
    cost = product["cost_c"]
    salvage = product["salvage_s"]
    _validate_economic_order(price, cost, salvage)
    _close(price, cost * (1.1 + instance["price_markup_draw"]), f"product {index} price")
    _close(salvage, product["salvage_multiplier"] * cost, f"product {index} salvage")
    _require(-0.9 <= product["salvage_multiplier"] <= 0.9, "salvage multiplier outside [-0.9,0.9]")
    _require(product["x_u"] >= 0.0, f"product {index} x_u is negative")
    _require_finite(f"product {index} fixed_cost_score", product["fixed_cost_score"])


def validate_tables_1_to_3_instance(instance: dict[str, Any]) -> float:
    family = instance["family"]
    n = instance["n"]
    replication = instance["replication"]
    _require(family in FAMILIES, f"unknown family {family!r}")
    _require(n > 0 and replication > 0, "n and replication must be positive")
    _require(instance["instance_id"] == stable_instance_id(family, n, replication), "instance ID mismatch")
    _require(instance["master_seed"] == MASTER_SEED, "master seed mismatch")
    _require(instance["instance_seed"] == derive_instance_seed(family, n, replication), "instance seed mismatch")
    _require(instance["generation_version"] == GENERATION_VERSION, "generation version mismatch")
    _require(instance["kappa"] == KAPPA, "kappa mismatch")
    _require(instance["family_constants"] == _family_constants(family), "family constants mismatch")
    _require(1.0 / 5.0 <= instance["capacity_fraction"] <= 1.0 / 3.0, "capacity fraction out of range")
    products = instance["products"]
    _require(len(products) == n, f"expected {n} products; observed {len(products)}")

    maximum_df_residual = 0.0
    for index, product in enumerate(products, 1):
        _validate_product_common(product, instance, index)
        price, cost, salvage = product["price_p"], product["cost_c"], product["salvage_s"]
        if family == "UNIFORM":
            a, b = product["a"], product["b"]
            _require(0.0 <= a <= 100.0, f"uniform a outside [0,100] for product {index}")
            _require(100.0 <= b - a <= 500.0, f"uniform span outside [100,500] for product {index}")
            expected_x = uniform_unconstrained_quantity(a, b, price, cost, salvage)
            expected_score = uniform_expected_profit(expected_x, a, b, price, cost, salvage)
        elif family == "TRIANGULAR":
            a, m, b = product["a"], product["m"], product["b"]
            _require(0.0 <= a < m < b, f"triangular strict domain failed for product {index}")
            _require(a <= 100.0 and 100.0 <= b - a <= 500.0, "triangular support draw outside range")
            expected_x = triangular_unconstrained_quantity(a, m, b, price, cost, salvage)
            alpha = (price - cost) / (price - salvage)
            _close(triangular_cdf(expected_x, a, m, b), alpha, f"product {index} triangular fractile")
            current = triangular_current_direct_profit(expected_x, a, m, b, price, cost, salvage)
            normalized = triangular_corrected_2018_normalized_objective(
                expected_x, a, m, b, price, cost, salvage
            )
            _require(
                _scaled_difference(current, normalized) <= TRI_OBJECTIVE_SCALED_TOLERANCE,
                f"triangular objective normalization mismatch for product {index}",
            )
            expected_score = current
        else:
            mu, sigma, lambda_ = product["mu"], product["sigma"], product["lambda"]
            _require(100.0 <= mu <= 500.0, f"Distribution-Free mu outside range for product {index}")
            _require(mu / 6.0 <= sigma <= mu / 2.0, f"Distribution-Free sigma outside range for product {index}")
            _require(lambda_ == DF_LAMBDA, f"Distribution-Free lambda mismatch for product {index}")
            expected_x = product["x_u"]
            maximum_df_residual = max(
                maximum_df_residual,
                validate_distribution_free_minimizer(
                    expected_x, mu, sigma, lambda_, price, cost, salvage
                ),
            )
            expected_score = df_g(0.0, mu, sigma, lambda_, price, cost, salvage) - df_g(
                expected_x, mu, sigma, lambda_, price, cost, salvage
            )
        _close(product["x_u"], expected_x, f"product {index} x_u")
        _close(product["fixed_cost_score"], expected_score, f"product {index} fixed-cost score")

    scores = [product["fixed_cost_score"] for product in products]
    shift = max(scores) / 10.0
    for index, product in enumerate(products, 1):
        expected_k = product["fixed_cost_score"] + shift
        _close(product["k"], expected_k, f"product {index} k")
        _require(product["k"] >= 0.0, f"theoretically invalid negative k for product {index}")
    expected_K = KAPPA / 101.0 * sum(product["k"] for product in products)
    expected_C = instance["capacity_fraction"] * sum(product["x_u"] for product in products)
    _close(instance["K"], expected_K, "K")
    _close(instance["C"], expected_C, "C")

    if family == "DISTRIBUTION_FREE":
        _require(max(scores) > DF_SCORE_TOLERANCE, "accepted DF candidate fails the score threshold")
        records = _parse_rejection_records(instance["rejection_records"])
        _require(len(records) == instance["candidate_rejection_count"], "DF rejection count mismatch")
        _require(instance["candidate_attempt"] == len(records) + 1, "DF candidate attempt mismatch")
        _require(instance["candidate_attempt"] <= MAX_DF_GENERATION_ATTEMPTS, "DF attempt exceeds maximum")
        for expected_index, record in enumerate(records, 1):
            _require(tuple(record) == REJECTION_FIELDS, "DF rejection record fields/order are not exact")
            _require(record["candidate_index"] == expected_index, "DF rejection candidate index mismatch")
            _require(record["reason_code"] == DF_REJECTION_REASON, "DF rejection reason mismatch")
            _require(record["observed_max_df_score"] <= DF_SCORE_TOLERANCE, "invalid rejected DF score")
            _require(record["score_zero_tolerance"] == DF_SCORE_TOLERANCE, "DF rejection tolerance mismatch")
        _require(
            instance["rejection_records"] == _serialize_rejection_records(records),
            "DF rejection records are not deterministic",
        )
        _close(instance["accepted_max_df_score"], max(scores), "accepted maximum DF score")
    return maximum_df_residual


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        _require_finite("CSV float", value)
        return repr(value)
    return str(value)


def rows_for_instance(instance: dict[str, Any]) -> Iterator[dict[str, str]]:
    family = instance["family"]
    for product in instance["products"]:
        common: dict[str, Any] = {
            "family": family,
            "n": instance["n"],
            "replication": instance["replication"],
            "instance_id": instance["instance_id"],
            "product_id": product["product_id"],
            "master_seed": instance["master_seed"],
            "instance_seed": instance["instance_seed"],
            "cost_c": product["cost_c"],
            "price_p": product["price_p"],
            "salvage_s": product["salvage_s"],
            "price_markup_draw": instance["price_markup_draw"],
            "salvage_multiplier": product["salvage_multiplier"],
            "x_u": product["x_u"],
            "fixed_cost_score": product["fixed_cost_score"],
            "k": product["k"],
            "kappa": instance["kappa"],
            "K": instance["K"],
            "capacity_fraction": instance["capacity_fraction"],
            "C": instance["C"],
            "generation_version": instance["generation_version"],
        }
        if family == "UNIFORM":
            common.update(a=product["a"], b=product["b"])
        elif family == "TRIANGULAR":
            common.update(a=product["a"], m=product["m"], b=product["b"])
        else:
            common.update(
                mu=product["mu"],
                sigma=product["sigma"],
                **{"lambda": product["lambda"]},
                candidate_attempt=instance["candidate_attempt"],
                candidate_rejection_count=instance["candidate_rejection_count"],
                accepted_max_df_score=instance["accepted_max_df_score"],
                rejection_records=instance["rejection_records"],
            )
        yield {column: _csv_value(common[column]) for column in CSV_HEADERS[family]}


def _parse_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise ValidationError(f"invalid integer in field {field!r}: {row.get(field)!r}") from error


def _parse_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ValidationError(f"invalid float in field {field!r}: {row.get(field)!r}") from error
    _require_finite(field, value)
    return value


def _instance_from_csv_group(family: str, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    first = rows[0]
    products: list[dict[str, Any]] = []
    for row in rows:
        product: dict[str, Any] = {
            "product_id": _parse_int(row, "product_id"),
            "cost_c": _parse_float(row, "cost_c"),
            "price_p": _parse_float(row, "price_p"),
            "salvage_s": _parse_float(row, "salvage_s"),
            "salvage_multiplier": _parse_float(row, "salvage_multiplier"),
            "x_u": _parse_float(row, "x_u"),
            "fixed_cost_score": _parse_float(row, "fixed_cost_score"),
            "k": _parse_float(row, "k"),
        }
        if family == "UNIFORM":
            product.update(a=_parse_float(row, "a"), b=_parse_float(row, "b"))
        elif family == "TRIANGULAR":
            product.update(
                a=_parse_float(row, "a"),
                m=_parse_float(row, "m"),
                b=_parse_float(row, "b"),
            )
        else:
            product.update(
                mu=_parse_float(row, "mu"),
                sigma=_parse_float(row, "sigma"),
                **{"lambda": _parse_float(row, "lambda")},
            )
        products.append(product)
    instance: dict[str, Any] = {
        "family": first["family"],
        "n": _parse_int(first, "n"),
        "replication": _parse_int(first, "replication"),
        "instance_id": first["instance_id"],
        "master_seed": _parse_int(first, "master_seed"),
        "instance_seed": _parse_int(first, "instance_seed"),
        "products": products,
        "price_markup_draw": _parse_float(first, "price_markup_draw"),
        "kappa": _parse_int(first, "kappa"),
        "K": _parse_float(first, "K"),
        "capacity_fraction": _parse_float(first, "capacity_fraction"),
        "C": _parse_float(first, "C"),
        "family_constants": _family_constants(family),
        "generation_version": first["generation_version"],
    }
    if family == "DISTRIBUTION_FREE":
        instance.update(
            candidate_attempt=_parse_int(first, "candidate_attempt"),
            candidate_rejection_count=_parse_int(first, "candidate_rejection_count"),
            accepted_max_df_score=_parse_float(first, "accepted_max_df_score"),
            rejection_records=first["rejection_records"],
        )
    return instance


def validate_csv_file(path: Path, family: str, *, require_full_grid: bool = True) -> dict[str, int]:
    """Validate a future CSV without modifying it."""
    expected_header = CSV_HEADERS[family]
    row_count = 0
    instance_count = 0
    current_key: tuple[int, int] | None = None
    group: list[dict[str, str]] = []

    expected_positions: Iterator[tuple[int, int, int]] | None = None
    if require_full_grid:
        expected_positions = iter(
            (n, replication, product_id)
            for n in N_VALUES
            for replication in range(1, REPLICATIONS + 1)
            for product_id in range(1, n + 1)
        )

    def finish_group() -> None:
        nonlocal instance_count
        if not group:
            return
        first = group[0]
        shared_fields = set(expected_header) - {
            "product_id", "a", "m", "b", "mu", "sigma", "cost_c", "price_p",
            "salvage_s", "salvage_multiplier", "x_u", "fixed_cost_score", "k",
        }
        for row in group[1:]:
            for field in shared_fields:
                _require(row[field] == first[field], f"field {field!r} varies within one instance")
        instance = _instance_from_csv_group(family, group)
        validate_tables_1_to_3_instance(instance)
        instance_count += 1

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            observed_header = tuple(next(reader))
        except StopIteration as error:
            raise ValidationError(f"CSV is empty: {path}") from error
        _require(observed_header == expected_header, f"CSV header mismatch for {family}")
        for physical_line, values in enumerate(reader, 2):
            _require(len(values) == len(expected_header), f"wrong column count on line {physical_line}")
            _require(tuple(values) != expected_header, f"duplicate header on line {physical_line}")
            row = dict(zip(expected_header, values, strict=True))
            _require(row["family"] == family, f"wrong family on line {physical_line}")
            position = (_parse_int(row, "n"), _parse_int(row, "replication"), _parse_int(row, "product_id"))
            if expected_positions is not None:
                try:
                    expected = next(expected_positions)
                except StopIteration as error:
                    raise ValidationError("CSV has more rows than the configured grid") from error
                _require(position == expected, f"row order mismatch: observed {position}, expected {expected}")
            key = position[:2]
            if current_key is None:
                current_key = key
            elif key != current_key:
                finish_group()
                group.clear()
                current_key = key
            group.append(row)
            row_count += 1
    finish_group()
    if expected_positions is not None:
        try:
            missing = next(expected_positions)
        except StopIteration:
            missing = None
        _require(missing is None, f"CSV ended before configured position {missing}")
    return {"rows": row_count, "instances": instance_count}


def _write_table_dataset_to_temporary_csv(family: str) -> Path:
    filename = CSV_FILENAMES[family]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=BASE_DIRECTORY
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=CSV_HEADERS[family],
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for n in N_VALUES:
                for replication in range(1, REPLICATIONS + 1):
                    instance = generate_tables_1_to_3_instance(family, n, replication)
                    writer.writerows(rows_for_instance(instance))
            stream.flush()
            os.fsync(stream.fileno())
        validate_csv_file(temporary_path, family, require_full_grid=True)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def generate_tables_1_to_3_csv_files(*, force: bool) -> None:
    """Generate all three deterministic CSVs transactionally for future Stage 1B."""
    final_paths = {family: BASE_DIRECTORY / name for family, name in CSV_FILENAMES.items()}
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing CSV file(s): {names}")

    temporary_paths: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    committed: list[str] = []
    try:
        for family in FAMILIES:
            temporary_paths[family] = _write_table_dataset_to_temporary_csv(family)

        # Preserve old exact deliverables until every prepared file is committed.
        for family, final_path in final_paths.items():
            if final_path.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{final_path.name}.", suffix=".backup", dir=BASE_DIRECTORY
                )
                os.close(descriptor)
                backup_path = Path(backup_name)
                backup_path.unlink()
                os.replace(final_path, backup_path)
                backups[family] = backup_path

        try:
            for family in FAMILIES:
                os.replace(temporary_paths[family], final_paths[family])
                committed.append(family)
        except BaseException:
            for family in committed:
                final_paths[family].unlink(missing_ok=True)
            for family, backup_path in backups.items():
                os.replace(backup_path, final_paths[family])
            raise

        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        for family, backup_path in backups.items():
            if backup_path.exists() and not final_paths[family].exists():
                os.replace(backup_path, final_paths[family])


def validate_tables_1_to_3_csv_files() -> dict[str, dict[str, int]]:
    """Read and validate all exact deliverables; this function never writes."""
    results: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        path = BASE_DIRECTORY / CSV_FILENAMES[family]
        _require(path.is_file(), f"required CSV does not exist: {path}")
        results[family] = validate_csv_file(path, family, require_full_grid=True)
    return results


def validate_configuration() -> None:
    _require(N_VALUES == [100, 200, 300, 400, 500, 600, 700], "N_VALUES mismatch")
    _require(700 in N_VALUES, "Triangular n=700 must be configured")
    _require(REPLICATIONS == 30, "REPLICATIONS mismatch")
    _require(MASTER_SEED == 20260829, "MASTER_SEED mismatch")
    _require(KAPPA == 50, "KAPPA mismatch")
    _require(DF_LAMBDA == 1e-9, "DF_LAMBDA mismatch")
    _require(DF_SCORE_TOLERANCE == 1e-10, "DF_SCORE_TOLERANCE mismatch")
    _require(MAX_DF_GENERATION_ATTEMPTS == 100, "MAX_DF_GENERATION_ATTEMPTS mismatch")
    _require(GENERATION_VERSION == "TABLES_1_3_CORRECTED_STAGE1_V1", "generation version mismatch")
    _require(set(CSV_HEADERS) == set(FAMILIES), "CSV header families mismatch")


def _assert_raises(function: Callable[[], Any], expected_exception: type[BaseException], message: str) -> None:
    try:
        function()
    except expected_exception:
        return
    raise AssertionError(message)


def run_self_tests() -> int:
    """Run small in-memory checks and write no files."""
    before = tuple(sorted(path.name for path in BASE_DIRECTORY.iterdir()))
    state: dict[str, Any] = {}

    def test_configuration() -> None:
        validate_configuration()

    def test_stable_seed() -> None:
        expected = 202608291010001
        observed = derive_instance_seed("UNIFORM", 100, 1)
        assert observed == expected, f"stable seed regression: {observed} != {expected}"
        assert observed == derive_instance_seed("UNIFORM", 100, 1), "seed derivation is not stable"

    def test_seed_discrimination() -> None:
        seeds = {
            derive_instance_seed("UNIFORM", 100, 1),
            derive_instance_seed("TRIANGULAR", 100, 1),
            derive_instance_seed("UNIFORM", 200, 1),
            derive_instance_seed("UNIFORM", 100, 2),
        }
        assert len(seeds) == 4, "family, n, and replication must all change the seed"

    def test_uniform_domains() -> None:
        _validate_economic_order(5.0, 3.0, 1.0)
        _assert_raises(
            lambda: _validate_economic_order(3.0, 3.0, 1.0),
            ValidationError,
            "equal price/cost should fail",
        )

    def test_uniform_xu() -> None:
        value = uniform_unconstrained_quantity(10.0, 110.0, 5.0, 3.0, 1.0)
        assert value == 60.0, f"uniform critical fractile expected 60, observed {value}"

    def test_uniform_profit() -> None:
        assert uniform_expected_profit(5.0, 10.0, 110.0, 5.0, 3.0, 1.0) == 10.0
        assert uniform_expected_profit(60.0, 10.0, 110.0, 5.0, 3.0, 1.0) == 70.0
        assert uniform_expected_profit(120.0, 10.0, 110.0, 5.0, 3.0, 1.0) == 0.0

    def test_triangular_strict_mode() -> None:
        class EndpointStub:
            def __init__(self, values: Iterable[float]) -> None:
                self.values = iter(values)
                self.calls = 0

            def uniform(self, _lower: float, _upper: float) -> float:
                self.calls += 1
                return next(self.values)

        mode_stub = EndpointStub((0.0, 1.0, 0.25))
        assert _draw_open_mode_fraction(mode_stub) == 0.25 and mode_stub.calls == 3
        cost_stub = EndpointStub((0.0, 5.0))
        assert _draw_positive_cost(cost_stub) == 5.0 and cost_stub.calls == 2

    def test_triangular_xu_cdf() -> None:
        args = (0.0, 40.0, 100.0, 5.0, 3.0, 1.0)
        x_u = triangular_unconstrained_quantity(*args)
        alpha = (5.0 - 3.0) / (5.0 - 1.0)
        _close(triangular_cdf(x_u, 0.0, 40.0, 100.0), alpha, "triangular fixture CDF")
        assert 40.0 < x_u < 100.0, "triangular fixture should use upper branch"

    def test_triangular_objective_mapping() -> None:
        args = (10.0, 50.0, 110.0, 8.0, 4.0, 1.0)
        for x in (0.0, 10.0, 30.0, 50.0, 80.0, 110.0, 120.0):
            current = triangular_current_direct_profit(x, *args)
            normalized = triangular_corrected_2018_normalized_objective(x, *args)
            assert _scaled_difference(current, normalized) <= TRI_OBJECTIVE_SCALED_TOLERANCE, (
                f"triangular objective mismatch at x={x}: {current} versus {normalized}"
            )

    def test_df_functions_derivatives() -> None:
        args = (200.0, 50.0, DF_LAMBDA, 8.0, 5.0, 1.0)
        x = 150.0
        step = 1e-3
        analytic_first = df_g_prime(x, *args)
        finite_first = (df_g(x + step, *args) - df_g(x - step, *args)) / (2.0 * step)
        analytic_second = df_g_second(x, *args)
        finite_second = (df_g_prime(x + step, *args) - df_g_prime(x - step, *args)) / (2.0 * step)
        assert abs(analytic_first - finite_first) <= 1e-7, "DF first derivative finite-difference mismatch"
        assert abs(analytic_second - finite_second) <= 1e-9, "DF second derivative finite-difference mismatch"
        assert df_g(x, *args) == df_theta(x, args[0], args[1], args[3], args[4], args[5]) + DF_LAMBDA * (8.0 - 1.0) ** 2 * 50.0**2 * x**2

    def test_df_golden_residual() -> None:
        args = (200.0, 50.0, DF_LAMBDA, 8.0, 5.0, 1.0)
        x_u, _ = distribution_free_unconstrained_quantity(*args)
        residual = validate_distribution_free_minimizer(x_u, *args)
        assert residual <= DF_DERIVATIVE_TOLERANCE, f"DF derivative residual too large: {residual}"
        state["oracle_residual"] = residual

    def test_df_all_zero_diagnostic() -> None:
        args = (0.0, 1.0, DF_LAMBDA, 2.0, 1.0, 0.0)
        x_u, _ = distribution_free_unconstrained_quantity(*args)
        score = df_g(0.0, *args) - df_g(x_u, *args)
        assert x_u == 0.0, f"all-zero diagnostic expected x_u=0, observed {x_u}"
        assert score == 0.0, f"all-zero diagnostic expected score=0, observed {score}"

    def test_df_whole_candidate_rejection() -> None:
        rng = random.Random(12345)

        def factory(active_rng: random.Random, attempt: int) -> dict[str, Any]:
            return {"attempt_payload": attempt, "draw": active_rng.random(), "score": 0.0 if attempt == 1 else 2.0}

        accepted = _whole_candidate_rejection(
            rng,
            factory,
            lambda candidate: [candidate["score"]],
            score_zero_tolerance=DF_SCORE_TOLERANCE,
            max_generation_attempts=3,
        )
        assert accepted[1] == 2 and len(accepted[3]) == 1, "stub must reject once then accept"
        assert accepted[0]["attempt_payload"] == 2, "rejected candidate data was improperly retained"
        state["stub_acceptance"] = accepted

    def test_df_rejection_record() -> None:
        record = state["stub_acceptance"][3][0]
        assert tuple(record) == REJECTION_FIELDS, f"rejection fields mismatch: {tuple(record)}"
        assert record["reason_code"] == DF_REJECTION_REASON, "rejection reason mismatch"
        assert record["candidate_index"] == 1, "rejection candidate index mismatch"
        assert record["observed_max_df_score"] == 0.0, "rejection observed maximum mismatch"
        assert record["score_zero_tolerance"] == DF_SCORE_TOLERANCE, "rejection tolerance mismatch"

    def test_fixed_cost_capacity_fixture() -> None:
        scores = (10.0, 20.0)
        k_values = tuple(score + max(scores) / 10.0 for score in scores)
        budget = KAPPA / 101.0 * sum(k_values)
        capacity = 0.25 * sum((2.0, 3.0))
        assert k_values == (12.0, 22.0), f"fixture k mismatch: {k_values}"
        assert budget == 1700.0 / 101.0, f"fixture K mismatch: {budget}"
        assert capacity == 1.25, f"fixture C mismatch: {capacity}"

    def test_three_mini_instances() -> None:
        minis = {
            family: generate_tables_1_to_3_instance(family, 5, 1)
            for family in FAMILIES
        }
        assert all(len(instance["products"]) == 5 for instance in minis.values())
        assert all(instance["replication"] == 1 for instance in minis.values())
        state["minis"] = minis
        state["mini_df_residual"] = validate_tables_1_to_3_instance(
            minis["DISTRIBUTION_FREE"]
        )

    def test_deterministic_regeneration() -> None:
        for family, original in state["minis"].items():
            regenerated = generate_tables_1_to_3_instance(family, 5, 1)
            assert original == regenerated, (
                f"{family} mini-instance did not regenerate deterministically"
            )

    def test_csv_headers() -> None:
        assert CSV_HEADERS["UNIFORM"] == (
            "family", "n", "replication", "instance_id", "product_id", "master_seed",
            "instance_seed", "a", "b", "cost_c", "price_p", "salvage_s",
            "price_markup_draw", "salvage_multiplier", "x_u", "fixed_cost_score", "k",
            "kappa", "K", "capacity_fraction", "C", "generation_version",
        )
        assert CSV_HEADERS["TRIANGULAR"][7:10] == ("a", "m", "b")
        assert CSV_HEADERS["DISTRIBUTION_FREE"][7:10] == ("mu", "sigma", "lambda")
        assert CSV_HEADERS["DISTRIBUTION_FREE"][-5:] == (
            "candidate_attempt", "candidate_rejection_count",
            "accepted_max_df_score", "rejection_records", "generation_version",
        )

    def test_no_forbidden_imports_or_writes() -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        forbidden = roots & FORBIDDEN_IMPORT_ROOTS
        assert not forbidden, f"forbidden imports found: {sorted(forbidden)}"
        after = tuple(sorted(path.name for path in BASE_DIRECTORY.iterdir()))
        assert after == before, f"self-test changed directory contents: before={before}, after={after}"

    tests = (
        ("01 configuration validation", test_configuration),
        ("02 stable seed derivation", test_stable_seed),
        ("03 seed discrimination", test_seed_discrimination),
        ("04 Uniform domain checks", test_uniform_domains),
        ("05 Uniform critical fractile", test_uniform_xu),
        ("06 Uniform expected profit", test_uniform_profit),
        ("07 Triangular strict mode handling", test_triangular_strict_mode),
        ("08 Triangular x_u/CDF consistency", test_triangular_xu_cdf),
        ("09 Triangular objective normalization", test_triangular_objective_mapping),
        ("10 Distribution-Free g derivatives", test_df_functions_derivatives),
        ("11 Distribution-Free golden-section residual", test_df_golden_residual),
        ("12 Distribution-Free all-zero diagnostic", test_df_all_zero_diagnostic),
        ("13 Distribution-Free whole-candidate rejection", test_df_whole_candidate_rejection),
        ("14 Distribution-Free rejection record", test_df_rejection_record),
        ("15 k, K, C fixture", test_fixed_cost_capacity_fixture),
        ("16 three in-memory mini-instances", test_three_mini_instances),
        ("17 deterministic mini-instance regeneration", test_deterministic_regeneration),
        ("18 future CSV headers", test_csv_headers),
        ("19 no forbidden imports or file writes", test_no_forbidden_imports_or_writes),
    )

    passes = 0
    failures = 0
    for name, test in tests:
        try:
            test()
        except BaseException as error:
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            passes += 1
            print(f"PASS {name}")
    residual = state.get("mini_df_residual", state.get("oracle_residual", math.nan))
    print(f"DF_MINI_DERIVATIVE_RESIDUAL={residual:.17g}")
    print(f"SELF_TEST_SUMMARY PASS={passes} FAIL={failures}")
    return 0 if failures == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true", help="run small in-memory tests only")
    actions.add_argument("--generate", action="store_true", help="generate the Table 1--3 CSV datasets")
    actions.add_argument("--validate-only", action="store_true", help="validate existing CSV files read-only")
    parser.add_argument("--force", action="store_true", help="with --generate, replace only the three exact CSVs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.force and not arguments.generate:
        raise SystemExit("--force is valid only together with --generate")
    if arguments.self_test:
        return run_self_tests()
    if arguments.generate:
        validate_configuration()
        generate_tables_1_to_3_csv_files(force=arguments.force)
        print("Generated and validated the CSV instance datasets for Tables 1--3")
        return 0
    validate_configuration()
    results = validate_tables_1_to_3_csv_files()
    for family in FAMILIES:
        print(f"{family}: rows={results[family]['rows']} instances={results[family]['instances']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
