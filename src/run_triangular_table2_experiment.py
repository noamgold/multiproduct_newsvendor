#!/usr/bin/env python3
"""Solve and compare the two Triangular-demand formulations used in Table 2.

The script is deliberately server-portable. It uses CVXPY as the modeling
layer (via triangular_current.py) and selects CPLEX, MOSEK, or GUROBI at run
time. It never generates or modifies input data.

For every frozen Triangular instance it builds:

* CURRENT_TRIANGULAR_BIGM_FIXED_COST (manuscript formulation (9)); and
* CURRENT_TRIANGULAR_PERSPECTIVE_FIXED_COST (manuscript (9) with the
  hyperbolic constraint (9c) replaced by the perspective/bilinear
  constraint (5), i.e. the model behind manuscript equation (12)'s Table 2
  "Conic (9)" column).

Both formulations receive the same immutable in-memory instance. The common
economic objective is recomputed independently from the returned physical
order quantities using the closed-form triangular profit function (7),
rather than by evaluating a CVXPY objective expression.

Outputs are append-only raw CSV records plus derived CSV comparison and
summary tables. A time-limit result is never classified as optimal.

Required runtime packages:

    numpy
    cvxpy
    cplex       # when --solver CPLEX is selected
    Mosek       # when --solver MOSEK is selected
    gurobipy    # when --solver GUROBI is selected

Model construction itself lives in triangular_current.py (build_current_
triangular_bigm_fixed_cost / build_current_triangular_perspective_fixed_
cost); this script only reads frozen instances, drives the solve, and
independently validates and reports the results.

Typical server workflow:

    python run_triangular_table2_experiment.py --self-test
    python run_triangular_table2_experiment.py --preflight --solver BOTH
    python run_triangular_table2_experiment.py --validate-data-only
    python run_triangular_table2_experiment.py --solver CPLEX --n-values 100 --replications 1
    python run_triangular_table2_experiment.py --solver GUROBI --n-values 100 --replications 1
    python run_triangular_table2_experiment.py --solver CPLEX --n-values all --replications all

Use --help for all configuration options.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import datetime as dt
import importlib.metadata
import itertools
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


# ---------------------------------------------------------------------------
# EDITABLE DEFAULT CONFIGURATION
# Command-line arguments override every default below.
# ---------------------------------------------------------------------------

DEFAULT_SOLVER = "GUROBI"                  # CPLEX, MOSEK, GUROBI, BOTH, or ALL
SOLVER_NAMES = ("CPLEX", "MOSEK", "GUROBI")
DEFAULT_TIME_LIMIT_SECONDS = 3600.0       # Table 2: one hour per model
DEFAULT_MIP_GAP = 0.0005                  # Table 2: 0.05 percent
DEFAULT_THREADS = 1                       # Reproducible/fair default
DEFAULT_N_VALUES = "100,200,300,400,500,600,700"
DEFAULT_REPLICATIONS = "1-30"
DEFAULT_MODELS = "both"                  # both, bigm, or perspective

# Tight tolerance for checking solver objective vs. independently recomputed
# economic objective at the returned solution.
DEFAULT_RECOMPUTE_ABS_TOL = 1.0e-5
DEFAULT_RECOMPUTE_REL_TOL = 1.0e-7

# A solver may return sub-milliscale continuous-variable leakage around an
# otherwise integral MISOCP solution. A deterministic repair is permitted
# only below this explicit physical-quantity threshold. Raw values and raw
# residuals are always retained in the append-only result record.
DEFAULT_NUMERICAL_REPAIR_ABS_TOL = 1.0e-3
DEFAULT_NUMERICAL_REPAIR_REL_TOL = 1.0e-8

# The repair must not materially change the objective. This guard is separate
# from both solver MIP gap and model-to-model comparison tolerance.
DEFAULT_REPAIR_OBJECTIVE_ABS_TOL = 1.0e-4
DEFAULT_REPAIR_OBJECTIVE_REL_TOL = 5.0e-6

# Model-to-model optimal-value comparison tolerance. This is intentionally
# separate from the MIP gap and from the objective-recomputation tolerance.
DEFAULT_MODEL_MATCH_ABS_TOL = 1.0e-4
DEFAULT_MODEL_MATCH_REL_TOL = 0.001        # 0.1 percent; twice default MIP gap

DEFAULT_FEASIBILITY_ABS_TOL = 1.0e-6
DEFAULT_FEASIBILITY_REL_TOL = 1.0e-8
DEFAULT_INTEGRALITY_TOL = 1.0e-5

TABLE2_N_VALUES = (100, 200, 300, 400, 500, 600, 700)
TABLE2_REPLICATIONS = tuple(range(1, 31))

BIGM_MODEL_ID = "CURRENT_TRIANGULAR_BIGM_FIXED_COST"
PERSPECTIVE_MODEL_ID = "CURRENT_TRIANGULAR_PERSPECTIVE_FIXED_COST"
# Disabled experimental model ID:
# PERSPECTIVE_WITH_15I_MODEL_ID = "CURRENT_TRIANGULAR_PERSPECTIVE_FIXED_COST_WITH_15I"
MODEL_LABELS = {
    BIGM_MODEL_ID: "Big-M (9)",
    PERSPECTIVE_MODEL_ID: "Perspective",
    # PERSPECTIVE_WITH_15I_MODEL_ID: "Perspective with (15i)",
}

EXPECTED_GENERATION_VERSION = "TABLES_1_3_CORRECTED_STAGE1_V1"
EXPECTED_FAMILY = "TRIANGULAR"
EXPECTED_KAPPA = 50

EXPECTED_CSV_HEADER = (
    "family", "n", "replication", "instance_id", "product_id",
    "master_seed", "instance_seed", "a", "m", "b", "cost_c", "price_p",
    "salvage_s", "price_markup_draw", "salvage_multiplier", "x_u",
    "fixed_cost_score", "k", "kappa", "K", "capacity_fraction", "C",
    "generation_version",
)

RESULT_FORMAT_VERSION = "TRIANGULAR_TABLE2_CVXPY_V1"
RAW_RESULT_FILENAME = "triangular_table2_solver_records.csv"
COMPARISON_FILENAME = "triangular_table2_model_comparison.csv"
SUMMARY_FILENAME = "triangular_table2_runtime_summary.csv"
ENVIRONMENT_FILENAME = "triangular_table2_run_settings.csv"


class Table2Error(RuntimeError):
    """Raised when input, configuration, modeling, or validation fails."""


@dataclass(frozen=True, slots=True)
class TriangularInstance:
    n: int
    replication: int
    instance_id: str
    master_seed: int
    instance_seed: int
    a: tuple[float, ...]
    m: tuple[float, ...]
    b: tuple[float, ...]
    cost_c: tuple[float, ...]
    price_p: tuple[float, ...]
    salvage_s: tuple[float, ...]
    x_u: tuple[float, ...]
    fixed_cost_score: tuple[float, ...]
    k: tuple[float, ...]
    kappa: int
    K: float
    capacity_fraction: float
    C: float
    generation_version: str


@dataclass(frozen=True, slots=True)
class RunSettings:
    solver: str
    time_limit_seconds: float
    mip_gap: float
    threads: int
    recompute_abs_tol: float
    recompute_rel_tol: float
    numerical_repair_abs_tol: float
    numerical_repair_rel_tol: float
    repair_objective_abs_tol: float
    repair_objective_rel_tol: float
    model_match_abs_tol: float
    model_match_rel_tol: float
    feasibility_abs_tol: float
    feasibility_rel_tol: float
    integrality_tol: float
    verbose_solver: bool


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def scaled_tolerance(abs_tol: float, rel_tol: float, *values: float) -> float:
    return abs_tol + rel_tol * max((1.0, *(abs(value) for value in values)))


def close_enough(
    observed: float,
    expected: float,
    *,
    abs_tol: float = 1.0e-9,
    rel_tol: float = 2.0e-12,
) -> bool:
    return abs(observed - expected) <= scaled_tolerance(
        abs_tol, rel_tol, observed, expected
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Table2Error(message)


def parse_int_selection(text: str, allowed: Sequence[int], label: str) -> tuple[int, ...]:
    allowed_set = set(allowed)
    cleaned = text.strip().lower()
    if cleaned == "all":
        return tuple(allowed)
    selected: list[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-", 1)
            try:
                start, stop = int(pieces[0]), int(pieces[1])
            except ValueError as error:
                raise Table2Error(f"invalid {label} range: {token!r}") from error
            require(start <= stop, f"invalid descending {label} range: {token!r}")
            selected.extend(value for value in allowed if start <= value <= stop)
        else:
            try:
                selected.append(int(token))
            except ValueError as error:
                raise Table2Error(f"invalid {label} value: {token!r}") from error
    require(bool(selected), f"no {label} values were selected")
    invalid = sorted(set(selected) - allowed_set)
    require(not invalid, f"unsupported {label} values: {invalid}; allowed={list(allowed)}")
    return tuple(sorted(set(selected)))


def parse_solvers(text: str) -> tuple[str, ...]:
    value = text.strip().upper()
    if value == "BOTH":
        # Legacy alias, kept for backward compatibility with scripts/notes
        # written before GUROBI support was added.
        return ("CPLEX", "MOSEK")
    if value == "ALL":
        return SOLVER_NAMES
    tokens = tuple(sorted({token.strip() for token in value.split(",") if token.strip()}))
    invalid = sorted(set(tokens) - set(SOLVER_NAMES))
    require(
        bool(tokens) and not invalid,
        f"--solver must be one of {', '.join(SOLVER_NAMES)}, a comma-separated combination, BOTH, or ALL",
    )
    return tokens


def parse_models(text: str) -> tuple[str, ...]:
    value = text.strip().lower()
    mapping = {
        "both": (BIGM_MODEL_ID, PERSPECTIVE_MODEL_ID),
        "bigm": (BIGM_MODEL_ID,),
        "perspective": (PERSPECTIVE_MODEL_ID,),
        "conic": (PERSPECTIVE_MODEL_ID,),
        # Disabled experimental selectors:
        # "perspective15i": (PERSPECTIVE_WITH_15I_MODEL_ID,),
        # "perspective_with_15i": (PERSPECTIVE_WITH_15I_MODEL_ID,),
        "all": (BIGM_MODEL_ID, PERSPECTIVE_MODEL_ID),
    }
    require(
        value in mapping,
        "--models must be both, bigm, perspective, conic, or all",
    )
    return mapping[value]


def parse_int(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise Table2Error(f"invalid integer field {field!r}: {row.get(field)!r}") from error


def parse_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise Table2Error(f"invalid numeric field {field!r}: {row.get(field)!r}") from error
    require(math.isfinite(value), f"non-finite numeric field {field!r}: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Closed-form Triangular oracles (manuscript equations (7) and the critical
# fractile x_u). These are reimplemented independently here (rather than
# imported from generate_manuscript_table_instances.py) so this script's own
# validation, at solve time, does not share a bug with the data generator.
# ---------------------------------------------------------------------------

def triangular_cdf(x: float, a: float, m: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x <= m:
        return (x - a) ** 2 / ((b - a) * (m - a))
    if x < b:
        return 1.0 - (b - x) ** 2 / ((b - a) * (b - m))
    return 1.0


def triangular_unconstrained_quantity(
    a: float, m: float, b: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    alpha = (price_p - cost_c) / (price_p - salvage_s)
    mode_cdf = (m - a) / (b - a)
    if alpha <= mode_cdf:
        return a + math.sqrt(alpha * (b - a) * (m - a))
    return b - math.sqrt((1.0 - alpha) * (b - a) * (b - m))


def triangular_segment_quantities(x: float, a: float, m: float, b: float) -> tuple[float, float, float]:
    # The tolerance here is deliberately not machine-epsilon tight: it only
    # needs to reject a genuinely wrong (not just solver-noise-sized)
    # quantity, since the values checked come from a MISOCP solver's
    # returned floating-point solution (GUROBI/MOSEK/CPLEX), not from exact
    # arithmetic -- residuals up to ~1e-6 on larger instances are routine.
    require(-1.0e-6 <= x <= b + 1.0e-6, f"triangular segment quantity must lie in [0,b]: {x!r}")
    x = min(max(x, 0.0), b)
    x_bar = min(x, a)
    x_hat = min(max(x - a, 0.0), m - a)
    x_tilde = min(max(x - m, 0.0), b - m)
    return x_bar, x_hat, x_tilde


def triangular_expected_profit(
    x: float,
    a: float,
    m: float,
    b: float,
    price_p: float,
    cost_c: float,
    salvage_s: float,
) -> float:
    """Manuscript equation (7), evaluated through the equivalent closed
    (non-piecewise) polynomial (8) restricted to x in [0, b].
    """
    # See the matching note in triangular_segment_quantities: this is a
    # solver-noise-scale tolerance, not machine epsilon.
    require(x >= -1.0e-6, f"negative Triangular quantity: {x!r}")
    x = max(0.0, x)
    if x > b:
        # Never used by the Table 2 models (x is bounded by x_u <= b), kept
        # as a defensive full oracle for independent validation only.
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


def _scalar_field(rows: Sequence[Mapping[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    require(len(values) == 1, f"instance rows disagree on scalar field {field!r}")
    return next(iter(values))


def parse_triangular_table2_instance(
    rows: Sequence[Mapping[str, str]]
) -> TriangularInstance:
    require(bool(rows), "cannot build an instance from zero rows")
    first = rows[0]
    n = parse_int(first, "n")
    replication = parse_int(first, "replication")
    instance_id = first["instance_id"]
    require(len(rows) == n, f"{instance_id}: expected {n} rows, observed {len(rows)}")
    require(
        [parse_int(row, "product_id") for row in rows] == list(range(1, n + 1)),
        f"{instance_id}: product_id sequence is not exactly 1..n",
    )

    scalar_fields = (
        "family", "n", "replication", "instance_id", "master_seed",
        "instance_seed", "price_markup_draw", "kappa", "K",
        "capacity_fraction", "C", "generation_version",
    )
    for field in scalar_fields:
        _scalar_field(rows, field)

    require(first["family"] == EXPECTED_FAMILY, f"{instance_id}: family is not TRIANGULAR")
    expected_id = f"TRIANGULAR-N{n:04d}-R{replication:03d}"
    require(instance_id == expected_id, f"unexpected instance_id {instance_id!r}")
    require(
        first["generation_version"] == EXPECTED_GENERATION_VERSION,
        f"{instance_id}: unexpected generation version {first['generation_version']!r}",
    )
    a = tuple(parse_float(row, "a") for row in rows)
    m = tuple(parse_float(row, "m") for row in rows)
    b = tuple(parse_float(row, "b") for row in rows)
    cost = tuple(parse_float(row, "cost_c") for row in rows)
    price = tuple(parse_float(row, "price_p") for row in rows)
    salvage = tuple(parse_float(row, "salvage_s") for row in rows)
    x_u = tuple(parse_float(row, "x_u") for row in rows)
    scores = tuple(parse_float(row, "fixed_cost_score") for row in rows)
    k = tuple(parse_float(row, "k") for row in rows)
    kappa = parse_int(first, "kappa")
    budget_K = parse_float(first, "K")
    capacity_fraction = parse_float(first, "capacity_fraction")
    capacity_C = parse_float(first, "C")

    require(n in TABLE2_N_VALUES, f"{instance_id}: unsupported Table 2 n={n}")
    require(replication in TABLE2_REPLICATIONS, f"{instance_id}: invalid replication")
    require(kappa == EXPECTED_KAPPA, f"{instance_id}: kappa must equal 50")
    require(1.0 / 5.0 <= capacity_fraction <= 1.0 / 3.0, f"{instance_id}: bad C fraction")

    for index, values in enumerate(zip(a, m, b, cost, price, salvage, x_u, scores), start=1):
        ai, mi, bi, ci, pi, si, xui, score = values
        require(0.0 <= ai < mi < bi, f"{instance_id} product {index}: invalid a<m<b")
        require(pi > ci, f"{instance_id} product {index}: price must exceed cost")
        require(pi > si, f"{instance_id} product {index}: price must exceed salvage")
        expected_xu = triangular_unconstrained_quantity(ai, mi, bi, pi, ci, si)
        require(
            close_enough(xui, expected_xu),
            f"{instance_id} product {index}: x_u formula mismatch",
        )
        expected_score = triangular_expected_profit(xui, ai, mi, bi, pi, ci, si)
        require(
            close_enough(score, expected_score),
            f"{instance_id} product {index}: fixed_cost_score mismatch",
        )

    score_shift = max(scores) / 10.0
    for index, (observed, score) in enumerate(zip(k, scores), start=1):
        require(
            close_enough(observed, score + score_shift),
            f"{instance_id} product {index}: k formula mismatch",
        )
    require(
        close_enough(budget_K, kappa / 101.0 * sum(k)),
        f"{instance_id}: K formula mismatch",
    )
    require(
        close_enough(capacity_C, capacity_fraction * sum(x_u)),
        f"{instance_id}: C formula mismatch",
    )

    return TriangularInstance(
        n=n,
        replication=replication,
        instance_id=instance_id,
        master_seed=parse_int(first, "master_seed"),
        instance_seed=parse_int(first, "instance_seed"),
        a=a,
        m=m,
        b=b,
        cost_c=cost,
        price_p=price,
        salvage_s=salvage,
        x_u=x_u,
        fixed_cost_score=scores,
        k=k,
        kappa=kappa,
        K=budget_K,
        capacity_fraction=capacity_fraction,
        C=capacity_C,
        generation_version=first["generation_version"],
    )


def read_selected_triangular_table2_instances(
    csv_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> Iterator[TriangularInstance]:
    requested = {(n, replication) for n in n_values for replication in replications}
    observed: set[tuple[int, int]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            tuple(reader.fieldnames or ()) == EXPECTED_CSV_HEADER,
            "triangular_table2_instances.csv has an unexpected column layout",
        )
        for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
            rows = list(group)
            require(bool(rows), f"empty CSV group for {instance_id}")
            key = (parse_int(rows[0], "n"), parse_int(rows[0], "replication"))
            if key not in requested:
                continue
            require(key not in observed, f"duplicate selected instance group: {key}")
            instance = parse_triangular_table2_instance(rows)
            observed.add(key)
            yield instance
    missing = sorted(requested - observed)
    require(not missing, f"Table 2 CSV is missing selected instances: {missing}")


def import_optimization_stack() -> tuple[Any, Any, Any]:
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError as error:
        raise Table2Error(
            "CVXPY and NumPy are required on the server. Install them in the server "
            "environment before using solver or preflight modes."
        ) from error
    try:
        import triangular_current as tc
    except ImportError as error:
        raise Table2Error(
            "triangular_current.py (this script's model-building module) could not "
            "be imported; make sure it is on sys.path alongside this script."
        ) from error
    require(tc.BIGM_MODEL_ID == BIGM_MODEL_ID, "triangular_current BIGM_MODEL_ID drifted")
    require(
        tc.PERSPECTIVE_MODEL_ID == PERSPECTIVE_MODEL_ID,
        "triangular_current PERSPECTIVE_MODEL_ID drifted",
    )
    # Disabled perspective-with-(15i) API drift check:
    # require(
    #     tc.PERSPECTIVE_WITH_15I_MODEL_ID == PERSPECTIVE_WITH_15I_MODEL_ID,
    #     "triangular_current PERSPECTIVE_WITH_15I_MODEL_ID drifted",
    # )
    return cp, np, tc


def build_current_manuscript_triangular_model(
    instance: TriangularInstance, model_id: str
) -> Any:
    _, _, tc = import_optimization_stack()
    require(model_id in MODEL_LABELS, f"unsupported model_id: {model_id}")
    builders = {
        BIGM_MODEL_ID: tc.build_current_triangular_bigm_fixed_cost,
        PERSPECTIVE_MODEL_ID: tc.build_current_triangular_perspective_fixed_cost,
        # PERSPECTIVE_WITH_15I_MODEL_ID:
        #     tc.build_current_triangular_perspective_fixed_cost_with_15i,
    }
    builder = builders[model_id]
    bundle = builder(
        instance.a, instance.m, instance.b,
        instance.price_p, instance.cost_c, instance.salvage_s,
        instance.x_u, instance.k, instance.C, instance.K,
    )
    require(bundle.problem.is_dcp(), f"{model_id}: CVXPY model is not DCP")
    require(bundle.problem.is_mixed_integer(), f"{model_id}: CVXPY model is not mixed-integer")
    return bundle


def solver_options(settings: RunSettings) -> dict[str, Any]:
    _, _, tc = import_optimization_stack()
    return tc.solver_options(
        settings.solver,
        time_limit_seconds=settings.time_limit_seconds,
        mip_gap=settings.mip_gap,
        threads=settings.threads,
    )


def _safe_call(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except Exception:
        return default


def _find_solver_object(value: Any, *, max_depth: int = 3) -> Any:
    if value is None or max_depth < 0:
        return None
    if hasattr(value, "solution") and hasattr(value, "objective"):
        return value
    if isinstance(value, Mapping):
        for key in ("model", "cplex", "solver", "task"):
            if key in value:
                found = _find_solver_object(value[key], max_depth=max_depth - 1)
                if found is not None:
                    return found
        for child in value.values():
            found = _find_solver_object(child, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def extract_native_stats(raw_solution: Any, solver: str) -> dict[str, Any]:
    """Extract native MIP statistics before CVXPY releases solver objects."""
    stats: dict[str, Any] = {
        "native_incumbent_objective": None,
        "native_best_bound": None,
        "native_relative_gap": None,
        "node_count": None,
        "nodes_remaining": None,
        "native_status_text": None,
        "node_count_source": None,
    }
    if solver == "CPLEX":
        model = _find_solver_object(raw_solution)
        if model is None:
            return stats
        solution = getattr(model, "solution", None)
        if solution is None:
            return stats
        stats["native_status_text"] = _safe_call(solution.get_status_string)
        progress = getattr(solution, "progress", None)
        if progress is not None:
            nodes = _safe_call(progress.get_num_nodes_processed)
            if nodes is not None and math.isfinite(float(nodes)):
                stats["node_count"] = int(round(float(nodes)))
                stats["node_count_source"] = "CPLEX.solution.progress.get_num_nodes_processed"
            remaining = _safe_call(progress.get_num_nodes_remaining)
            if remaining is not None and math.isfinite(float(remaining)):
                stats["nodes_remaining"] = int(round(float(remaining)))
        mip = getattr(solution, "MIP", None)
        if mip is not None:
            stats["native_best_bound"] = _safe_call(mip.get_best_objective)
            stats["native_relative_gap"] = _safe_call(mip.get_mip_relative_gap)
        stats["native_incumbent_objective"] = _safe_call(solution.get_objective_value)
        return stats

    if solver == "MOSEK":
        task = raw_solution.get("task") if isinstance(raw_solution, Mapping) else None
        if task is None:
            return stats
        try:
            import mosek
        except ImportError:
            return stats
        trm = _safe_call(getattr(task, "gettrmcode", lambda: None))
        stats["native_status_text"] = None if trm is None else str(trm)
        for enum_name in ("mio_num_branch", "mio_num_relax"):
            enum = getattr(mosek.iinfitem, enum_name, None)
            if enum is None:
                continue
            nodes = _safe_call(lambda enum=enum: task.getintinf(enum))
            if nodes is not None and math.isfinite(float(nodes)):
                stats["node_count"] = int(round(float(nodes)))
                stats["node_count_source"] = f"MOSEK.iinfitem.{enum_name}"
                break
        for output_key, enum_name in (
            ("native_incumbent_objective", "mio_obj_int"),
            ("native_best_bound", "mio_obj_bound"),
            ("native_relative_gap", "mio_obj_rel_gap"),
        ):
            enum = getattr(mosek.dinfitem, enum_name, None)
            if enum is not None:
                stats[output_key] = _safe_call(lambda enum=enum: task.getdouinf(enum))
        return stats

    if solver == "GUROBI":
        # CVXPY's GUROBI interface stores the raw gurobipy Model directly
        # under the "model" key (see cvxpy.reductions.solvers.conic_solvers.
        # gurobi_conif); it exposes plain attributes rather than the
        # method-based introspection CPLEX/MOSEK use above.
        model = raw_solution.get("model") if isinstance(raw_solution, Mapping) else None
        if model is None:
            return stats
        # Gurobi's own status is a bare numeric code (GRB.Status.*), unlike
        # MOSEK's/CPLEX's descriptive strings; classify_termination()
        # recognizes a time-limited run by looking for "time"+"limit" in
        # this text, so translate the code to a matching name here rather
        # than special-casing GUROBI in the generic classifier. In
        # particular, code 9 (TIME_LIMIT) covers exactly the "no incumbent
        # found before the time budget ran out" case CVXPY otherwise reports
        # as the misleading "infeasible_inaccurate" status.
        gurobi_status_names = {
            1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
            5: "UNBOUNDED", 6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT",
            9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
            12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
            16: "WORK_LIMIT", 17: "MEM_LIMIT",
        }
        status_code = _safe_call(lambda: int(model.Status))
        stats["native_status_text"] = gurobi_status_names.get(status_code, str(status_code))
        node_count = _safe_call(lambda: model.NodeCount)
        if node_count is not None and math.isfinite(float(node_count)):
            stats["node_count"] = int(round(float(node_count)))
            stats["node_count_source"] = "GUROBI.Model.NodeCount"
        stats["native_incumbent_objective"] = _safe_call(lambda: model.ObjVal)
        stats["native_best_bound"] = _safe_call(lambda: model.ObjBound)
        stats["native_relative_gap"] = _safe_call(lambda: model.MIPGap)
        return stats
    return stats


def normalize_native_objectives(stats: dict[str, Any], cvxpy_objective: float | None) -> None:
    incumbent = stats.get("native_incumbent_objective")
    sign = -1.0
    if cvxpy_objective is not None and incumbent is not None and math.isfinite(float(incumbent)):
        incumbent_float = float(incumbent)
        sign = (
            -1.0
            if abs(-incumbent_float - cvxpy_objective) < abs(incumbent_float - cvxpy_objective)
            else 1.0
        )
    bound = stats.get("native_best_bound")
    stats["objective_orientation_sign"] = sign
    stats["best_bound"] = (
        None if bound is None or not math.isfinite(float(bound)) else sign * float(bound)
    )


def finite_vector(value: Any, expected_size: int) -> list[float] | None:
    if value is None:
        return None
    try:
        flattened = value.reshape(-1).tolist()
    except Exception:
        try:
            flattened = list(value)
        except Exception:
            return None
    if len(flattened) != expected_size:
        return None
    result = [float(item) for item in flattened]
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def recompute_triangular_expected_profit(
    instance: TriangularInstance, q: Sequence[float]
) -> float:
    return sum(
        triangular_expected_profit(quantity, a, m, b, p, c, s)
        for quantity, a, m, b, p, c, s in zip(
            q,
            instance.a,
            instance.m,
            instance.b,
            instance.price_p,
            instance.cost_c,
            instance.salvage_s,
            strict=True,
        )
    )


def lifted_triangular_objective(
    instance: TriangularInstance,
    solution: Mapping[str, Sequence[float]],
) -> float:
    """Manuscript equation (10)/(9a), evaluated directly at the returned lift
    variables (bar_x, hat_x, y, w_hat, w_y) -- independent of CVXPY's own
    objective expression evaluation.
    """
    bar = solution["bar_x"]
    hat = solution["hat_x"]
    y = solution["y"]
    w_hat = solution["w_hat"]
    w_y = solution["w_y"]
    total = 0.0
    for i in range(instance.n):
        a, m, b = instance.a[i], instance.m[i], instance.b[i]
        p, c, s = instance.price_p[i], instance.cost_c[i], instance.salvage_s[i]
        coeff_q = p - c
        coeff_y = -(s - c)
        coeff3 = (s - p) / (3.0 * (b - a))
        wy_coeff = coeff3 / (b - m)
        wh_coeff = coeff3 / (m - a)
        const_term = (b - m) * (p - c - (s - p) * (3.0 * a - 2.0 * b - m) / (3.0 * (b - a)))
        total += (
            coeff_q * (bar[i] + hat[i])
            + coeff_y * y[i]
            + wy_coeff * w_y[i]
            + wh_coeff * w_hat[i]
            + const_term
        )
    return total


def nonnegative_violation(values: Sequence[float]) -> float:
    return max((max(0.0, -value) for value in values), default=0.0)


def validate_solution(
    instance: TriangularInstance,
    model_id: str,
    solution: Mapping[str, Sequence[float]],
    reported_objective: float,
    settings: RunSettings,
) -> tuple[str, dict[str, Any], float, float]:
    bar = list(solution["bar_x"])
    hat = list(solution["hat_x"])
    y = list(solution["y"])
    u_hat = list(solution["u_hat"])
    w_hat = list(solution["w_hat"])
    u_y = list(solution["u_y"])
    w_y = list(solution["w_y"])
    z = list(solution["z"])
    tilde = [instance.b[i] - instance.m[i] - y[i] for i in range(instance.n)]
    q = [bar[i] + hat[i] + tilde[i] for i in range(instance.n)]

    common_objective = recompute_triangular_expected_profit(instance, q)
    lifted_objective = lifted_triangular_objective(instance, solution)
    reconstruction_abs = abs(reported_objective - lifted_objective)
    reconstruction_rel = reconstruction_abs / max(
        1.0, abs(reported_objective), abs(lifted_objective)
    )
    economic_abs = abs(reported_objective - common_objective)
    economic_rel = economic_abs / max(
        1.0, abs(reported_objective), abs(common_objective)
    )

    max_domain = max(
        nonnegative_violation(bar),
        nonnegative_violation(hat),
        nonnegative_violation(y),
        nonnegative_violation(u_hat),
        nonnegative_violation(w_hat),
        nonnegative_violation(u_y),
        nonnegative_violation(w_y),
        nonnegative_violation(tilde),
        max((max(0.0, -value, value - 1.0) for value in z), default=0.0),
    )
    hat_bigm = [max(0.0, min(instance.x_u[i], instance.m[i]) - instance.a[i]) for i in range(instance.n)]
    tilde_bigm = [max(0.0, instance.x_u[i] - instance.m[i]) for i in range(instance.n)]
    bar_link = max(
        (max(0.0, bar[i] - instance.a[i] * z[i]) for i in range(instance.n)),
        default=0.0,
    )
    hat_link = max(
        (max(0.0, hat[i] - hat_bigm[i] * z[i]) for i in range(instance.n)),
        default=0.0,
    )
    tilde_link = max(
        (max(0.0, tilde[i] - tilde_bigm[i] * z[i]) for i in range(instance.n)),
        default=0.0,
    )
    y_upper_violation = max(
        (max(0.0, y[i] - (instance.b[i] - instance.m[i])) for i in range(instance.n)),
        default=0.0,
    )
    capacity_violation = max(0.0, sum(q) - instance.C)
    budget_violation = max(0.0, sum(k * zi for k, zi in zip(instance.k, z)) - instance.K)
    max_linear = max(bar_link, hat_link, tilde_link, y_upper_violation, capacity_violation, budget_violation)

    # The four hyperbolic/cube-tower constraints span wildly different
    # magnitudes within a single instance (hat_x ~ O(1e2) but w_hat = hat_x^3
    # can reach O(1e7)), so each is checked against its own scale via
    # scaled_tolerance rather than against one global bound tied to C/K/x_u
    # (which would be far tighter than any conic solver's own primal
    # feasibility tolerance on the cubic-scale variables, and would reject
    # solutions that are numerically fine).
    multiplier = (lambda i: 1.0) if model_id == BIGM_MODEL_ID else (lambda i: z[i])
    soc_ok = True
    max_soc = 0.0
    max_soc_relative = 0.0
    for i in range(instance.n):
        for lhs, rhs in (
            (hat[i] ** 2, u_hat[i] * multiplier(i)),
            (u_hat[i] ** 2, w_hat[i] * hat[i]),
            (y[i] ** 2, u_y[i]),
            (u_y[i] ** 2, w_y[i] * y[i]),
        ):
            violation = max(0.0, lhs - rhs)
            max_soc = max(max_soc, violation)
            max_soc_relative = max(
                max_soc_relative, violation / max(1.0, abs(lhs), abs(rhs))
            )
            limit = scaled_tolerance(settings.feasibility_abs_tol, settings.feasibility_rel_tol, lhs, rhs)
            if violation > limit:
                soc_ok = False
    max_integrality = max((min(abs(value), abs(1.0 - value)) for value in z), default=0.0)

    reconstruction_ok = reconstruction_abs <= scaled_tolerance(
        settings.recompute_abs_tol,
        settings.recompute_rel_tol,
        reported_objective,
        lifted_objective,
    )
    economic_ok = economic_abs <= scaled_tolerance(
        settings.recompute_abs_tol,
        settings.recompute_rel_tol,
        reported_objective,
        common_objective,
    )
    feasibility_scale = max(1.0, instance.C, instance.K, max(instance.x_u))
    feasibility_limit = settings.feasibility_abs_tol + settings.feasibility_rel_tol * feasibility_scale
    feasibility_ok = max(max_domain, max_linear) <= feasibility_limit and soc_ok
    integrality_ok = max_integrality <= settings.integrality_tol
    objective_ok = reconstruction_ok and economic_ok
    status = "PASS" if objective_ok and feasibility_ok and integrality_ok else "FAIL"
    residuals = {
        "objective_expression_reconstruction_absolute": reconstruction_abs,
        "objective_expression_reconstruction_relative": reconstruction_rel,
        "objective_economic_consistency_absolute": economic_abs,
        "objective_economic_consistency_relative": economic_rel,
        "max_soc_violation": max_soc,
        "max_soc_relative_violation": max_soc_relative,
        "soc_check_pass": soc_ok,
        "max_linear_violation": max_linear,
        "max_domain_violation": max_domain,
        "max_integrality_violation": max_integrality,
        "capacity_violation": capacity_violation,
        "fixed_cost_budget_violation": budget_violation,
        "bar_link_violation": bar_link,
        "hat_link_violation": hat_link,
        "tilde_link_violation": tilde_link,
        "y_upper_bound_violation": y_upper_violation,
        "feasibility_limit": feasibility_limit,
        "objective_expression_reconstruction_pass": reconstruction_ok,
        "objective_economic_consistency_pass": economic_ok,
        "objective_check_pass": objective_ok,
        "feasibility_check_pass": feasibility_ok,
        "integrality_check_pass": integrality_ok,
    }
    return status, residuals, common_objective, lifted_objective


def repair_numerical_solution(
    instance: TriangularInstance,
    raw_solution: Mapping[str, Sequence[float]],
    settings: RunSettings,
) -> tuple[dict[str, list[float]] | None, dict[str, Any]]:
    """Project only sub-tolerance numerical leakage to an exact physical point.

    The repair never changes a binary decision. It rounds an already-integral
    z, removes only sub-tolerance physical quantity from inactive products,
    clips only sub-tolerance quantity bound violations, reconstructs the
    canonical Triangular (bar, hat, tilde) split from the resulting q, and
    makes every square/cube lift tight. Raw values remain in the result
    record regardless of whether repair is allowed.
    """
    bar = list(raw_solution["bar_x"])
    hat = list(raw_solution["hat_x"])
    y = list(raw_solution["y"])
    z = list(raw_solution["z"])
    quantity_scale = max(1.0, max(instance.x_u))
    quantity_limit = (
        settings.numerical_repair_abs_tol
        + settings.numerical_repair_rel_tol * quantity_scale
    )

    repaired_bar: list[float] = []
    repaired_hat: list[float] = []
    repaired_y: list[float] = []
    repaired_u_hat: list[float] = []
    repaired_w_hat: list[float] = []
    repaired_u_y: list[float] = []
    repaired_w_y: list[float] = []
    repaired_z: list[float] = []
    repaired_q: list[float] = []
    reasons: list[str] = []
    max_z_adjustment = 0.0
    max_quantity_adjustment = 0.0
    max_inactive_quantity_removed = 0.0
    total_inactive_quantity_removed = 0.0

    for i in range(instance.n):
        zi = z[i]
        rounded_z = 0.0 if zi < 0.5 else 1.0
        z_adjustment = abs(zi - rounded_z)
        max_z_adjustment = max(max_z_adjustment, z_adjustment)
        if z_adjustment > settings.integrality_tol:
            reasons.append(
                f"product {i + 1}: binary adjustment {z_adjustment} exceeds integrality tolerance"
            )

        tilde_i = instance.b[i] - instance.m[i] - y[i]
        raw_q = bar[i] + hat[i] + tilde_i
        if rounded_z == 0.0:
            quantity_adjustment = abs(raw_q)
            if quantity_adjustment > quantity_limit:
                reasons.append(
                    f"product {i + 1}: inactive quantity {raw_q} exceeds repair limit {quantity_limit}"
                )
            target_q = 0.0
            max_inactive_quantity_removed = max(
                max_inactive_quantity_removed, quantity_adjustment
            )
            total_inactive_quantity_removed += quantity_adjustment
        else:
            target_q = min(max(raw_q, 0.0), instance.x_u[i])
            quantity_adjustment = abs(target_q - raw_q)
            if quantity_adjustment > quantity_limit:
                reasons.append(
                    f"product {i + 1}: active quantity adjustment {quantity_adjustment} "
                    f"exceeds repair limit {quantity_limit}"
                )

        canonical_bar = min(target_q, instance.a[i])
        canonical_hat = min(max(target_q - instance.a[i], 0.0), instance.m[i] - instance.a[i])
        canonical_tilde = min(max(target_q - instance.m[i], 0.0), instance.b[i] - instance.m[i])
        canonical_y = instance.b[i] - instance.m[i] - canonical_tilde
        max_quantity_adjustment = max(max_quantity_adjustment, quantity_adjustment)
        repaired_bar.append(canonical_bar)
        repaired_hat.append(canonical_hat)
        repaired_y.append(canonical_y)
        repaired_u_hat.append(canonical_hat * canonical_hat)
        repaired_w_hat.append(canonical_hat ** 3)
        repaired_u_y.append(canonical_y * canonical_y)
        repaired_w_y.append(canonical_y ** 3)
        repaired_z.append(rounded_z)
        repaired_q.append(target_q)

    feasibility_scale = max(1.0, instance.C, instance.K, max(instance.x_u))
    feasibility_limit = (
        settings.feasibility_abs_tol
        + settings.feasibility_rel_tol * feasibility_scale
    )
    capacity_violation = max(0.0, sum(repaired_q) - instance.C)
    budget_violation = max(
        0.0,
        sum(k * zi for k, zi in zip(instance.k, repaired_z, strict=True))
        - instance.K,
    )
    if max(capacity_violation, budget_violation) > feasibility_limit:
        reasons.append(
            "repaired point violates capacity or fixed-cost budget beyond feasibility tolerance"
        )

    metrics = {
        "repair_allowed": not reasons,
        "repair_reasons": reasons,
        "quantity_repair_limit": quantity_limit,
        "max_z_adjustment": max_z_adjustment,
        "max_physical_quantity_adjustment": max_quantity_adjustment,
        "max_inactive_quantity_removed": max_inactive_quantity_removed,
        "total_inactive_quantity_removed": total_inactive_quantity_removed,
        "repaired_capacity_violation": capacity_violation,
        "repaired_fixed_cost_budget_violation": budget_violation,
    }
    if reasons:
        return None, metrics
    return {
        "bar_x": repaired_bar,
        "hat_x": repaired_hat,
        "y": repaired_y,
        "u_hat": repaired_u_hat,
        "w_hat": repaired_w_hat,
        "u_y": repaired_u_y,
        "w_y": repaired_w_y,
        "z": repaired_z,
        "q": repaired_q,
    }, metrics


def classify_termination(
    cvxpy_status: str | None,
    native_status_text: str | None,
    has_incumbent: bool,
    *,
    time_limit_seconds: float,
    observed_solve_seconds: float,
) -> tuple[str, str, str]:
    status = (cvxpy_status or "unknown").lower()
    native = (native_status_text or "").lower()
    near_configured_limit = observed_solve_seconds >= 0.98 * time_limit_seconds
    time_limited = "time" in native and "limit" in native
    time_limited = (
        time_limited
        or "trm_max_time" in native
        or status == "user_limit"
        or (status in {"optimal_inaccurate", "unknown"} and near_configured_limit)
    )

    if time_limited:
        return (
            "FEASIBLE" if has_incumbent else "NO_SOLUTION",
            "TIME_LIMIT",
            "FEASIBLE_POINT" if has_incumbent else "NO_SOLUTION",
        )
    if status == "optimal":
        return "OPTIMAL", "OPTIMALITY_PROVEN", "FEASIBLE_POINT"
    if status == "optimal_inaccurate":
        return (
            "FEASIBLE" if has_incumbent else "UNKNOWN",
            "OTHER",
            "FEASIBLE_POINT" if has_incumbent else "UNKNOWN",
        )
    if status == "infeasible":
        return "NO_SOLUTION", "INFEASIBLE", "NO_SOLUTION"
    if status == "unbounded":
        return "NO_SOLUTION", "UNBOUNDED", "NO_SOLUTION"
    if status in {"infeasible_or_unbounded", "infeasible_inaccurate", "unbounded_inaccurate"}:
        return "NO_SOLUTION", "INFEASIBLE_OR_UNBOUNDED", "NO_SOLUTION"
    if has_incumbent:
        return "FEASIBLE", "OTHER", "FEASIBLE_POINT"
    return "UNKNOWN", "OTHER", "UNKNOWN"


def solve_triangular_table2_formulation(
    instance: TriangularInstance,
    model_id: str,
    settings: RunSettings,
    *,
    input_path: Path,
    invocation_id: str,
) -> dict[str, Any]:
    cp, _, _ = import_optimization_stack()
    started_at = utc_now()
    build_start = time.perf_counter()
    bundle = build_current_manuscript_triangular_model(instance, model_id)
    build_seconds = time.perf_counter() - build_start

    compile_start = time.perf_counter()
    data, chain, inverse_data = bundle.problem.get_problem_data(solver=settings.solver)
    canonicalization_seconds = time.perf_counter() - compile_start

    native_stats: dict[str, Any] = {}
    raw_solution: Any = None
    solve_exception: str | None = None
    solve_traceback: str | None = None
    solve_start = time.perf_counter()
    try:
        raw_solution = chain.solve_via_data(
            bundle.problem,
            data,
            warm_start=False,
            verbose=settings.verbose_solver,
            solver_opts=solver_options(settings),
        )
        native_stats = extract_native_stats(raw_solution, settings.solver)
        bundle.problem.unpack_results(raw_solution, chain, inverse_data)
    except Exception as error:
        solve_exception = f"{type(error).__name__}: {error}"
        solve_traceback = traceback.format_exc(limit=20)
    wall_solve_seconds = time.perf_counter() - solve_start
    finished_at = utc_now()

    cvxpy_status = None if solve_exception else str(bundle.problem.status)
    solver_stats = None if solve_exception else bundle.problem.solver_stats
    solver_time = None
    setup_time = None
    num_iters = None
    solver_name_observed = settings.solver
    if solver_stats is not None:
        solver_time = solver_stats.solve_time
        setup_time = solver_stats.setup_time
        num_iters = solver_stats.num_iters
        solver_name_observed = str(solver_stats.solver_name or settings.solver)

    vector_names = ("bar_x", "hat_x", "y", "u_hat", "w_hat", "u_y", "w_y", "z")
    vectors = {
        name: finite_vector(bundle.variables[name].value, instance.n)
        for name in vector_names
    }
    has_incumbent = solve_exception is None and all(vectors[name] is not None for name in vector_names)
    reported_objective = None
    if has_incumbent and bundle.problem.value is not None:
        candidate = float(bundle.problem.value)
        if math.isfinite(candidate):
            reported_objective = candidate
        else:
            has_incumbent = False

    solver_status, termination_reason, primal_status = classify_termination(
        cvxpy_status,
        native_stats.get("native_status_text"),
        has_incumbent,
        time_limit_seconds=settings.time_limit_seconds,
        observed_solve_seconds=wall_solve_seconds,
    )
    if solve_exception is not None:
        if wall_solve_seconds >= 0.98 * settings.time_limit_seconds:
            solver_status, termination_reason, primal_status = (
                "NO_SOLUTION", "TIME_LIMIT", "NO_SOLUTION"
            )
        else:
            solver_status, termination_reason, primal_status = (
                "ERROR", "SOLVER_ERROR", "NO_SOLUTION"
            )
        has_incumbent = False
        reported_objective = None

    raw_solution_payload: dict[str, list[float]] | None = None
    repaired_solution_payload: dict[str, list[float]] | None = None
    raw_validation_residuals: dict[str, Any] | None = None
    repaired_validation_residuals: dict[str, Any] | None = None
    repair_metrics: dict[str, Any] | None = None
    raw_independent_objective = None
    raw_lifted_objective = None
    repaired_independent_objective = None
    repaired_lifted_objective = None
    independent_objective = None

    if has_incumbent and reported_objective is not None:
        normalize_native_objectives(native_stats, reported_objective)
        typed_vectors = {name: vectors[name] for name in vector_names}
        assert all(value is not None for value in typed_vectors.values())
        concrete_vectors = {name: value for name, value in typed_vectors.items() if value is not None}
        tilde = [
            instance.b[i] - instance.m[i] - concrete_vectors["y"][i] for i in range(instance.n)
        ]
        q = [
            concrete_vectors["bar_x"][i] + concrete_vectors["hat_x"][i] + tilde[i]
            for i in range(instance.n)
        ]
        raw_solution_payload = {**concrete_vectors, "q": q}
        (
            raw_validation_status,
            raw_validation_residuals,
            raw_independent_objective,
            raw_lifted_objective,
        ) = validate_solution(
            instance, model_id, raw_solution_payload, reported_objective, settings
        )
        repaired_solution_payload, repair_metrics = repair_numerical_solution(
            instance, raw_solution_payload, settings
        )
        repaired_validation_status = "NOT_RUN"
        repair_objective_ok = False
        if repaired_solution_payload is not None:
            repaired_lifted_objective = lifted_triangular_objective(
                instance, repaired_solution_payload
            )
            (
                repaired_validation_status,
                repaired_validation_residuals,
                repaired_independent_objective,
                reconstructed_repaired_lifted,
            ) = validate_solution(
                instance,
                model_id,
                repaired_solution_payload,
                repaired_lifted_objective,
                settings,
            )
            require(
                close_enough(
                    reconstructed_repaired_lifted,
                    repaired_lifted_objective,
                    abs_tol=1.0e-9,
                    rel_tol=1.0e-12,
                ),
                "internal repaired-objective reconstruction mismatch",
            )
            repair_objective_absolute = abs(
                repaired_lifted_objective - reported_objective
            )
            repair_objective_relative = repair_objective_absolute / max(
                1.0, abs(repaired_lifted_objective), abs(reported_objective)
            )
            repair_objective_limit = scaled_tolerance(
                settings.repair_objective_abs_tol,
                settings.repair_objective_rel_tol,
                repaired_lifted_objective,
                reported_objective,
            )
            repair_objective_ok = repair_objective_absolute <= repair_objective_limit
            repair_metrics.update({
                "objective_change_absolute": repair_objective_absolute,
                "objective_change_relative": repair_objective_relative,
                "objective_change_limit": repair_objective_limit,
                "objective_change_pass": repair_objective_ok,
            })
        elif repair_metrics is not None:
            repair_metrics.update({
                "objective_change_absolute": None,
                "objective_change_relative": None,
                "objective_change_limit": None,
                "objective_change_pass": False,
            })

        if raw_validation_status == "PASS":
            validation_status = "RAW_PASS"
            independent_objective = raw_independent_objective
        elif (
            repaired_solution_payload is not None
            and repaired_validation_status == "PASS"
            and repair_objective_ok
            and raw_validation_residuals["objective_expression_reconstruction_pass"]
        ):
            validation_status = "NUMERICALLY_REPAIRED_PASS"
            independent_objective = repaired_independent_objective
        else:
            validation_status = "FAIL"
    else:
        normalize_native_objectives(native_stats, None)
        validation_status = "NOT_RUN"

    relative_gap = native_stats.get("native_relative_gap")
    if relative_gap is not None:
        try:
            relative_gap = abs(float(relative_gap))
            if not math.isfinite(relative_gap):
                relative_gap = None
        except (TypeError, ValueError):
            relative_gap = None
    best_bound = native_stats.get("best_bound")
    absolute_gap = None
    if reported_objective is not None and best_bound is not None:
        absolute_gap = abs(float(best_bound) - reported_objective)

    run_settings = dataclasses.asdict(settings)
    record = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "record_id": str(uuid.uuid4()),
        "invocation_id": invocation_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "input_file": str(input_path.resolve()),
        "run_settings": run_settings,
        "generation_version": instance.generation_version,
        "family": EXPECTED_FAMILY,
        "n": instance.n,
        "replication": instance.replication,
        "instance_id": instance.instance_id,
        "instance_seed": instance.instance_seed,
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
        "solver_requested": settings.solver,
        "solver_name": solver_name_observed,
        "cvxpy_version": cp.__version__,
        "solver_status_raw_cvxpy": cvxpy_status,
        "solver_status": solver_status,
        "termination_reason": termination_reason,
        "primal_status": primal_status,
        "has_incumbent": has_incumbent,
        "objective_solver_reported": reported_objective,
        "objective_independent_common": independent_objective,
        "objective_independent_raw_q": raw_independent_objective,
        "objective_lifted_reconstructed_raw": raw_lifted_objective,
        "objective_independent_repaired": repaired_independent_objective,
        "objective_lifted_repaired": repaired_lifted_objective,
        "best_bound": best_bound,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "node_count": native_stats.get("node_count"),
        "nodes_remaining": native_stats.get("nodes_remaining"),
        "node_count_source": native_stats.get("node_count_source"),
        "model_build_seconds": build_seconds,
        "canonicalization_seconds": canonicalization_seconds,
        "solver_time_seconds": None if solver_time is None else float(solver_time),
        "solver_setup_time_seconds": None if setup_time is None else float(setup_time),
        "wall_solve_and_unpack_seconds": wall_solve_seconds,
        "solver_num_iters_raw": None if num_iters is None else int(num_iters),
        "time_limit_seconds": settings.time_limit_seconds,
        "mip_gap_target": settings.mip_gap,
        "threads": settings.threads,
        "validation_status": validation_status,
        "validation_residuals": raw_validation_residuals,
        "repaired_validation_residuals": repaired_validation_residuals,
        "numerical_repair": repair_metrics,
        "solution": raw_solution_payload,
        "repaired_solution": repaired_solution_payload,
        "native_statistics": native_stats,
        "exception": solve_exception,
        "traceback": solve_traceback,
    }
    return record


def run_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("solver_requested"),
        record.get("model_id"),
        record.get("instance_id"),
        tuple(sorted((record.get("run_settings") or {}).items())),
    )


def read_solver_records_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            record: dict[str, Any] = {}
            for field, encoded in row.items():
                try:
                    record[field] = ast.literal_eval(encoded)
                except (SyntaxError, ValueError) as error:
                    raise Table2Error(
                        f"invalid solver-record value at {path}:{row_number}:{field}"
                    ) from error
            records.append(record)
    return records


def _sanitize_for_repr_roundtrip(value: Any) -> Any:
    """Replace non-finite floats (nan/inf/-inf) with None, recursively.

    A native solver stat (e.g. GUROBI's Model.MIPGap when there is no
    incumbent) can be nan/inf. repr() renders those as the bare tokens
    ``nan``/``inf``, which are not valid Python literals, so
    ast.literal_eval -- used by read_solver_records_csv to parse this same
    CSV back in on a later run -- raises on them. None round-trips safely
    and is the correct value anyway: a non-finite "statistic" carries no
    usable information.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _sanitize_for_repr_roundtrip(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        sanitized = (_sanitize_for_repr_roundtrip(item) for item in value)
        return type(value)(sanitized)
    return value


def append_solver_record_csv(path: Path, record: Mapping[str, Any]) -> None:
    fields = tuple(record)
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as stream:
            observed_fields = tuple(csv.DictReader(stream).fieldnames or ())
        require(observed_fields == fields, "solver-record CSV columns changed")
    sanitized = {field: _sanitize_for_repr_roundtrip(record[field]) for field in fields}
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: repr(sanitized[field]) for field in fields})
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def latest_matching_records(
    records: Sequence[Mapping[str, Any]],
    *,
    settings_by_solver: Mapping[str, RunSettings],
) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        solver = record.get("solver_requested")
        settings = settings_by_solver.get(str(solver))
        if settings is None:
            continue
        if record.get("run_settings") != dataclasses.asdict(settings):
            continue
        latest[run_signature(record)] = record
    return sorted(
        latest.values(),
        key=lambda row: (
            row["solver_requested"], row["n"], row["replication"], row["model_id"]
        ),
    )


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def compare_triangular_table2_formulations(
    records: Sequence[Mapping[str, Any]],
    settings_by_solver: Mapping[str, RunSettings],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["solver_requested"]), str(record["instance_id"]))][
            str(record["model_id"])
        ] = record

    rows: list[dict[str, Any]] = []
    for (solver, instance_id), by_model in sorted(grouped.items()):
        if BIGM_MODEL_ID not in by_model or PERSPECTIVE_MODEL_ID not in by_model:
            continue
        bigm = by_model[BIGM_MODEL_ID]
        perspective = by_model[PERSPECTIVE_MODEL_ID]
        settings = settings_by_solver[solver]
        status_comparable = (
            bigm["solver_status"] == "OPTIMAL"
            and perspective["solver_status"] == "OPTIMAL"
        )
        accepted_validation_statuses = {"RAW_PASS", "NUMERICALLY_REPAIRED_PASS"}
        validation_comparable = (
            bigm["validation_status"] in accepted_validation_statuses
            and perspective["validation_status"] in accepted_validation_statuses
        )
        left = bigm.get("objective_independent_common")
        right = perspective.get("objective_independent_common")
        objectives_available = (
            validation_comparable and left is not None and right is not None
        )
        if objectives_available:
            absolute = abs(float(left) - float(right))
            relative = relative_difference(float(left), float(right))
        else:
            absolute = relative = None

        if not status_comparable:
            classification = "NOT_COMPARABLE_DUE_TO_STATUS"
        elif not objectives_available:
            classification = "NOT_COMPARABLE_DUE_TO_VALIDATION"
        else:
            threshold = scaled_tolerance(
                settings.model_match_abs_tol,
                settings.model_match_rel_tol,
                float(left),
                float(right),
            )
            classification = "PASS" if absolute <= threshold else "FAIL"

        rows.append({
            "solver": solver,
            "n": bigm["n"],
            "replication": bigm["replication"],
            "instance_id": instance_id,
            "bigm_record_id": bigm["record_id"],
            "perspective_record_id": perspective["record_id"],
            "bigm_solver_status": bigm["solver_status"],
            "perspective_solver_status": perspective["solver_status"],
            "bigm_validation_status": bigm["validation_status"],
            "perspective_validation_status": perspective["validation_status"],
            "bigm_common_objective": left,
            "perspective_common_objective": right,
            "objective_absolute_difference": absolute,
            "objective_relative_difference": relative,
            "model_match_abs_tolerance": settings.model_match_abs_tol,
            "model_match_rel_tolerance": settings.model_match_rel_tol,
            "comparison_classification": classification,
        })
    return rows


def mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not numeric else statistics.fmean(numeric)


def summarize_triangular_table2_results(
    records: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["solver_requested"]), int(record["n"]), str(record["model_id"]))].append(record)

    comparison_counts: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for comparison in comparisons:
        comparison_counts[(str(comparison["solver"]), int(comparison["n"]))][
            str(comparison["comparison_classification"])
        ] += 1

    rows: list[dict[str, Any]] = []
    for (solver, n, model_id), group in sorted(grouped.items()):
        optimal = [record for record in group if record["solver_status"] == "OPTIMAL"]
        time_limited = [record for record in group if record["termination_reason"] == "TIME_LIMIT"]
        validation_failures = [record for record in group if record["validation_status"] == "FAIL"]
        raw_validation_passes = [
            record for record in group if record["validation_status"] == "RAW_PASS"
        ]
        repaired_validation_passes = [
            record
            for record in group
            if record["validation_status"] == "NUMERICALLY_REPAIRED_PASS"
        ]
        other_failures = [
            record for record in group
            if record["solver_status"] not in {"OPTIMAL", "FEASIBLE"}
            and record["termination_reason"] != "TIME_LIMIT"
        ]
        counts = comparison_counts[(solver, n)]
        rows.append({
            "solver": solver,
            "formulation": MODEL_LABELS[model_id],
            "model_id": model_id,
            "n": n,
            "runs_recorded": len(group),
            "optimal_count": len(optimal),
            "time_limit_count": len(time_limited),
            "other_failure_count": len(other_failures),
            "validation_failure_count": len(validation_failures),
            "raw_validation_pass_count": len(raw_validation_passes),
            "numerically_repaired_pass_count": len(repaired_validation_passes),
            "solver_seconds_average_optimal": mean_or_none(
                record["solver_time_seconds"] for record in optimal
            ),
            "solver_seconds_average_including_limits": mean_or_none(
                record["solver_time_seconds"] for record in group
            ),
            "wall_seconds_average_optimal": mean_or_none(
                record["wall_solve_and_unpack_seconds"] for record in optimal
            ),
            "wall_seconds_average_including_limits": mean_or_none(
                record["wall_solve_and_unpack_seconds"] for record in group
            ),
            "nodes_average_optimal": mean_or_none(record["node_count"] for record in optimal),
            "nodes_average_including_limits": mean_or_none(record["node_count"] for record in group),
            "comparison_pass_count": counts.get("PASS", 0),
            "comparison_fail_count": counts.get("FAIL", 0),
            "comparison_not_comparable_status_count": counts.get(
                "NOT_COMPARABLE_DUE_TO_STATUS", 0
            ),
            "comparison_not_comparable_validation_count": counts.get(
                "NOT_COMPARABLE_DUE_TO_VALIDATION", 0
            ),
        })
    return rows


COMPARISON_FIELDS = (
    "solver", "n", "replication", "instance_id",
    "bigm_record_id", "perspective_record_id", "bigm_solver_status",
    "perspective_solver_status", "bigm_validation_status",
    "perspective_validation_status", "bigm_common_objective",
    "perspective_common_objective", "objective_absolute_difference",
    "objective_relative_difference", "model_match_abs_tolerance",
    "model_match_rel_tolerance", "comparison_classification",
)

SUMMARY_FIELDS = (
    "solver", "formulation", "model_id", "n", "runs_recorded", "optimal_count",
    "time_limit_count", "other_failure_count", "validation_failure_count",
    "raw_validation_pass_count", "numerically_repaired_pass_count",
    "solver_seconds_average_optimal", "solver_seconds_average_including_limits",
    "wall_seconds_average_optimal", "wall_seconds_average_including_limits",
    "nodes_average_optimal", "nodes_average_including_limits",
    "comparison_pass_count", "comparison_fail_count",
    "comparison_not_comparable_status_count",
    "comparison_not_comparable_validation_count",
)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_payload() -> dict[str, Any]:
    cp, np, _ = import_optimization_stack()
    return {
        "captured_at_utc": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cvxpy_version": cp.__version__,
        "numpy_version": np.__version__,
        "cplex_package_version": package_version("cplex"),
        "mosek_package_version": package_version("Mosek"),
        "gurobipy_package_version": package_version("gurobipy"),
        "cvxpy_installed_solvers": sorted(cp.installed_solvers()),
    }


def write_run_settings_csv(path: Path, value: Mapping[str, Any]) -> None:
    """Write nested environment/settings metadata as readable key-value CSV."""
    rows: list[dict[str, str]] = []

    def visit(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in sorted(item.items()):
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(item, (list, tuple)):
            rows.append({"parameter": prefix, "value": ";".join(map(str, item))})
        else:
            rows.append({"parameter": prefix, "value": "" if item is None else str(item)})

    visit("", value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("parameter", "value"))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def preflight_solver(solver: str, settings: RunSettings) -> dict[str, Any]:
    cp, _, _ = import_optimization_stack()
    installed = set(cp.installed_solvers())
    require(solver in installed, f"CVXPY does not report {solver} as installed")

    x = cp.Variable(nonneg=True, name="preflight_x")
    u = cp.Variable(nonneg=True, name="preflight_u")
    z = cp.Variable(boolean=True, name="preflight_z")
    constraints = [
        x <= 2.0 * z,
        cp.SOC(u + z, cp.hstack([2.0 * x, u - z])),
    ]
    problem = cp.Problem(cp.Maximize(3.0 * x - u), constraints)
    require(problem.is_dcp() and problem.is_mixed_integer(), "preflight MISOCP is invalid")
    data, chain, inverse_data = problem.get_problem_data(solver=solver)
    started = time.perf_counter()
    raw = chain.solve_via_data(
        problem,
        data,
        warm_start=False,
        verbose=settings.verbose_solver,
        solver_opts=solver_options(settings),
    )
    native = extract_native_stats(raw, solver)
    problem.unpack_results(raw, chain, inverse_data)
    elapsed = time.perf_counter() - started
    require(problem.status == cp.OPTIMAL, f"{solver} preflight status is {problem.status!r}")
    require(problem.value is not None and abs(float(problem.value) - 2.25) <= 1.0e-5, (
        f"{solver} preflight objective mismatch: {problem.value!r}"
    ))
    return {
        "solver": solver,
        "status": problem.status,
        "objective": float(problem.value),
        "expected_objective": 2.25,
        "elapsed_seconds": elapsed,
        "native_status_text": native.get("native_status_text"),
        "result": "PASS",
    }


def run_self_tests() -> None:
    tests: list[tuple[str, Any]] = []

    def test_triangular_oracle() -> None:
        args = (10.0, 50.0, 110.0, 8.0, 4.0, 1.0)
        for x in (0.0, 10.0, 30.0, 50.0, 80.0, 110.0):
            profit = triangular_expected_profit(x, *args)
            require(math.isfinite(profit), "oracle produced a non-finite profit")
        require(triangular_expected_profit(0.0, *args) == 0.0, "oracle zero-order profit")
        require(triangular_expected_profit(10.0, *args) == 40.0, "oracle guaranteed-segment profit")

    def test_xu_matches_cdf() -> None:
        args = (10.0, 50.0, 110.0, 8.0, 4.0, 1.0)
        x_u = triangular_unconstrained_quantity(*args)
        alpha = (args[3] - args[4]) / (args[3] - args[5])
        require(
            close_enough(triangular_cdf(x_u, args[0], args[1], args[2]), alpha),
            "x_u must equal the alpha-fractile of the triangular CDF",
        )

    def test_selection_parser() -> None:
        require(parse_int_selection("100,300-400", TABLE2_N_VALUES, "n") == (100, 300, 400), "n parser")
        require(parse_int_selection("1-3,3", TABLE2_REPLICATIONS, "replication") == (1, 2, 3), "rep parser")

    def test_rotated_cone_identity() -> None:
        for x, u, v in ((0.0, 0.0, 0.0), (2.0, 4.0, 1.0), (3.0, 2.0, 4.5)):
            left = math.sqrt((2.0 * x) ** 2 + (u - v) ** 2)
            right = u + v
            require(
                (left <= right + 1.0e-12) == (x * x <= u * v + 1.0e-12),
                "cone identity",
            )

    def test_comparison_threshold() -> None:
        threshold = scaled_tolerance(1.0e-4, 0.002, 100.0, 100.1)
        require(abs(100.0 - 100.1) <= threshold, "comparison threshold")

    def test_timeout_status_guard() -> None:
        observed = classify_termination(
            "optimal_inaccurate",
            None,
            True,
            time_limit_seconds=10.0,
            observed_solve_seconds=9.9,
        )
        require(
            observed == ("FEASIBLE", "TIME_LIMIT", "FEASIBLE_POINT"),
            f"timeout guard returned {observed!r}",
        )

    def test_pairwise_comparison_logic() -> None:
        settings = RunSettings(
            solver="CPLEX",
            time_limit_seconds=3600.0,
            mip_gap=0.001,
            threads=1,
            recompute_abs_tol=1.0e-5,
            recompute_rel_tol=1.0e-7,
            numerical_repair_abs_tol=1.0e-3,
            numerical_repair_rel_tol=1.0e-8,
            repair_objective_abs_tol=1.0e-4,
            repair_objective_rel_tol=5.0e-6,
            model_match_abs_tol=1.0e-4,
            model_match_rel_tol=0.002,
            feasibility_abs_tol=1.0e-6,
            feasibility_rel_tol=1.0e-8,
            integrality_tol=1.0e-5,
            verbose_solver=False,
        )
        base = {
            "solver_requested": "CPLEX",
            "instance_id": "TRIANGULAR-N0100-R001",
            "solver_status": "OPTIMAL",
            "validation_status": "RAW_PASS",
            "n": 100,
            "replication": 1,
        }
        records = [
            {**base, "record_id": "left", "model_id": BIGM_MODEL_ID,
             "objective_independent_common": 100.0},
            {**base, "record_id": "right", "model_id": PERSPECTIVE_MODEL_ID,
             "validation_status": "NUMERICALLY_REPAIRED_PASS",
             "objective_independent_common": 100.1},
        ]
        comparisons = compare_triangular_table2_formulations(
            records, {"CPLEX": settings}
        )
        require(len(comparisons) == 1, "comparison count")
        require(comparisons[0]["comparison_classification"] == "PASS", "comparison class")

    def test_controlled_numerical_repair() -> None:
        instance = TriangularInstance(
            n=1,
            replication=1,
            instance_id="TRIANGULAR-N0001-R001",
            master_seed=1,
            instance_seed=2,
            a=(10.0,),
            m=(40.0,),
            b=(100.0,),
            cost_c=(3.0,),
            price_p=(5.0,),
            salvage_s=(1.0,),
            x_u=(45.0,),
            fixed_cost_score=(25.0,),
            k=(27.5,),
            kappa=50,
            K=27.5,
            capacity_fraction=1.0,
            C=45.0,
            generation_version=EXPECTED_GENERATION_VERSION,
        )
        settings = RunSettings(
            solver="MOSEK",
            time_limit_seconds=300.0,
            mip_gap=0.001,
            threads=1,
            recompute_abs_tol=1.0e-5,
            recompute_rel_tol=1.0e-7,
            numerical_repair_abs_tol=1.0e-3,
            numerical_repair_rel_tol=1.0e-8,
            repair_objective_abs_tol=1.0e-4,
            repair_objective_rel_tol=5.0e-6,
            model_match_abs_tol=1.0e-4,
            model_match_rel_tol=0.002,
            feasibility_abs_tol=1.0e-6,
            feasibility_rel_tol=1.0e-8,
            integrality_tol=1.0e-5,
            verbose_solver=False,
        )
        small_leak = {
            "bar_x": [2.0e-4], "hat_x": [6.0e-4], "y": [60.0], "z": [0.0]
        }
        repaired, metrics = repair_numerical_solution(instance, small_leak, settings)
        require(repaired is not None and metrics["repair_allowed"], "small repair rejected")
        require(repaired["q"] == [0.0] and repaired["u_hat"] == [0.0], "small repair wrong")
        large_leak = {
            "bar_x": [2.0e-3], "hat_x": [6.0e-3], "y": [60.0], "z": [0.0]
        }
        rejected, metrics = repair_numerical_solution(instance, large_leak, settings)
        require(rejected is None and not metrics["repair_allowed"], "large repair accepted")

    def test_settings_comparison_includes_validation_contract() -> None:
        settings = RunSettings(
            solver="MOSEK",
            time_limit_seconds=300.0,
            mip_gap=0.001,
            threads=1,
            recompute_abs_tol=1.0e-5,
            recompute_rel_tol=1.0e-7,
            numerical_repair_abs_tol=1.0e-3,
            numerical_repair_rel_tol=1.0e-8,
            repair_objective_abs_tol=1.0e-4,
            repair_objective_rel_tol=5.0e-6,
            model_match_abs_tol=1.0e-4,
            model_match_rel_tol=0.002,
            feasibility_abs_tol=1.0e-6,
            feasibility_rel_tol=1.0e-8,
            integrality_tol=1.0e-5,
            verbose_solver=False,
        )
        changed = dataclasses.replace(settings, numerical_repair_abs_tol=2.0e-3)
        require(
            dataclasses.asdict(settings) != dataclasses.asdict(changed),
            "settings comparison ignored validation contract",
        )

    tests.extend([
        ("Triangular expected-profit oracle", test_triangular_oracle),
        ("Triangular x_u matches CDF fractile", test_xu_matches_cdf),
        ("CLI selection parsing", test_selection_parser),
        ("Rotated SOC algebra", test_rotated_cone_identity),
        ("Objective comparison tolerance", test_comparison_threshold),
        ("Timeout cannot be optimal", test_timeout_status_guard),
        ("Pairwise objective comparison", test_pairwise_comparison_logic),
        ("Controlled numerical repair", test_controlled_numerical_repair),
        ("Settings comparison includes validation", test_settings_comparison_includes_validation_contract),
    ])
    failures = 0
    for index, (name, function) in enumerate(tests, start=1):
        try:
            function()
            print(f"SELF-TEST {index:02d} PASS - {name}")
        except Exception as error:
            failures += 1
            print(f"SELF-TEST {index:02d} FAIL - {name}: {error}")
    print(f"Self-tests: {len(tests) - failures} PASS, {failures} FAIL")
    if failures:
        raise Table2Error("one or more self-tests failed")


def validate_settings(args: argparse.Namespace) -> None:
    require(args.time_limit > 0.0 and math.isfinite(args.time_limit), "time limit must be positive")
    require(0.0 <= args.mip_gap < 1.0 and math.isfinite(args.mip_gap), "MIP gap must be in [0,1)")
    require(args.threads >= 1, "threads must be at least 1")
    for name in (
        "recompute_abs_tol", "recompute_rel_tol",
        "numerical_repair_abs_tol", "numerical_repair_rel_tol",
        "repair_objective_abs_tol", "repair_objective_rel_tol",
        "model_match_abs_tol",
        "model_match_rel_tol", "feasibility_abs_tol", "feasibility_rel_tol",
        "integrality_tol",
    ):
        value = getattr(args, name)
        require(value >= 0.0 and math.isfinite(value), f"{name} must be finite and nonnegative")


def settings_for_solver(args: argparse.Namespace, solver: str) -> RunSettings:
    return RunSettings(
        solver=solver,
        time_limit_seconds=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        recompute_abs_tol=args.recompute_abs_tol,
        recompute_rel_tol=args.recompute_rel_tol,
        numerical_repair_abs_tol=args.numerical_repair_abs_tol,
        numerical_repair_rel_tol=args.numerical_repair_rel_tol,
        repair_objective_abs_tol=args.repair_objective_abs_tol,
        repair_objective_rel_tol=args.repair_objective_rel_tol,
        model_match_abs_tol=args.model_match_abs_tol,
        model_match_rel_tol=args.model_match_rel_tol,
        feasibility_abs_tol=args.feasibility_abs_tol,
        feasibility_rel_tol=args.feasibility_rel_tol,
        integrality_tol=args.integrality_tol,
        verbose_solver=args.verbose_solver,
    )


def make_parser() -> argparse.ArgumentParser:
    script_directory = Path(__file__).resolve().parent
    table2_csv = script_directory / "triangular_table2_instances.csv"
    parser = argparse.ArgumentParser(
        description="CVXPY comparison of Triangular Big-M (9) and its perspective "
        "reformulation ((9c) replaced by (5)) for revised Table 2."
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--self-test", action="store_true", help="run tests that do not call a solver")
    actions.add_argument("--preflight", action="store_true", help="check installation/license and solve a tiny MISOCP")
    actions.add_argument("--validate-data-only", action="store_true", help="validate selected frozen CSV instances; do not import CVXPY")
    actions.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild comparison and summary CSVs from solver-record CSV",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=table2_csv,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_directory / "triangular_table2_results",
    )
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help="CPLEX, MOSEK, GUROBI, a comma-separated combination (e.g. MOSEK,GUROBI), "
        "BOTH (CPLEX+MOSEK, legacy alias), or ALL",
    )
    parser.add_argument("--models", default=DEFAULT_MODELS, help="both, bigm, or perspective")
    parser.add_argument("--n-values", default=DEFAULT_N_VALUES, help="all or comma/range selection")
    parser.add_argument("--replications", default=DEFAULT_REPLICATIONS, help="all or comma/range selection")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--recompute-abs-tol", type=float, default=DEFAULT_RECOMPUTE_ABS_TOL)
    parser.add_argument("--recompute-rel-tol", type=float, default=DEFAULT_RECOMPUTE_REL_TOL)
    parser.add_argument(
        "--numerical-repair-abs-tol",
        type=float,
        default=DEFAULT_NUMERICAL_REPAIR_ABS_TOL,
    )
    parser.add_argument(
        "--numerical-repair-rel-tol",
        type=float,
        default=DEFAULT_NUMERICAL_REPAIR_REL_TOL,
    )
    parser.add_argument(
        "--repair-objective-abs-tol",
        type=float,
        default=DEFAULT_REPAIR_OBJECTIVE_ABS_TOL,
    )
    parser.add_argument(
        "--repair-objective-rel-tol",
        type=float,
        default=DEFAULT_REPAIR_OBJECTIVE_REL_TOL,
    )
    parser.add_argument("--model-match-abs-tol", type=float, default=DEFAULT_MODEL_MATCH_ABS_TOL)
    parser.add_argument("--model-match-rel-tol", type=float, default=DEFAULT_MODEL_MATCH_REL_TOL)
    parser.add_argument("--feasibility-abs-tol", type=float, default=DEFAULT_FEASIBILITY_ABS_TOL)
    parser.add_argument("--feasibility-rel-tol", type=float, default=DEFAULT_FEASIBILITY_REL_TOL)
    parser.add_argument("--integrality-tol", type=float, default=DEFAULT_INTEGRALITY_TOL)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="append a new result even when an exact completed signature already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_settings(args)
    solvers = parse_solvers(args.solver)
    models = parse_models(args.models)
    n_values = parse_int_selection(args.n_values, TABLE2_N_VALUES, "n")
    replications = parse_int_selection(args.replications, TABLE2_REPLICATIONS, "replication")

    if args.self_test:
        run_self_tests()
        print("TABLE2_SELF_TESTS_PASS")
        return 0

    settings_by_solver = {
        solver: settings_for_solver(args, solver) for solver in solvers
    }

    if args.preflight:
        payload = environment_payload()
        payload["solver_preflight"] = [
            preflight_solver(solver, settings_by_solver[solver]) for solver in solvers
        ]
        for key, value in sorted(payload.items()):
            print(f"{key}: {value}")
        print("TABLE2_SERVER_PREFLIGHT_PASS")
        return 0

    data_path = args.data.resolve()
    require(data_path.is_file(), f"Triangular CSV does not exist: {data_path}")
    if args.validate_data_only:
        count = sum(
            1
            for _ in read_selected_triangular_table2_instances(
                data_path, n_values, replications
            )
        )
        expected = len(n_values) * len(replications)
        require(count == expected, f"validated {count} instances; expected {expected}")
        print(f"Validated {count} frozen Triangular instances from {data_path}")
        print("TABLE2_DATA_VALIDATION_PASS")
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_RESULT_FILENAME
    comparison_path = output_dir / COMPARISON_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    environment_path = output_dir / ENVIRONMENT_FILENAME

    if not args.aggregate_only:
        environment = environment_payload()
        missing_solvers = sorted(set(solvers) - set(environment["cvxpy_installed_solvers"]))
        require(not missing_solvers, f"CVXPY does not report installed solver(s): {missing_solvers}")
        environment.update({
            "input_file": str(data_path),
            "settings": {
                solver: dataclasses.asdict(settings) for solver, settings in settings_by_solver.items()
            },
        })
        write_run_settings_csv(environment_path, environment)

        existing = read_solver_records_csv(raw_path)
        existing_signatures = {
            run_signature(record)
            for record in existing
            if record.get("solver_status") != "ERROR"
        }
        invocation_id = str(uuid.uuid4())
        completed = skipped = 0
        for instance in read_selected_triangular_table2_instances(
            data_path, n_values, replications
        ):
            for solver in solvers:
                settings = settings_by_solver[solver]
                for model_id in models:
                    signature = (
                        solver,
                        model_id,
                        instance.instance_id,
                        tuple(sorted(dataclasses.asdict(settings).items())),
                    )
                    if signature in existing_signatures and not args.rerun:
                        skipped += 1
                        print(f"SKIP existing {solver} {model_id} {instance.instance_id}")
                        continue
                    print(f"SOLVE {solver} {model_id} {instance.instance_id}", flush=True)
                    record = solve_triangular_table2_formulation(
                        instance,
                        model_id,
                        settings,
                        input_path=data_path,
                        invocation_id=invocation_id,
                    )
                    append_solver_record_csv(raw_path, record)
                    existing_signatures.add(signature)
                    completed += 1
                    print(
                        f"RESULT {record['solver_status']} validation={record['validation_status']} "
                        f"objective={record['objective_independent_common']}",
                        flush=True,
                    )
        print(f"Completed {completed} new model runs; skipped {skipped} existing runs")

    require(
        raw_path.is_file(),
        f"raw results do not exist yet: {raw_path}; run models before --aggregate-only",
    )
    all_records = read_solver_records_csv(raw_path)
    relevant = latest_matching_records(
        all_records,
        settings_by_solver=settings_by_solver,
    )
    comparisons = compare_triangular_table2_formulations(
        relevant, settings_by_solver
    )
    summaries = summarize_triangular_table2_results(relevant, comparisons)
    atomic_write_csv(comparison_path, COMPARISON_FIELDS, comparisons)
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summaries)

    print(f"Raw results: {raw_path}")
    print(f"Model comparisons: {comparison_path}")
    print(f"Table 2 summary: {summary_path}")
    print(f"Relevant raw records: {len(relevant)}; comparisons: {len(comparisons)}")
    print("TABLE2_RESULTS_READY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Table2Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
