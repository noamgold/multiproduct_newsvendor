"""Solve Triangular fixed-cost instances with an MINLP solver -- Bonmin or
SCIP, selectable via --minlp-solver / the DEFAULT_SOLVER global (see
MINLP_SOLVER_CHOICES) -- using the un-lifted cubic MINLP formulation below
(including the formerly separate model implementation). It compares the
objective, node count, and CPU time against GUROBI's results for the same
instances, for either of two datasets selected by ``--dataset``:

* "table2" (the default): the original manuscript (equation (13)) Table 2
  instances, triangular_table2_instances.csv. GUROBI's reference results
  come from the official, append-only, frozen log
  triangular_table2_results/triangular_table2_solver_records.csv -- which
  this script only ever READS, never writes to (see "Why a separate output
  file" below). The MINLP solver's own results, plus their comparison
  against that GUROBI reference, are written to
  triangular_table2_bonmin_comparison.csv.
* "strongly_correlated": the harder, minimal-fixed-cost-shift instances
  from generate_strongly_correlated_instances.py
  (triangular_table2_instances_strongly_correlated.csv). GUROBI's reference
  results AND the MINLP solver's own results/comparison live together in
  strongly_correlated_table2_results.csv, produced by
  run_triangular_strongly_correlated.py.

Bonmin vs. SCIP: both are invoked through the identical Pyomo/AMPL-interface
API (``pyo.SolverFactory(solver, executable=...).solve(model, ...)``) on
the exact same build_triangular_2018_fixed_cost model -- see
minlp_solver_options() for the (different) option key names each expects
and parse_minlp_solver_log() for the (different) console-log formats each
prints its node-count/CPU-time in. Bonmin's option names and log format are
confirmed against a real run in this project; SCIP's are not yet (no SCIP
executable installed here as of this code's introduction) -- spot-check and
adjust once SCIP is actually run, the same way Bonmin's own regex was fixed
after its first real run.

Why a separate output file for "table2" rather than writing into
triangular_table2_solver_records.csv directly: that file is appended to,
one row per solve, by run_triangular_table2_experiment.py's
append_solver_record_csv(), which enforces (via a require() check) that
every future append's column set matches the file's existing header
exactly. Adding Bonmin-only columns to that file would break that
invariant the next time run_triangular_table2_experiment.py is run with
--rerun. triangular_table2_bonmin_comparison.csv instead mirrors that same
per-(instance, model) row shape as a plain, non-append-only comparison
table, upserted (not strictly appended) on every run of this script.

Model background. Two
formulations of essentially the same combinatorial problem already exist
in this project:

* triangular_current.py's ``build_current_triangular_bigm_fixed_cost`` /
  ``build_current_triangular_perspective_fixed_cost`` -- manuscript
  formulation (9) and its perspective reformulation. Both *lift* the cubic
  terms into a tower of auxiliary variables (u_hat, w_hat, u_y, w_y) and
  second-order-cone constraints so the whole problem is a convex MISOCP
  that a conic MIP solver (CPLEX/MOSEK/GUROBI) can handle.
* triangular_2018.py's ``build_cg2018_triangular_continuous`` -- the 2018
  paper's own model (12)-(13), written directly in the three physical
  segment variables (bar_x, hat_x, tilde_x) with the *un-lifted* cubic
  objective. The 2018 paper itself has no fixed-cost/display constraint or
  binary variables at all; extending it that way was explicitly left as
  future work (2018 paper, Section 7 conclusion: "we are studying an
  extension of the problem with an additional discrete display capacity
  constraint ... a knapsack-type constraint that corresponds to a limit on
  the total fixed cost of carrying different types of items").

``build_triangular_2018_fixed_cost`` below is that extension: it adds
triangular_current.py's binary knapsack structure (a 0/1 indicator z_i per
product, Big-M constraints linking each segment to z_i, and a fixed-cost
budget sum_i k_i z_i <= K) on top of triangular_2018.py's *un-lifted*
cubic objective -- i.e. it is the natural MINLP that formulation (9) would
be if it were not first convexified into an MISOCP. No auxiliary lift
variables are needed here (u_hat, w_hat, u_y, w_y and their SOC constraints
have no purpose without a conic solver), so this model has roughly half as
many variables and no second-order-cone constraints; what it has instead is
a genuinely nonlinear (though per-segment concave) integer program, which
is exactly the class of problem Bonmin (Basic Open-source Nonlinear Mixed
INteger programming, COIN-OR) targets: a B&B over the integer variables
where each node solves an NLP relaxation (internally, via Ipopt) rather
than an SOCP relaxation.

The model, per product i with support [a_i, b_i], mode m_i, and the same
Big-M constants used by triangular_current.py's fixed-cost models
(hat_bigm_i = max(0, min(x_U_i, m_i) - a_i), tilde_bigm_i = max(0, x_U_i - m_i)):

    max  sum_i (p_i - c_i)(bar_x_i + hat_x_i + tilde_x_i)
              + (s_i - p_i)(m_i - a_i)/(b_i - a_i) * tilde_x_i
              + (s_i - p_i)/(b_i - a_i) * tilde_x_i^2
              + (s_i - p_i)/(3(b_i - a_i)) * (hat_x_i^3/(m_i-a_i) - tilde_x_i^3/(b_i-m_i))
    s.t. sum_i (bar_x_i + hat_x_i + tilde_x_i) <= C
         sum_i k_i * z_i <= K
         bar_x_i <= a_i * z_i,  hat_x_i <= hat_bigm_i * z_i,  tilde_x_i <= tilde_bigm_i * z_i
         0 <= tilde_x_i <= b_i - m_i,  z_i in {0, 1}

Pyomo is used here (rather than CVXPY, as in triangular_current.py) because
Bonmin is a general MINLP solver invoked through the AMPL Solver Library
interface, which is what Pyomo's SolverFactory targets; CVXPY has no Bonmin
backend (its solver list is restricted to DCP-reducible conic/QP solvers).

Node-count/CPU-time capture. solve_triangular_2018_fixed_cost() (kept below
for API completeness) reports solve_seconds (wall clock) and a
success/status flag from Pyomo's structured SolverResults, but neither
solver's B&B node count is part of that structured result -- Pyomo's
generic ASL solver interface does not parse it out for either. Bonmin's
B&B tree is managed internally by Cbc, which prints a standard end-of-solve
summary block to its console output containing lines of the form

    Enumerated nodes:               <N>
    Time (CPU seconds):             <T>

(confirmed format: see CBC_SUMMARY_PATTERN). SCIP prints its own
end-of-solve statistics block (see SCIP_NODE_COUNT_PATTERNS /
SCIP_CPU_SECONDS_PATTERNS -- not yet confirmed against a real run). So
solve_and_collect_minlp_stats() (the function this script's comparison
driver actually calls) captures the raw solver log (via Pyomo's
``logfile=`` option) and parses it for those two figures with
parse_minlp_solver_log(), which dispatches to the right pattern set for
whichever solver was selected.

Single-threading: Bonmin/Cbc's branch-and-bound has no general parallel
tree search enabled by default and no Threads-style option analogous to
GUROBI's; the only realistic multi-threading risk is an underlying BLAS/
LAPACK library used by Ipopt's linear solver at each NLP-relaxation node.
force_single_threaded_environment() sets OMP_NUM_THREADS / MKL_NUM_THREADS
/ OPENBLAS_NUM_THREADS to "1" in this process before invoking Bonmin, which
subprocess.Popen (used internally by Pyomo's ASL solver interface)
inherits by default.

Usage:

    python triangular2018withfixedcosts.py --n-values 100,200 --replications 1-30
    python triangular2018withfixedcosts.py --minlp-solver scip --n-values 100 --replications 1-5
    python triangular2018withfixedcosts.py --dataset strongly_correlated --n-values all --replications all
    python triangular2018withfixedcosts.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import pyomo.environ as pyo
except ImportError as error:  # pragma: no cover - import guard only
    raise ImportError(
        "triangular2018withfixedcosts requires Pyomo to be installed (pip install pyomo)."
    ) from error

import generate_strongly_correlated_instances as gsc
import run_triangular_table2_experiment as rte

# ===========================================================================
# The un-lifted cubic MINLP
# model Bonmin solves. See module docstring above for the model itself.
# ===========================================================================

MODEL_ID = "CG2018_TRIANGULAR_CUBIC_FIXED_COST_MINLP"

# Which Pyomo/AMPL-interface MINLP solver build_triangular_2018_fixed_cost's
# model is solved with, selectable via --minlp-solver on the comparison
# driver below (or by passing solver=... directly to
# solve_triangular_2018_fixed_cost / solve_and_collect_minlp_stats).
# DEFAULT_SOLVER is the fallback when no --minlp-solver / solver= is given
# -- change this module-level global to switch the project-wide default
# without touching call sites.
MINLP_SOLVER_CHOICES = ("bonmin", "scip")
DEFAULT_SOLVER = "scip"


class Triangular2018FixedCostError(RuntimeError):
    """Raised when input data or solver configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Triangular2018FixedCostError(message)


def _solver_executable_present(solver: str) -> bool:
    """Whether ``solver``'s executable can be found, checked with a plain
    PATH lookup first. Pyomo's ASL-based SolverFactory(...) eagerly
    validates its executable at construction time and prints a noisy
    WARNING plus a full traceback to stderr when it cannot find one (even
    though it then raises cleanly), so this avoids triggering that in the
    common case where the solver just is not installed.
    """
    import shutil

    if shutil.which(solver) is None:
        return False
    try:
        return bool(pyo.SolverFactory(solver).available())
    except Exception:
        return False


def _as_vector(values: Sequence[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    _require(array.shape == (n,), f"{name} must have shape ({n},); observed {array.shape}")
    _require(bool(np.all(np.isfinite(array))), f"{name} contains a non-finite value")
    return array


def _validate_support(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> None:
    _require(bool(np.all(a >= 0.0)), "a must be nonnegative")
    _require(bool(np.all(a < m)), "a must be strictly less than m")
    _require(bool(np.all(m < b)), "m must be strictly less than b")


def minlp_solver_options(
    solver: str, *, time_limit_seconds: float, mip_gap: float
) -> dict[str, Any]:
    """Pyomo ``solver.options[...]`` entries for the selected MINLP solver,
    both invoked through the identical Pyomo/AMPL-interface API
    (``pyo.SolverFactory(solver, executable=...).solve(model, ...)``) on the
    SAME build_triangular_2018_fixed_cost model -- only the option key
    names differ, since each solver defines its own AMPL option namespace.

    "bonmin": bonmin.time_limit (seconds) / bonmin.allowable_fraction_gap
    (relative gap fraction) -- confirmed working against a real Bonmin run
    in this project (see CBC_SUMMARY_PATTERN's docstring).

    "scip": limits/time (seconds) / limits/gap (relative primal-dual gap
    fraction) -- SCIP's own documented parameter paths (matching its
    interactive-shell ``set limits time/gap`` commands and .set file
    format), passed through unchanged by Pyomo's ASL options-string
    mechanism. NOT yet validated against a real SCIP run in this project;
    spot-check the first time SCIP is actually invoked, the same way
    Bonmin's log-parsing regex was fixed after its first real run.
    """
    solver = solver.lower()
    if solver == "bonmin":
        return {
            "bonmin.time_limit": time_limit_seconds,
            "bonmin.allowable_fraction_gap": mip_gap,
        }
    if solver == "scip":
        return {
            "limits/time": time_limit_seconds,
            "limits/gap": mip_gap,
        }
    raise Triangular2018FixedCostError(
        f"unsupported MINLP solver: {solver!r}; expected one of {MINLP_SOLVER_CHOICES}"
    )


@dataclasses.dataclass(slots=True)
class Triangular2018FixedCostFormulation:
    """A built Pyomo ConcreteModel plus the data needed to interpret it."""

    model: Any
    n: int
    a: np.ndarray
    m: np.ndarray
    b: np.ndarray
    price_p: np.ndarray
    cost_c: np.ndarray
    salvage_s: np.ndarray
    x_u: np.ndarray
    k: np.ndarray
    C: float
    K: float
    model_id: str = MODEL_ID


def build_triangular_2018_fixed_cost(
    a: Sequence[float],
    m: Sequence[float],
    b: Sequence[float],
    price_p: Sequence[float],
    cost_c: Sequence[float],
    salvage_s: Sequence[float],
    x_u: Sequence[float],
    k: Sequence[float],
    C: float,
    K: float,
) -> Triangular2018FixedCostFormulation:
    """Build the fixed-cost cubic MINLP described in the module docstring.

    ``x_u`` (each product's unconstrained single-product optimum) is used
    only to tighten the Big-M constants exactly as
    triangular_current.py._build_triangular_fixed_cost_model does (manuscript
    Proposition 1): hat_bigm_i = max(0, min(x_u_i, m_i) - a_i),
    tilde_bigm_i = max(0, x_u_i - m_i).
    """
    n = len(a)
    a, m, b = (_as_vector(v, n, name) for v, name in ((a, "a"), (m, "m"), (b, "b")))
    p = _as_vector(price_p, n, "price_p")
    c = _as_vector(cost_c, n, "cost_c")
    s = _as_vector(salvage_s, n, "salvage_s")
    x_u = _as_vector(x_u, n, "x_u")
    k = _as_vector(k, n, "k")
    _validate_support(a, m, b)
    _require(bool(np.all(x_u >= 0.0)), "x_u must be nonnegative")
    _require(bool(np.all(k >= 0.0)), "k must be nonnegative")
    _require(np.isfinite(C) and C >= 0.0, "C must be finite and nonnegative")
    _require(np.isfinite(K) and K >= 0.0, "K must be finite and nonnegative")

    coeff_lin = p - c
    coeff3 = (s - p) / (b - a)
    hat_bigm = np.maximum(0.0, np.minimum(x_u, m) - a)
    tilde_bigm = np.maximum(0.0, x_u - m)

    model = pyo.ConcreteModel(name=MODEL_ID)
    model.I = pyo.RangeSet(0, n - 1)
    model.z = pyo.Var(model.I, domain=pyo.Binary)
    model.bar_x = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.hat_x = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.tilde_x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=lambda mdl, i: (0.0, b[i] - m[i]))

    def _objective(mdl: Any) -> Any:
        return sum(
            coeff_lin[i] * (mdl.bar_x[i] + mdl.hat_x[i] + mdl.tilde_x[i])
            + coeff3[i] * (m[i] - a[i]) * mdl.tilde_x[i]
            + coeff3[i] * mdl.tilde_x[i] ** 2
            + (coeff3[i] / 3.0) * (mdl.hat_x[i] ** 3 / (m[i] - a[i]) - mdl.tilde_x[i] ** 3 / (b[i] - m[i]))
            for i in mdl.I
        )

    model.objective = pyo.Objective(rule=_objective, sense=pyo.maximize)

    model.bar_link = pyo.Constraint(model.I, rule=lambda mdl, i: mdl.bar_x[i] <= a[i] * mdl.z[i])
    model.hat_link = pyo.Constraint(model.I, rule=lambda mdl, i: mdl.hat_x[i] <= hat_bigm[i] * mdl.z[i])
    model.tilde_link = pyo.Constraint(model.I, rule=lambda mdl, i: mdl.tilde_x[i] <= tilde_bigm[i] * mdl.z[i])
    model.capacity = pyo.Constraint(
        expr=sum(model.bar_x[i] + model.hat_x[i] + model.tilde_x[i] for i in model.I) <= C
    )
    model.fixed_cost_budget = pyo.Constraint(expr=sum(k[i] * model.z[i] for i in model.I) <= K)

    return Triangular2018FixedCostFormulation(
        model=model, n=n, a=a, m=m, b=b, price_p=p, cost_c=c, salvage_s=s,
        x_u=x_u, k=k, C=float(C), K=float(K),
    )


@dataclasses.dataclass(slots=True)
class MinlpSolveResult:
    """A minimal, solver-agnostic result: enough for extract_solution() and
    for reporting success/failure/timing uniformly with the other modules'
    solve result objects (triangular_2018.py's SimpleSolveResult in
    particular has the same field names).
    """

    bar_x: np.ndarray
    hat_x: np.ndarray
    tilde_x: np.ndarray
    z: np.ndarray
    success: bool
    status: int
    message: str
    fun: float
    constr_violation: float
    solve_seconds: float


def solve_triangular_2018_fixed_cost(
    formulation: Triangular2018FixedCostFormulation,
    *,
    solver: str = DEFAULT_SOLVER,
    executable: str | None = None,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 1.0e-3,
    tee: bool = False,
) -> MinlpSolveResult:
    """Solve ``formulation`` with Bonmin, SCIP (see MINLP_SOLVER_CHOICES),
    or another Pyomo/ASL MINLP solver -- see module docstring on why a
    local optimum is in fact global here, so any correct MINLP solver
    should agree. Kept for API completeness / standalone use; the
    comparison driver below (solve_and_collect_minlp_stats) duplicates
    most of this logic itself so it can additionally capture the solver's
    console log for node-count parsing, which this function's structured
    Pyomo result does not expose.
    """
    import time as _time

    _require(
        executable is not None or _solver_executable_present(solver),
        f"{solver} is not available; install it and ensure the executable is on PATH, "
        "or pass executable=... explicitly.",
    )
    augment_path_for_conda_executable(executable)
    solver_object = (
        pyo.SolverFactory(solver, executable=executable) if executable is not None
        else pyo.SolverFactory(solver)
    )
    for option_name, option_value in minlp_solver_options(
        solver, time_limit_seconds=time_limit_seconds, mip_gap=mip_gap
    ).items():
        solver_object.options[option_name] = option_value

    started = _time.perf_counter()
    results = solver_object.solve(formulation.model, tee=tee)
    elapsed = _time.perf_counter() - started

    termination = results.solver.termination_condition
    success = termination == pyo.TerminationCondition.optimal
    n = formulation.n
    bar = np.array([pyo.value(formulation.model.bar_x[i]) for i in range(n)])
    hat = np.array([pyo.value(formulation.model.hat_x[i]) for i in range(n)])
    tilde = np.array([pyo.value(formulation.model.tilde_x[i]) for i in range(n)])
    z = np.array([pyo.value(formulation.model.z[i]) for i in range(n)])
    objective_value = float(pyo.value(formulation.model.objective))

    capacity_violation = max(0.0, float(np.sum(bar) + np.sum(hat) + np.sum(tilde)) - formulation.C)
    budget_violation = max(0.0, float(np.sum(formulation.k * z)) - formulation.K)
    violation = max(capacity_violation, budget_violation)

    return MinlpSolveResult(
        bar_x=bar, hat_x=hat, tilde_x=tilde, z=z,
        success=success and violation <= 1.0e-6 + 1.0e-6 * max(1.0, formulation.C, formulation.K),
        status=1 if success else 0,
        message=f"{solver} termination_condition={termination}",
        fun=-objective_value,
        constr_violation=violation,
        solve_seconds=elapsed,
    )


def extract_solution(result: MinlpSolveResult) -> dict[str, np.ndarray]:
    return {
        "bar_x": result.bar_x,
        "hat_x": result.hat_x,
        "tilde_x": result.tilde_x,
        "z": result.z,
        "q": result.bar_x + result.hat_x + result.tilde_x,
    }


# ---------------------------------------------------------------------------
# Closed-form scalar oracle, duplicated (rather than imported) from the
# sibling scripts on purpose, used only to independently validate the
# objective returned by solve_triangular_2018_fixed_cost in _self_test().
# ---------------------------------------------------------------------------

def triangular_unconstrained_quantity(
    a: float, m: float, b: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    alpha = (price_p - cost_c) / (price_p - salvage_s)
    mode_cdf = (m - a) / (b - a)
    if alpha <= mode_cdf:
        return a + math.sqrt(alpha * (b - a) * (m - a))
    return b - math.sqrt((1.0 - alpha) * (b - a) * (b - m))


def triangular_expected_profit(
    x: float, a: float, m: float, b: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    x = max(0.0, x)
    x_bar = min(x, a)
    x_hat = min(max(x - a, 0.0), m - a)
    x_tilde = min(max(x - m, 0.0), b - m)
    return (
        (price_p - cost_c) * (x_bar + x_hat + x_tilde)
        + (salvage_s - price_p) * (m - a) * x_tilde / (b - a)
        + (salvage_s - price_p) * x_tilde ** 2 / (b - a)
        + (salvage_s - price_p) * (x_hat ** 3 / (m - a) - x_tilde ** 3 / (b - m)) / (3.0 * (b - a))
    )


def _brute_force_optimum(
    a: np.ndarray, m: np.ndarray, b: np.ndarray, p: np.ndarray, c: np.ndarray, s: np.ndarray,
    x_u: np.ndarray, k: np.ndarray, C: float, K: float,
) -> float:
    """Exhaustive reference solution for tiny n: enumerate every knapsack-
    feasible subset of active products and exactly solve the resulting
    (unconstrained-by-z) continuous relaxation with triangular_2018.py's
    exact dual bisection, restricted to that subset. Used only in
    _self_test() as an independent global-optimality check for Bonmin.
    """
    import itertools

    import triangular_2018 as t2018

    n = a.size
    best = float("-inf")
    for r in range(n + 1):
        for subset in itertools.combinations(range(n), r):
            if sum(k[i] for i in subset) > K + 1.0e-9:
                continue
            if not subset:
                best = max(best, 0.0)
                continue
            idx = list(subset)
            formulation = t2018.build_cg2018_triangular_continuous(
                a[idx], m[idx], b[idx], p[idx], c[idx], s[idx], C,
            )
            result = t2018.solve_cg2018_triangular_continuous(formulation, method="dual")
            if not result.success:
                continue
            best = max(best, -result.fun)
    return best


def _self_test() -> None:
    """In-memory smoke tests: build the model, and -- when Bonmin is
    available -- solve a tiny instance and cross-check both feasibility and
    global optimality against an independent brute-force enumeration built
    on triangular_2018.py's exact per-subset dual bisection.
    """
    rng = np.random.default_rng(0)
    n = 5
    a = rng.uniform(0.0, 50.0, size=n)
    spans = rng.uniform(80.0, 150.0, size=n)
    b = a + spans
    m = a + rng.uniform(0.1, 0.9, size=n) * spans
    c = rng.uniform(1.0, 5.0, size=n)
    p = c * rng.uniform(1.2, 2.0, size=n)
    s = rng.uniform(-0.5, 0.5, size=n) * c
    x_u = np.array([triangular_unconstrained_quantity(a[i], m[i], b[i], p[i], c[i], s[i]) for i in range(n)])
    k = rng.uniform(1.0, 5.0, size=n)
    C = 0.3 * float(np.sum(x_u))
    K = 0.5 * float(np.sum(k))

    formulation = build_triangular_2018_fixed_cost(a, m, b, p, c, s, x_u, k, C, K)
    _require(formulation.model.z[0].is_binary(), "self-test: z must be declared binary")
    _require(len(formulation.model.I) == n, "self-test: wrong number of products in model")

    bonmin_available = _solver_executable_present(DEFAULT_SOLVER)

    if not bonmin_available:
        print(
            "TRIANGULAR2018WITHFIXEDCOSTS_SELF_TEST_NOTE: "
            f"{DEFAULT_SOLVER!r} unavailable, skipping solve/optimality cross-check"
        )
        print("TRIANGULAR2018WITHFIXEDCOSTS_SELF_TEST_PASS")
        return

    result = solve_triangular_2018_fixed_cost(formulation, time_limit_seconds=60.0)
    _require(result.success, f"self-test: Bonmin did not report success: {result.message}")
    solution = extract_solution(result)

    for i in range(n):
        zi = solution["z"][i]
        _require(min(abs(zi), abs(1.0 - zi)) <= 1.0e-5, f"self-test: z[{i}] is not integral: {zi}")
    q = solution["q"]
    economic_objective = sum(
        triangular_expected_profit(q[i], a[i], m[i], b[i], p[i], c[i], s[i]) for i in range(n)
    )
    reported_objective = -result.fun
    gap = abs(economic_objective - reported_objective)
    _require(
        gap <= 1.0e-4 + 1.0e-6 * max(1.0, abs(reported_objective)),
        f"self-test: reconstructed and reported objectives disagree (gap {gap})",
    )

    reference = _brute_force_optimum(a, m, b, p, c, s, x_u, k, C, K)
    optimality_gap = abs(reference - reported_objective)
    _require(
        optimality_gap <= 1.0e-3 + 1.0e-6 * max(1.0, abs(reference)),
        f"self-test: Bonmin solution does not match brute-force global optimum "
        f"(bonmin={reported_objective}, brute_force={reference}, gap={optimality_gap})",
    )

    print("TRIANGULAR2018WITHFIXEDCOSTS_SELF_TEST_PASS")


# ===========================================================================
# Comparison-experiment driver
# ===========================================================================

TABLE2_INSTANCES_CSV = "triangular_table2_instances.csv"
TABLE2_RESULTS_CSV = "triangular_table2_bonmin_comparison.csv"
STRONGLY_CORRELATED_INSTANCES_CSV = "triangular_table2_instances_strongly_correlated.csv"
STRONGLY_CORRELATED_RESULTS_CSV = "strongly_correlated_table2_results.csv"

RESULT_FIELDNAMES = (
    "instance_id", "n", "replication", "model_id", "solver",
    "status", "objective", "independent_objective",
    "node_count", "native_relative_gap", "solve_seconds",
    "has_incumbent", "soc_ok", "economic_abs_diff",
    "baseline_node_count", "baseline_solve_seconds", "baseline_status",
    "bigm_vs_perspective_abs_diff", "bigm_vs_perspective_rel_diff",
    "bigm_vs_perspective_classification",
)

# Solver-agnostic column names: whichever MINLP solver (bonmin or scip) was
# selected for a run, its results land in these same columns, disambiguated
# by MINLP_SOLVER_NAME_COLUMN recording which one produced them. (Older
# data written before SCIP support was added used bonmin_-prefixed column
# names; merge_minlp_columns_into_results_csv unions old and new fieldnames
# rather than renaming/dropping any existing column, so old bonmin_* data
# stays intact alongside new rows/columns.)
MINLP_SOLVER_NAME_COLUMN = "minlp_solver_name"
MINLP_NODE_COUNT_COLUMN = "minlp_node_count"
MINLP_CPU_SECONDS_COLUMN = "minlp_cpu_seconds"
MINLP_WALL_SECONDS_COLUMN = "minlp_wall_seconds"
MINLP_STATUS_COLUMN = "minlp_status"
MINLP_OBJECTIVE_COLUMN = "minlp_objective"
MINLP_INDEPENDENT_OBJECTIVE_COLUMN = "minlp_independent_objective"
# Per-row comparison of THIS row's own independent_objective (Big-M's or
# Perspective's, whichever model_id the row is) against the MINLP solver's
# independent_objective, using the identical tolerance machinery
# run_triangular_table2_experiment.py uses to compare Big-M vs. Perspective
# against each other (DEFAULT_MODEL_MATCH_ABS_TOL / _REL_TOL, scaled_tolerance).
MINLP_ABS_DIFF_COLUMN = "minlp_objective_absolute_difference"
MINLP_REL_DIFF_COLUMN = "minlp_objective_relative_difference"
MINLP_MATCH_COLUMN = "minlp_model_match_classification"
NEW_COLUMNS = (
    MINLP_SOLVER_NAME_COLUMN,
    MINLP_STATUS_COLUMN,
    MINLP_NODE_COUNT_COLUMN,
    MINLP_CPU_SECONDS_COLUMN,
    MINLP_WALL_SECONDS_COLUMN,
    MINLP_OBJECTIVE_COLUMN,
    MINLP_INDEPENDENT_OBJECTIVE_COLUMN,
    MINLP_ABS_DIFF_COLUMN,
    MINLP_REL_DIFF_COLUMN,
    MINLP_MATCH_COLUMN,
)

# Cbc's (the B&B engine Bonmin uses internally) standard end-of-solve
# summary line, confirmed against a real Bonmin run in this project (a
# time-limited strongly-correlated n=100 instance):
#
#   Cbc0005I Partial search - best objective -4104.517 (best possible
#   -4347.4381), took 1344951 iterations and 60036 nodes (3600.07 seconds)
#
# The "took N iterations and M nodes (T seconds)" tail is Cbc's shared,
# long-stable phrasing for both a limit-terminated run ("Partial search",
# as above) and a proven-optimal one ("Cbc0001I Search completed - ...",
# not yet observed directly in this project but documented Cbc phrasing),
# so this matches on that common tail rather than the varying prefix.
# Older PARSE_NODE_COUNT_PATTERNS-only fallbacks are kept after it in case
# some other Bonmin/Cbc build or version phrases this differently.
CBC_SUMMARY_PATTERN = re.compile(
    r"took\s+(?P<iterations>[\d,]+)\s+iterations\s+and\s+(?P<nodes>[\d,]+)\s+nodes"
    r"\s*\(\s*(?P<seconds>[\d.]+)\s+seconds\s*\)",
    re.IGNORECASE,
)
PARSE_NODE_COUNT_PATTERNS = (
    re.compile(r"Enumerated\s+nodes\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"Total\s+nodes\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"Number\s+of\s+Nodes\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bnodes\s*=\s*(\d+)", re.IGNORECASE),
)
PARSE_CPU_SECONDS_PATTERNS = (
    re.compile(r"Time\s*\(CPU\s*seconds\)\s*:\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"CPU\s+[Tt]ime\s*:\s*([\d.]+)", re.IGNORECASE),
)

# SCIP's own end-of-solve statistics block (SCIPprintStatistics), which its
# AMPL/.nl driver prints by default just like Bonmin's console trace does.
# CONFIRMED against a real SCIP 10.0.3 run in this project (n=100
# strongly-correlated instance, stopped at the configured 0.1% gap rather
# than proven exact optimality):
#
#   SCIP Status        : solving was interrupted [gap limit reached]
#   Solving Time (sec) : 8.00
#   Solving Nodes      : 251
#   Primal Bound       : +4.41735440677722e+03 (119 solutions)
#   Dual Bound         : +4.42109157397012e+03
#   Gap                : 0.08 %
#
# SCIP_NODE_COUNT_PATTERNS / SCIP_CPU_SECONDS_PATTERNS matched this
# correctly on the first try -- no adjustment needed. SCIP_STATUS_PATTERN
# extracts the "SCIP Status" line's human-readable text, used instead of
# Pyomo's own TerminationCondition enum for reporting (see
# solve_and_collect_minlp_stats): Pyomo's ASL result parser maps this
# "gap limit reached" outcome (SCIP correctly stopping once within our
# configured limits/gap, NOT a failure) to its generic 'other' bucket
# rather than 'optimal', so gating objective extraction on
# termination_condition == optimal would incorrectly discard a perfectly
# good, gap-respecting solution.
SCIP_NODE_COUNT_PATTERNS = (
    re.compile(r"Solving\s+Nodes\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"number\s+of\s+nodes\s*(?:\(total\))?\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"nodes\s*\(total\)\s*:\s*(\d+)", re.IGNORECASE),
)
SCIP_CPU_SECONDS_PATTERNS = (
    re.compile(r"Solving\s+Time\s*\(sec\)\s*:\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"Total\s+Time\s*:\s*([\d.]+)", re.IGNORECASE),
)
SCIP_STATUS_PATTERN = re.compile(r"SCIP\s+Status\s*:\s*(.+)", re.IGNORECASE)


def parse_scip_status_line(log_text: str) -> str | None:
    """Extract SCIP's own human-readable status line (e.g. "solving was
    interrupted [gap limit reached]" or "problem is solved [optimal
    solution found]") -- confirmed format, see SCIP_STATUS_PATTERN's
    comment above. Returns None if not found (never raises).
    """
    matches = list(SCIP_STATUS_PATTERN.finditer(log_text))
    return matches[-1].group(1).strip() if matches else None

SINGLE_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def force_single_threaded_environment() -> None:
    """Set the linear-algebra threading env vars the solver's subprocess
    inherits, so the underlying BLAS/LAPACK calls run single-threaded (for
    Bonmin, at each Ipopt NLP-relaxation node). Must be called before
    solving. Neither Bonmin's B&B (Cbc) nor a default single-mode SCIP
    build has general parallel tree search enabled to separately disable;
    this only guards the linear-algebra layer underneath.
    """
    for var in SINGLE_THREAD_ENV_VARS:
        os.environ[var] = "1"


def _conda_environment_root(executable: str) -> Path | None:
    """If ``executable`` sits at ``<env_root>/Library/bin/<name>.exe`` (the
    standard conda-forge Windows package layout), return <env_root>;
    otherwise None.
    """
    path = Path(executable).resolve()
    if path.parent.name.lower() == "bin" and path.parent.parent.name.lower() == "library":
        return path.parent.parent.parent
    return None


def augment_path_for_conda_executable(executable: str | None) -> None:
    """conda-forge Windows packages (this project's SCIP build in
    particular) load several sibling DLLs -- Ipopt, SoPlex, the MinGW-w64
    C/C++/Fortran runtime -- that live in OTHER <env_root>/Library/*
    subdirectories, not next to the executable itself. Launching the
    executable outside an activated conda environment (as Pyomo's
    subprocess does by default) fails with a Windows STATUS_DLL_NOT_FOUND
    error (exit code 3221225781 / 0xC0000135) unless those directories are
    also on PATH.

    Confirmed necessary in this project: scip.exe (conda-forge win-64,
    both the 8.0.0 build in the base environment and the 10.0.3 build in
    scip10_env) fails to start at all without Library/mingw-w64/bin on
    PATH, even though ipopt-3.dll itself IS present in Library/bin --
    the actual missing dependency is transitive (the MinGW-w64 runtime
    Ipopt/SoPlex are built against), which Windows' error reporting
    doesn't name directly. This mirrors what conda's own activate.bat
    adds to PATH; only that subset is reproduced here.

    A no-op when ``executable`` isn't inside a conda environment's
    Library/bin (e.g. Bonmin, installed via the ampl_module_coin pip
    package rather than conda) or is None.
    """
    if executable is None:
        return
    root = _conda_environment_root(executable)
    if root is None:
        return
    conda_dirs = [
        str(root),
        str(root / "Library" / "mingw-w64" / "bin"),
        str(root / "Library" / "usr" / "bin"),
        str(root / "Library" / "bin"),
        str(root / "Scripts"),
        str(root / "bin"),
    ]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(conda_dirs) + os.pathsep + existing_path


def _search_last(patterns: Sequence[re.Pattern[str]], log_text: str) -> str | None:
    for pattern in patterns:
        matches = list(pattern.finditer(log_text))
        if matches:
            return matches[-1].group(1)
    return None


def parse_bonmin_log(log_text: str) -> tuple[int | None, float | None]:
    """Best-effort extraction of (node_count, cpu_seconds) from a captured
    Bonmin/Cbc solver log. Tries CBC_SUMMARY_PATTERN (confirmed against a
    real run) first, taking its LAST match in case the log contains more
    than one such line; falls back to the older per-field guess patterns
    if that doesn't match. Returns (None, None) for anything not found;
    never raises on an unparsed log, since this is diagnostic, not
    solution-correctness-affecting, data.
    """
    summary_matches = list(CBC_SUMMARY_PATTERN.finditer(log_text))
    if summary_matches:
        match = summary_matches[-1]
        node_count = int(match.group("nodes").replace(",", ""))
        cpu_seconds = float(match.group("seconds"))
        return node_count, cpu_seconds

    nodes_text = _search_last(PARSE_NODE_COUNT_PATTERNS, log_text)
    cpu_text = _search_last(PARSE_CPU_SECONDS_PATTERNS, log_text)
    node_count = int(nodes_text.replace(",", "")) if nodes_text is not None else None
    cpu_seconds = float(cpu_text) if cpu_text is not None else None
    return node_count, cpu_seconds


def parse_scip_log(log_text: str) -> tuple[int | None, float | None]:
    """Best-effort extraction of (node_count, cpu_seconds) from a captured
    SCIP solver log -- see SCIP_NODE_COUNT_PATTERNS / SCIP_CPU_SECONDS_
    PATTERNS docstring above on why this is NOT yet validated against a
    real SCIP run. Same never-raises, take-the-last-match contract as
    parse_bonmin_log.
    """
    nodes_text = _search_last(SCIP_NODE_COUNT_PATTERNS, log_text)
    cpu_text = _search_last(SCIP_CPU_SECONDS_PATTERNS, log_text)
    node_count = int(nodes_text.replace(",", "")) if nodes_text is not None else None
    cpu_seconds = float(cpu_text) if cpu_text is not None else None
    return node_count, cpu_seconds


def parse_minlp_solver_log(log_text: str, solver: str) -> tuple[int | None, float | None]:
    """Dispatch to the right (node_count, cpu_seconds) log parser for
    ``solver`` (one of MINLP_SOLVER_CHOICES). Returns (None, None) for an
    unrecognized solver name rather than raising, matching the
    never-raises-on-an-unparsed-log contract of the parsers it dispatches
    to -- this is diagnostic data, not solution-correctness-affecting.
    """
    solver = solver.lower()
    if solver == "bonmin":
        return parse_bonmin_log(log_text)
    if solver == "scip":
        return parse_scip_log(log_text)
    return None, None


def solve_and_collect_minlp_stats(
    instance: rte.TriangularInstance,
    *,
    solver: str = DEFAULT_SOLVER,
    executable: str | None,
    time_limit_seconds: float,
    mip_gap: float,
    log_path: Path,
) -> dict[str, Any]:
    """Build the un-lifted cubic MINLP (build_triangular_2018_fixed_cost,
    above) for ``instance`` and solve it with ``solver`` (one of
    MINLP_SOLVER_CHOICES: "bonmin" or "scip"), capturing CPU time and B&B
    node count from its console log in addition to the wall-clock time and
    success/status solve_triangular_2018_fixed_cost's own Pyomo result
    already reports. The SAME Pyomo model/build path is used regardless of
    solver -- only the option names (minlp_solver_options) and log format
    (parse_minlp_solver_log) differ.
    """
    formulation = build_triangular_2018_fixed_cost(
        instance.a, instance.m, instance.b,
        instance.price_p, instance.cost_c, instance.salvage_s,
        instance.x_u, instance.k, instance.C, instance.K,
    )

    if executable is None and not _solver_executable_present(solver):
        return {
            "solver_name": solver.upper(),
            "status": "UNAVAILABLE",
            "node_count": None,
            "cpu_seconds": None,
            "wall_seconds": None,
            "objective": None,
            "independent_objective": None,
        }

    augment_path_for_conda_executable(executable)
    solver_object = (
        pyo.SolverFactory(solver, executable=executable) if executable is not None
        else pyo.SolverFactory(solver)
    )
    for option_name, option_value in minlp_solver_options(
        solver, time_limit_seconds=time_limit_seconds, mip_gap=mip_gap
    ).items():
        solver_object.options[option_name] = option_value

    force_single_threaded_environment()
    start = time.perf_counter()
    results = solver_object.solve(formulation.model, tee=False, logfile=str(log_path))
    wall_seconds = time.perf_counter() - start

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    node_count, cpu_seconds = parse_minlp_solver_log(log_text, solver)

    termination = results.solver.termination_condition
    status = str(termination)
    if solver == "scip":
        # SCIP's own status text (e.g. "solving was interrupted [gap limit
        # reached]") is far more informative than Pyomo's generic
        # TerminationCondition enum, which has no specific bucket for
        # "stopped once within our configured gap" and lumps it under
        # 'other' -- see SCIP_STATUS_PATTERN's comment above.
        scip_status = parse_scip_status_line(log_text)
        if scip_status is not None:
            status = scip_status

    # The raw Pyomo-reported objective trusts whatever (bar_x, hat_x,
    # tilde_x) the solver returns; independent_objective instead recomputes
    # the SAME closed-form profit function (7) run_triangular_table2_
    # experiment.py and run_triangular_strongly_correlated.py use, directly
    # from the physical order quantities q = bar_x + hat_x + tilde_x. Since
    # this model has no lift variables at all, the two should already agree
    # almost exactly for a trustworthy solution; computing both anyway keeps
    # this comparison apples-to-apples with how GUROBI's independent_
    # objective column is computed.
    #
    # Extraction is attempted unconditionally (not gated on termination_
    # condition == optimal) and relies on the try/except below to fall back
    # to None/None when no solution was actually loaded (pyo.value() on an
    # uninitialized Var raises ValueError) -- matching how the GUROBI path
    # elsewhere in this project determines has_incumbent from whether
    # values are present, not from a strict status-string match. This
    # matters concretely for SCIP: a "gap limit reached" solve (the normal,
    # desired outcome once within our configured mip_gap, not a failure)
    # reports termination_condition == other, and gating on == optimal
    # would incorrectly discard that perfectly good solution.
    objective = None
    independent_objective = None
    try:
        bar = [pyo.value(formulation.model.bar_x[i]) for i in range(instance.n)]
        hat = [pyo.value(formulation.model.hat_x[i]) for i in range(instance.n)]
        tilde = [pyo.value(formulation.model.tilde_x[i]) for i in range(instance.n)]
        q = [bar[i] + hat[i] + tilde[i] for i in range(instance.n)]
        objective = float(pyo.value(formulation.model.objective))
        independent_objective = rte.recompute_triangular_expected_profit(instance, q)
    except (ValueError, TypeError):
        pass

    return {
        "solver_name": solver.upper(),
        "status": status,
        "node_count": node_count,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "objective": objective,
        "independent_objective": independent_objective,
    }


def _parse_optional_float(value: Any) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: Any) -> int | None:
    parsed = _parse_optional_float(value)
    return None if parsed is None else int(round(parsed))


def _parse_repr_float(value: Any) -> float | None:
    """Like _parse_optional_float, but for fields read out of
    triangular_table2_solver_records.csv, where a missing value was
    written as the literal 4-character text 'None' (repr(None)) rather
    than an empty CSV cell -- see append_solver_record_csv in
    run_triangular_table2_experiment.py.
    """
    return _parse_optional_float(value)


def _parse_repr_int(value: Any) -> int | None:
    return _parse_optional_int(value)


def compare_minlp_to_row(
    minlp_independent_objective: float | None,
    row: Mapping[str, Any],
) -> tuple[float | None, float | None, str]:
    """Classify the MINLP solver's (Bonmin or SCIP) independent_objective
    against THIS row's own independent_objective (whichever GUROBI
    model_id the row represents), using the exact same tolerances/formula
    run_triangular_table2_experiment.py uses for its own Big-M vs.
    Perspective comparison (scaled_tolerance with
    DEFAULT_MODEL_MATCH_ABS_TOL/_REL_TOL, PASS iff absolute difference <=
    that threshold). This is a genuinely cross-solver comparison
    (Bonmin/SCIP vs. GUROBI); see compare_bigm_vs_perspective for the
    separate, same-solver comparison.
    """
    row_objective = _parse_optional_float(row.get("independent_objective"))
    row_status = row.get("status")
    if minlp_independent_objective is None:
        return None, None, "NOT_COMPARABLE_DUE_TO_MINLP_SOLVER_STATUS"
    if row_objective is None or row_status != "optimal":
        return None, None, "NOT_COMPARABLE_DUE_TO_GUROBI_STATUS"
    absolute = abs(minlp_independent_objective - row_objective)
    relative = rte.relative_difference(minlp_independent_objective, row_objective)
    threshold = rte.scaled_tolerance(
        rte.DEFAULT_MODEL_MATCH_ABS_TOL,
        rte.DEFAULT_MODEL_MATCH_REL_TOL,
        minlp_independent_objective,
        row_objective,
    )
    classification = "PASS" if absolute <= threshold else "FAIL"
    return absolute, relative, classification


def load_gurobi_reference_table2(
    n_values: Sequence[int], replications: Sequence[int], base: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest GUROBI row per (model_id, instance_id) from the official,
    frozen Table 2 log (triangular_table2_results/triangular_table2_
    solver_records.csv), normalized into the same shape
    load_gurobi_reference_strongly_correlated returns (status/objective/
    independent_objective/node_count/solve_seconds), so the rest of this
    script's comparison/merge logic is dataset-agnostic. This file is only
    READ here, never written to.
    """
    path = base / "triangular_table2_results" / "triangular_table2_solver_records.csv"
    wanted_ids = {f"TRIANGULAR-N{n:04d}-R{r:03d}" for n in n_values for r in replications}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["solver_requested"].strip("'") != "GUROBI":
                continue
            instance_id = row["instance_id"].strip("'")
            if instance_id not in wanted_ids:
                continue
            model_id = row["model_id"].strip("'")
            key = (model_id, instance_id)
            if key in latest and row["started_at_utc"] <= latest[key]["started_at_utc"]:
                continue
            raw_status = row["solver_status"].strip("'")
            latest[key] = {
                "instance_id": instance_id,
                "n": int(row["n"]),
                "replication": int(row["replication"]),
                "model_id": model_id,
                "status": "optimal" if raw_status == "OPTIMAL" else raw_status.lower(),
                "objective": _parse_repr_float(row["objective_solver_reported"]),
                "independent_objective": _parse_repr_float(row["objective_independent_common"]),
                "node_count": _parse_repr_int(row["node_count"]),
                "solve_seconds": _parse_repr_float(row["wall_solve_and_unpack_seconds"]),
                "started_at_utc": row["started_at_utc"],
            }
    return latest


def load_gurobi_reference_strongly_correlated(
    n_values: Sequence[int], replications: Sequence[int], results_csv: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest GUROBI row per (model_id, instance_id) already sitting in
    strongly_correlated_table2_results.csv (written by
    run_triangular_strongly_correlated.py), normalized to the same shape
    load_gurobi_reference_table2 returns.
    """
    wanted_ids = {f"TRIANGULAR-N{n:04d}-R{r:03d}" for n in n_values for r in replications}
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    if not results_csv.exists():
        return reference
    with results_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("solver") != "GUROBI" or row["instance_id"] not in wanted_ids:
                continue
            key = (row["model_id"], row["instance_id"])
            reference[key] = {
                "instance_id": row["instance_id"],
                "n": int(row["n"]),
                "replication": int(row["replication"]),
                "model_id": row["model_id"],
                "status": row["status"],
                "objective": _parse_optional_float(row.get("objective")),
                "independent_objective": _parse_optional_float(row.get("independent_objective")),
                "node_count": _parse_optional_int(row.get("node_count")),
                "solve_seconds": _parse_optional_float(row.get("solve_seconds")),
            }
    return reference


def seed_base_rows_for_table2(
    results_path: Path, gurobi_reference: dict[tuple[str, str], dict[str, Any]]
) -> None:
    """Create/update triangular_table2_bonmin_comparison.csv with one row
    per (model_id, instance_id) pulled from the GUROBI reference, upserting
    by that key so repeated/incremental runs (e.g. a broader --n-values
    selection later) don't clobber rows for instances outside this run's
    scope. merge_minlp_columns_into_results_csv then adds the MINLP solver's
    columns on top in the usual way. (For --dataset strongly_correlated,
    this seeding step is unnecessary: run_triangular_strongly_correlated.py
    already writes GUROBI's own rows into that dataset's results file.)
    """
    existing_rows: list[dict[str, Any]] = []
    existing_fieldnames: tuple[str, ...] = ()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            existing_fieldnames = tuple(reader.fieldnames or ())
            existing_rows = list(reader)
    # Union with RESULT_FIELDNAMES rather than overwriting: existing_rows
    # may already carry MINLP-solver columns (MINLP_SOLVER_NAME_COLUMN etc.)
    # from a prior merge_minlp_columns_into_results_csv call, e.g. from a
    # run at a different n. Writing out only RESULT_FIELDNAMES here would
    # silently drop those columns for every row in the file, not just the
    # ones this call's gurobi_reference touches -- this previously erased
    # an earlier Bonmin run's results the next time this function ran (for
    # a different --n-values selection).
    all_fieldnames = tuple(RESULT_FIELDNAMES) + tuple(
        field for field in existing_fieldnames if field not in RESULT_FIELDNAMES
    )
    existing_by_key = {(row["model_id"], row["instance_id"]): row for row in existing_rows}
    for key, ref in gurobi_reference.items():
        row = existing_by_key.get(key, {})
        row.update({
            "instance_id": ref["instance_id"],
            "n": ref["n"],
            "replication": ref["replication"],
            "model_id": ref["model_id"],
            "solver": "GUROBI",
            "status": ref["status"],
            "objective": ref["objective"],
            "independent_objective": ref["independent_objective"],
            "node_count": ref["node_count"],
            "solve_seconds": ref["solve_seconds"],
        })
        existing_by_key[key] = row

    with results_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in existing_by_key.values():
            writer.writerow({field: row.get(field) for field in all_fieldnames})


def merge_minlp_columns_into_results_csv(
    results_path: Path,
    minlp_by_instance: dict[str, dict[str, Any]],
) -> None:
    """Add/overwrite the MINLP-solver columns on every existing row whose
    instance_id has a result (including the objective-match comparison
    against that row's own independent_objective), and append a new row
    (MINLP-only fields populated, GUROBI fields blank) for any instance
    with a result but no existing row at all (e.g. GUROBI hasn't been run
    yet for that instance). Works identically regardless of which solver
    (bonmin/scip) produced ``minlp_by_instance`` -- that's recorded per-row
    in MINLP_SOLVER_NAME_COLUMN.
    """
    existing_rows: list[dict[str, Any]] = []
    fieldnames: tuple[str, ...] = ()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            existing_rows = list(reader)

    base_fieldnames = fieldnames or RESULT_FIELDNAMES
    merged_fieldnames = tuple(base_fieldnames) + tuple(
        column for column in NEW_COLUMNS if column not in base_fieldnames
    )

    matched_instance_ids: set[str] = set()
    for row in existing_rows:
        stats = minlp_by_instance.get(row["instance_id"])
        if stats is None:
            continue
        matched_instance_ids.add(row["instance_id"])
        absolute, relative, classification = compare_minlp_to_row(
            stats["independent_objective"], row
        )
        row[MINLP_SOLVER_NAME_COLUMN] = stats["solver_name"]
        row[MINLP_STATUS_COLUMN] = stats["status"]
        row[MINLP_NODE_COUNT_COLUMN] = stats["node_count"]
        row[MINLP_CPU_SECONDS_COLUMN] = stats["cpu_seconds"]
        row[MINLP_WALL_SECONDS_COLUMN] = stats["wall_seconds"]
        row[MINLP_OBJECTIVE_COLUMN] = stats["objective"]
        row[MINLP_INDEPENDENT_OBJECTIVE_COLUMN] = stats["independent_objective"]
        row[MINLP_ABS_DIFF_COLUMN] = absolute
        row[MINLP_REL_DIFF_COLUMN] = relative
        row[MINLP_MATCH_COLUMN] = classification

    new_rows: list[dict[str, Any]] = []
    for instance_id, stats in minlp_by_instance.items():
        if instance_id in matched_instance_ids:
            continue
        n_str, r_str = instance_id.replace("TRIANGULAR-N", "").split("-R")
        row = {field: None for field in merged_fieldnames}
        row.update({
            "instance_id": instance_id,
            "n": int(n_str),
            "replication": int(r_str),
            "model_id": MODEL_ID,
            "solver": stats["solver_name"],
            MINLP_SOLVER_NAME_COLUMN: stats["solver_name"],
            MINLP_STATUS_COLUMN: stats["status"],
            MINLP_NODE_COUNT_COLUMN: stats["node_count"],
            MINLP_CPU_SECONDS_COLUMN: stats["cpu_seconds"],
            MINLP_WALL_SECONDS_COLUMN: stats["wall_seconds"],
            MINLP_OBJECTIVE_COLUMN: stats["objective"],
            MINLP_INDEPENDENT_OBJECTIVE_COLUMN: stats["independent_objective"],
            MINLP_MATCH_COLUMN: "NOT_COMPARABLE_NO_GUROBI_ROW",
        })
        new_rows.append(row)

    all_rows = existing_rows + new_rows
    with results_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field) for field in merged_fieldnames})


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("table2", "strongly_correlated"), default="table2",
        help=(
            "table2 (default): original manuscript (equation-13) instances; GUROBI reference read "
            "from triangular_table2_solver_records.csv (never written to); Bonmin results/comparison "
            "written to triangular_table2_bonmin_comparison.csv. "
            "strongly_correlated: the harder minimal-shift instances; GUROBI reference and Bonmin "
            "results/comparison both live in strongly_correlated_table2_results.csv."
        ),
    )
    parser.add_argument(
        "--minlp-solver", choices=MINLP_SOLVER_CHOICES, default=DEFAULT_SOLVER,
        help=(
            "which Pyomo/AMPL-interface MINLP solver to solve the un-lifted cubic model with "
            "(same Pyomo model either way; only solver options and log-parsing differ). "
            "'scip' is not yet validated against a real run in this project -- see "
            "minlp_solver_options()/parse_scip_log() docstrings."
        ),
    )
    parser.add_argument("--n-values", default="all")
    parser.add_argument("--replications", default="all")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=1.0e-3)
    parser.add_argument(
        "--executable", default=None,
        help="explicit path to the --minlp-solver executable (bonmin.exe / scip.exe)",
    )
    parser.add_argument(
        "--instances-csv", default=None,
        help="override the default instances CSV for the selected --dataset",
    )
    parser.add_argument(
        "--results-csv", default=None,
        help="override the default results CSV for the selected --dataset",
    )
    parser.add_argument("--log-dir", default="bonmin_logs")
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the in-memory model self-test (build + tiny Bonmin solve + brute-force "
        "optimality cross-check) instead of the comparison experiment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return 0

    n_values = rte.parse_int_selection(args.n_values, rte.TABLE2_N_VALUES, "n")
    replications = rte.parse_int_selection(args.replications, rte.TABLE2_REPLICATIONS, "replication")
    base = Path(__file__).parent

    if args.dataset == "table2":
        instances_csv = base / (args.instances_csv or TABLE2_INSTANCES_CSV)
        instances = list(rte.read_selected_triangular_table2_instances(instances_csv, n_values, replications))
        results_path = base / (args.results_csv or TABLE2_RESULTS_CSV)
        gurobi_reference = load_gurobi_reference_table2(n_values, replications, base)
        seed_base_rows_for_table2(results_path, gurobi_reference)
    else:
        instances_csv = base / (args.instances_csv or STRONGLY_CORRELATED_INSTANCES_CSV)
        instances = gsc.read_selected_strongly_correlated_instances(instances_csv, n_values, replications)
        results_path = base / (args.results_csv or STRONGLY_CORRELATED_RESULTS_CSV)

    log_dir = base / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    minlp_by_instance: dict[str, dict[str, Any]] = {}
    for instance in instances:
        log_path = log_dir / f"{instance.instance_id}.{args.minlp_solver}.log"
        stats = solve_and_collect_minlp_stats(
            instance,
            solver=args.minlp_solver,
            executable=args.executable,
            time_limit_seconds=args.time_limit,
            mip_gap=args.mip_gap,
            log_path=log_path,
        )
        minlp_by_instance[instance.instance_id] = stats
        print(
            f"{instance.instance_id}: {args.minlp_solver} status={stats['status']} "
            f"nodes={stats['node_count']} cpu_s={stats['cpu_seconds']} wall_s={stats['wall_seconds']}"
        )
        if stats["status"] == "UNAVAILABLE":
            print(
                f"{args.minlp_solver} executable not found; stopping (install it or pass --executable). "
                f"{len(minlp_by_instance) - 1} instance(s) solved before this one will still be merged."
            )
            break

    merge_minlp_columns_into_results_csv(results_path, minlp_by_instance)
    print(f"Merged {len(minlp_by_instance)} {args.minlp_solver} result(s) into {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
