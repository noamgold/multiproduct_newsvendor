"""CVXPY reconstructions of the manuscript's current Triangular-demand models.

This module used to hold Pyomo ``ConcreteModel`` builders that depended on a
``python_replication`` package (``.common`` / ``.construction_data``) that is
not part of this project directory, so it could not be imported or run on its
own. It has been rewritten to be self-contained: it depends only on NumPy and
CVXPY, and it builds the same three models algebraically (the continuous
conic model of Section 5/6, and the Big-M/perspective fixed-cost models of
formulation (9) and its perspective reformulation, replacing (9c) with (5)).

Every model is expressed with a *linear* objective (matching manuscript
equation (10)) and linear/second-order-cone constraints only, so CVXPY can
hand it directly to a conic mixed-integer solver. Three such solvers are
supported, CPLEX, MOSEK, and GUROBI; which one is used is controlled by a
single module-level parameter, ``DEFAULT_SOLVER``, unless a caller overrides
it explicitly. GUROBI was added to experiment with it as an alternative to
MOSEK, which was observed (see run_triangular_table2_experiment.py's Table 2
reproduction) to have a much less efficient MISOCP branch-and-bound than
CPLEX on this formulation -- Gurobi's own MISOCP implementation is generally
considered strong and is worth comparing directly.

Typical use from another script (e.g. run_triangular_table2_experiment.py):

    from triangular_current import (
        build_current_triangular_bigm_fixed_cost,
        build_current_triangular_perspective_fixed_cost,
    )

    bundle = build_current_triangular_bigm_fixed_cost(a, m, b, p, c, s, x_u, k, C, K)
    bundle.problem.solve(solver="CPLEX")
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import cvxpy as cp
except ImportError as error:  # pragma: no cover - import guard only
    raise ImportError(
        "triangular_current requires cvxpy to be installed (pip install cvxpy)."
    ) from error


# ---------------------------------------------------------------------------
# GLOBAL SOLVER SELECTION
#
# This single parameter decides which conic solver backend is used whenever a
# caller does not pass an explicit ``solver=`` argument to solve_current_
# triangular_model(). Valid values are "CPLEX", "MOSEK", and "GUROBI".
# ---------------------------------------------------------------------------
DEFAULT_SOLVER = "CPLEX"

SUPPORTED_SOLVERS = ("CPLEX", "MOSEK", "GUROBI")

CONTINUOUS_MODEL_ID = "CURRENT_TRIANGULAR_CONIC_CONTINUOUS"
BIGM_MODEL_ID = "CURRENT_TRIANGULAR_BIGM_FIXED_COST"
PERSPECTIVE_MODEL_ID = "CURRENT_TRIANGULAR_PERSPECTIVE_FIXED_COST"
# The perspective model plus the manuscript revision's valid inequality
# (15i), u_hat_i <= (min{x^U_i, m_i} - a_i)^2. NOTE ON NUMBERING: the
# revision renumbers the triangular formulation (9) -> (15) and the
# perspective replacement constraint (5) -> (11), so "(15c) replaced by
# (11)" names the same model this file already builds as
# PERSPECTIVE_MODEL_ID. Only (15i) is new.
PERSPECTIVE_BOUNDED_MODEL_ID = "CURRENT_TRIANGULAR_PERSPECTIVE_15I_FIXED_COST"

MODEL_LABELS = {
    BIGM_MODEL_ID: "Big-M (9)",
    PERSPECTIVE_MODEL_ID: "Conic (9) with (9c) replaced by (5)",
    PERSPECTIVE_BOUNDED_MODEL_ID: "Conic perspective + (15i)",
}


class TriangularModelError(RuntimeError):
    """Raised when input data or solver configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TriangularModelError(message)


def _as_vector(values: Sequence[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    _require(array.shape == (n,), f"{name} must have shape ({n},); observed {array.shape}")
    _require(bool(np.all(np.isfinite(array))), f"{name} contains a non-finite value")
    return array


@dataclasses.dataclass(slots=True)
class TriangularFormulation:
    """A built CVXPY problem plus the variables a caller needs after solving."""

    problem: Any
    variables: dict[str, Any]
    model_id: str


def _add_rotated_cone(constraints: list[Any], x: Any, u: Any, v: Any) -> None:
    """Add x_i^2 <= u_i * v_i for each i, with x, u >= 0 and v >= 0.

    ``v`` may be a Python float (the constant 1.0, used for a plain hyperbolic
    epigraph) or a length-matching CVXPY expression (e.g. a binary indicator
    vector z, for the perspective lift, or another variable vector, for the
    cube-power tower). This is the rotated-second-order-cone identity from
    manuscript Appendix A:

        ||(2*x_i, u_i - v_i)|| <= u_i + v_i   <=>   x_i^2 <= u_i * v_i.
    """
    n = x.shape[0]
    for i in range(n):
        v_i = v if isinstance(v, (int, float)) else v[i]
        constraints.append(cp.SOC(u[i] + v_i, cp.hstack([2.0 * x[i], u[i] - v_i])))


def _triangular_objective(
    a: np.ndarray,
    m: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    bar_x: Any,
    hat_x: Any,
    y: Any,
    w_hat: Any,
    w_y: Any,
) -> Any:
    """The linear objective (10)/(9a): sum_i pi-hat_i(bar_x_i, hat_x_i, y_i, w_hat_i, w_y_i)."""
    coeff_q = p - c
    coeff_y = -(s - c)
    coeff3 = (s - p) / (3.0 * (b - a))
    wy_coeff = coeff3 / (b - m)
    wh_coeff = coeff3 / (m - a)
    constant_term = float(
        np.sum((b - m) * (p - c - (s - p) * (3.0 * a - 2.0 * b - m) / (3.0 * (b - a))))
    )
    return (
        coeff_q @ (bar_x + hat_x)
        + coeff_y @ y
        + wy_coeff @ w_y
        + wh_coeff @ w_hat
        + constant_term
    )


def _validate_support(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> None:
    _require(bool(np.all(a >= 0.0)), "a must be nonnegative")
    _require(bool(np.all(a < m)), "a must be strictly less than m")
    _require(bool(np.all(m < b)), "m must be strictly less than b")


def build_current_triangular_conic_continuous(
    a: Sequence[float],
    m: Sequence[float],
    b: Sequence[float],
    price_p: Sequence[float],
    cost_c: Sequence[float],
    salvage_s: Sequence[float],
    C: float,
) -> TriangularFormulation:
    """The continuous SOCP of Section 6 (formulation (9) with k_i = 0, no z)."""
    n = len(a)
    a, m, b = (_as_vector(v, n, name) for v, name in ((a, "a"), (m, "m"), (b, "b")))
    p = _as_vector(price_p, n, "price_p")
    c = _as_vector(cost_c, n, "cost_c")
    s = _as_vector(salvage_s, n, "salvage_s")
    _validate_support(a, m, b)
    _require(np.isfinite(C) and C >= 0.0, "C must be finite and nonnegative")

    bar_x = cp.Variable(n, nonneg=True, name="bar_x")
    hat_x = cp.Variable(n, nonneg=True, name="hat_x")
    y = cp.Variable(n, nonneg=True, name="y")
    u_hat = cp.Variable(n, nonneg=True, name="u_hat")
    w_hat = cp.Variable(n, nonneg=True, name="w_hat")
    u_y = cp.Variable(n, nonneg=True, name="u_y")
    w_y = cp.Variable(n, nonneg=True, name="w_y")

    tilde_x = b - m - y
    q = bar_x + hat_x + tilde_x

    constraints: list[Any] = [
        bar_x <= a,
        hat_x <= m - a,
        y <= b - m,
        cp.sum(q) <= C,
    ]
    _add_rotated_cone(constraints, hat_x, u_hat, 1.0)
    _add_rotated_cone(constraints, u_hat, w_hat, hat_x)
    _add_rotated_cone(constraints, y, u_y, 1.0)
    _add_rotated_cone(constraints, u_y, w_y, y)

    objective = _triangular_objective(a, m, b, p, c, s, bar_x, hat_x, y, w_hat, w_y)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    _require(problem.is_dcp(), "continuous Triangular CVXPY model is not DCP")
    return TriangularFormulation(
        problem=problem,
        variables={
            "bar_x": bar_x, "hat_x": hat_x, "y": y,
            "u_hat": u_hat, "w_hat": w_hat, "u_y": u_y, "w_y": w_y,
            "q": q,
        },
        model_id=CONTINUOUS_MODEL_ID,
    )


def _build_triangular_fixed_cost_model(
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
    *,
    perspective: bool,
    include_u_hat_bound: bool = False,
) -> TriangularFormulation:
    """Formulation (9) (perspective=False) or its perspective reformulation
    (perspective=True), which replaces the hyperbolic constraint (9c) by the
    bilinear/perspective constraint (5), following manuscript Section 5.

    ``include_u_hat_bound`` adds the manuscript revision's valid inequality
    (15i), u_hat_i <= (min{x^U_i, m_i} - a_i)^2 -- see the comment at the
    constraint itself. It is only accepted together with perspective=True,
    which is where the revision states it.
    """
    _require(
        perspective or not include_u_hat_bound,
        "(15i) is specified only for the perspective formulation",
    )
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

    z = cp.Variable(n, boolean=True, name="z")
    bar_x = cp.Variable(n, nonneg=True, name="bar_x")
    y_scaled = cp.Variable(n, nonneg=True, name="y_scaled")
    hat_x_scaled = cp.Variable(n, nonneg=True, name="hat_x_scaled")
    u_hat_scaled = cp.Variable(n, nonneg=True, name="u_hat_scaled")
    w_hat_scaled = cp.Variable(n, nonneg=True, name="w_hat_scaled")
    u_y_scaled = cp.Variable(n, nonneg=True, name="u_y_scaled")
    w_y_scaled = cp.Variable(n, nonneg=True, name="w_y_scaled")

    # Big-M constants tightened by Proposition 1 (x* <= x^U componentwise).
    hat_bigm = np.maximum(0.0, np.minimum(x_u, m) - a)
    tilde_bigm = np.maximum(0.0, x_u - m)

    # The cube-tower cones below force w_hat/w_y (representing hat_x**3 and
    # y**3 at optimality) up to O(1e6-1e7) while hat_x/y themselves stay
    # O(1e2), which left GUROBI's default (and even substantially tightened)
    # numerical tolerances unable to keep the returned solution within the
    # validator's feasibility tolerance -- see run_triangular_table2_
    # experiment.py's Table 2 validation failures on this formulation.
    # Per-product characteristic scales (each variable's own natural upper
    # bound, floored at 1 to stay finite when a product's segment is empty)
    # turn every cone below into the dimensionless identity hhat**2<=uu*v,
    # uu**2<=ww*hhat, which keeps every solved variable within a couple of
    # orders of magnitude of 1 regardless of the instance's physical scale.
    hat_scale = np.maximum(hat_bigm, 1.0)
    y_scale = np.maximum(b - m, 1.0)

    hat_x = cp.multiply(hat_scale, hat_x_scaled)
    y = cp.multiply(y_scale, y_scaled)
    u_hat = cp.multiply(hat_scale**2, u_hat_scaled)
    w_hat = cp.multiply(hat_scale**3, w_hat_scaled)
    u_y = cp.multiply(y_scale**2, u_y_scaled)
    w_y = cp.multiply(y_scale**3, w_y_scaled)

    tilde_x = b - m - y
    q = bar_x + hat_x + tilde_x

    constraints: list[Any] = [
        bar_x <= cp.multiply(a, z),
        hat_x <= cp.multiply(hat_bigm, z),
        y <= b - m,
        (b - m) - y <= cp.multiply(tilde_bigm, z),
        cp.sum(q) <= C,
        k @ z <= K,
    ]
    if include_u_hat_bound:
        # (15i): u_hat_i <= (min{x^U_i, m_i} - a_i)^2 = hat_bigm_i^2. In the
        # rescaled variables u_hat = hat_scale^2 * u_hat_scaled, so the
        # bound becomes u_hat_scaled <= (hat_bigm / hat_scale)^2.
        #
        # Redundant at integer z (there (15g) gives hat_x <= hat_bigm, and
        # (15d) forces u_hat = 0 whenever hat_x = 0), but NOT redundant in
        # the continuous relaxation of the perspective model: there
        # hat_x^2 <= u_hat * z with fractional z admits u_hat >= hat_x^2/z,
        # which this caps. That is exactly why the revision states it for
        # the perspective formulation only.
        constraints.append(u_hat_scaled <= (hat_bigm / hat_scale) ** 2)
    multiplier = z if perspective else 1.0
    _add_rotated_cone(constraints, hat_x_scaled, u_hat_scaled, multiplier)
    _add_rotated_cone(constraints, u_hat_scaled, w_hat_scaled, hat_x_scaled)
    _add_rotated_cone(constraints, y_scaled, u_y_scaled, 1.0)
    _add_rotated_cone(constraints, u_y_scaled, w_y_scaled, y_scaled)

    objective = _triangular_objective(a, m, b, p, c, s, bar_x, hat_x, y, w_hat, w_y)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    _require(problem.is_dcp(), "Triangular fixed-cost CVXPY model is not DCP")
    _require(problem.is_mixed_integer(), "Triangular fixed-cost CVXPY model is not mixed-integer")

    if not perspective:
        model_id = BIGM_MODEL_ID
    elif include_u_hat_bound:
        model_id = PERSPECTIVE_BOUNDED_MODEL_ID
    else:
        model_id = PERSPECTIVE_MODEL_ID
    return TriangularFormulation(
        problem=problem,
        variables={
            "bar_x": bar_x, "hat_x": hat_x, "y": y, "z": z,
            "u_hat": u_hat, "w_hat": w_hat, "u_y": u_y, "w_y": w_y,
            "q": q,
        },
        model_id=model_id,
    )


def build_current_triangular_bigm_fixed_cost(
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
) -> TriangularFormulation:
    """Manuscript formulation (9): the naive big-M Triangular fixed-cost MISOCP."""
    return _build_triangular_fixed_cost_model(
        a, m, b, price_p, cost_c, salvage_s, x_u, k, C, K, perspective=False
    )


def build_current_triangular_perspective_fixed_cost(
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
) -> TriangularFormulation:
    """Manuscript's perspective reformulation of (9): (9c) replaced by (5)."""
    return _build_triangular_fixed_cost_model(
        a, m, b, price_p, cost_c, salvage_s, x_u, k, C, K, perspective=True
    )


def build_current_triangular_perspective_bounded_fixed_cost(
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
) -> TriangularFormulation:
    """The perspective reformulation PLUS the revision's valid inequality
    (15i), u_hat_i <= (min{x^U_i, m_i} - a_i)^2.
    """
    return _build_triangular_fixed_cost_model(
        a, m, b, price_p, cost_c, salvage_s, x_u, k, C, K,
        perspective=True, include_u_hat_bound=True,
    )


def solver_options(
    solver: str,
    *,
    time_limit_seconds: float,
    mip_gap: float,
    threads: int,
) -> dict[str, Any]:
    """CVXPY ``solver_opts`` for CPLEX, MOSEK, or GUROBI, matching the
    manuscript's Table 2 solver configuration (a wall time limit and a
    relative MIP gap).
    """
    solver = solver.upper()
    if solver == "CPLEX":
        return {
            "cplex_params": {
                "timelimit": time_limit_seconds,
                "mip.tolerances.mipgap": mip_gap,
                "threads": threads,
            },
            "reoptimize": True,
        }
    if solver == "MOSEK":
        return {
            "mosek_params": {
                "MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit_seconds,
                "MSK_DPAR_MIO_MAX_TIME": time_limit_seconds,
                "MSK_DPAR_MIO_TOL_REL_GAP": mip_gap,
                "MSK_IPAR_NUM_THREADS": threads,
            },
            "accept_unknown": False,
        }
    if solver == "GUROBI":
        # CVXPY's GUROBI interface passes solver_opts straight through as
        # gurobipy Model.setParam(key, value) calls (aside from a short list
        # of interface-level keys it recognizes itself, "reoptimize" among
        # them), so these are Gurobi's own parameter names (case-sensitive),
        # not a nested params dict. "reoptimize" mirrors the CPLEX branch
        # above: on Gurobi status 4 (INFEASIBLE_OR_UNBOUNDED, which its
        # default dual-reduction presolve can return instead of a definite
        # answer) CVXPY reruns with DualReductions disabled to get a
        # conclusive status rather than reporting NO_SOLUTION.
        #
        return {
            "TimeLimit": time_limit_seconds,
            "MIPGap": mip_gap,
            "Threads": threads,
            "reoptimize": True,
        }
    raise TriangularModelError(f"unsupported solver: {solver!r}; expected one of {SUPPORTED_SOLVERS}")


def solve_current_triangular_model(
    formulation: TriangularFormulation,
    *,
    solver: str | None = None,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 1.0e-3,
    threads: int = 1,
    verbose: bool = False,
) -> Any:
    """Solve ``formulation.problem`` in place using CPLEX, MOSEK, or GUROBI.

    The solver backend defaults to the module-level ``DEFAULT_SOLVER`` global;
    pass ``solver="CPLEX"``, ``solver="MOSEK"``, or ``solver="GUROBI"``
    explicitly to override it for a single call without changing the global
    default.
    """
    chosen = (solver or DEFAULT_SOLVER).upper()
    _require(chosen in SUPPORTED_SOLVERS, f"solver must be one of {SUPPORTED_SOLVERS}; observed {chosen!r}")
    options = solver_options(
        chosen, time_limit_seconds=time_limit_seconds, mip_gap=mip_gap, threads=threads
    )
    formulation.problem.solve(solver=chosen, verbose=verbose, **options)
    return formulation.problem


def _self_test() -> None:
    """A tiny in-memory smoke test: build both fixed-cost models and confirm
    they are recognized by CVXPY as convex conic mixed-integer programs.
    """
    rng = np.random.default_rng(0)
    n = 6
    a = rng.uniform(0.0, 50.0, size=n)
    spans = rng.uniform(80.0, 150.0, size=n)
    b = a + spans
    m = a + rng.uniform(0.1, 0.9, size=n) * spans
    c = rng.uniform(1.0, 5.0, size=n)
    p = c * rng.uniform(1.2, 2.0, size=n)
    s = rng.uniform(-0.5, 0.5, size=n) * c
    x_u = 0.5 * (a + b)
    k = rng.uniform(1.0, 5.0, size=n)
    C = 0.3 * float(np.sum(x_u))
    K = 0.5 * float(np.sum(k))

    continuous = build_current_triangular_conic_continuous(a, m, b, p, c, s, C)
    _require(continuous.problem.is_dcp(), "self-test: continuous model failed DCP check")

    bigm = build_current_triangular_bigm_fixed_cost(a, m, b, p, c, s, x_u, k, C, K)
    perspective = build_current_triangular_perspective_fixed_cost(a, m, b, p, c, s, x_u, k, C, K)
    _require(bigm.model_id == BIGM_MODEL_ID, "self-test: bigm model_id mismatch")
    _require(perspective.model_id == PERSPECTIVE_MODEL_ID, "self-test: perspective model_id mismatch")
    _require(bigm.problem.is_dcp() and bigm.problem.is_mixed_integer(), "self-test: bigm not a valid MISOCP")
    _require(
        perspective.problem.is_dcp() and perspective.problem.is_mixed_integer(),
        "self-test: perspective not a valid MISOCP",
    )

    bounded = build_current_triangular_perspective_bounded_fixed_cost(
        a, m, b, p, c, s, x_u, k, C, K
    )
    _require(
        bounded.model_id == PERSPECTIVE_BOUNDED_MODEL_ID,
        "self-test: perspective+(15i) model_id mismatch",
    )
    _require(
        bounded.problem.is_dcp() and bounded.problem.is_mixed_integer(),
        "self-test: perspective+(15i) not a valid MISOCP",
    )
    _require(
        len(bounded.problem.constraints) == len(perspective.problem.constraints) + 1,
        "self-test: perspective+(15i) did not add exactly one constraint",
    )
    try:
        _build_triangular_fixed_cost_model(
            a, m, b, p, c, s, x_u, k, C, K, perspective=False, include_u_hat_bound=True
        )
    except TriangularModelError:
        pass
    else:  # pragma: no cover - guard must reject the big-M combination
        raise AssertionError("self-test: (15i) was accepted without the perspective lift")

    print("TRIANGULAR_CURRENT_SELF_TEST_PASS")


if __name__ == "__main__":
    _self_test()
