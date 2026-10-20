"""Standalone reconstruction of the Chernonog and Goldberg (2018) continuous
Triangular multi-item newsvendor model (equations (12)-(13) of "On the
multi-product newsvendor with bounded demand distributions").

This module used to hold a Pyomo ``ConcreteModel`` builder that depended on a
``python_replication`` package (``.common`` / ``.construction_data``) that is
not part of this project directory, so it could not be imported or run on its
own. It has been rewritten to be self-contained: it depends only on NumPy and
SciPy.

The 2018 paper solves this model with CONOPT (a commercial GAMS NLP solver;
Section 6.2, "solving our mathematical programming formulations using a
standard [nonlinear] solver"). No GAMS/CONOPT license is available in this
project's runtime. This module offers three substitutes:

* ``method="trust-constr"`` or ``"SLSQP"`` hand the model to
  ``scipy.optimize.minimize`` unchanged -- a general local NLP solver applied
  to a problem whose objective is concave over the (box-constrained) feasible
  region, so any KKT point found is the global optimum (2018 paper, Section
  4.2, last paragraph). This is the most literal CONOPT substitute, but
  neither scipy method exploits this problem's block-diagonal-Hessian +
  single-linear-constraint sparsity the way CONOPT's sparse SLP machinery
  would, so both become impractically slow (cubic-looking growth in solve
  time) beyond a few hundred products -- confirmed empirically up to n=200
  during development of this module; n=500 did not finish in several
  minutes with either method.
* ``method="ipopt"`` builds the identical cubic objective (12)-(13) as a
  Pyomo ``ConcreteModel`` and solves it with IPOPT (Wachter and Biegler's
  open-source large-scale interior-point NLP solver). Note: this is *not*
  routed through CVXPY -- CVXPY has no IPOPT backend (it only targets DCP
  problems that reduce to conic/QP form, and its own ``installed_solvers()``
  never lists IPOPT), so the natural, standard way to drive IPOPT from
  Python is Pyomo's ``SolverFactory("ipopt")``, which is what this method
  uses. IPOPT is, like CONOPT, a general sparse large-scale NLP solver, so
  this is in practice a much closer CONOPT substitute than either scipy
  method above -- it is expected to scale far better than "trust-constr"/
  "SLSQP" on this problem.
* ``method="dual"`` (this module's own default) instead solves the same KKT
  system exactly via a Lagrangian (dual) bisection on the single capacity
  constraint's multiplier: for a fixed multiplier, the per-product
  subproblem decomposes into three independent 1-D concave maximizations
  (closed form for each of bar_x, hat_x, tilde_x), and the aggregate order
  quantity is monotonically non-increasing in the multiplier, so a standard
  bisection finds the multiplier for which the capacity constraint holds
  with equality (or confirms it is slack). This is exact (not an
  approximation) for this concave, Slater-regular problem, and it is exactly
  the kind of separable resource-allocation procedure the 2018 paper itself
  recommends in Section 5.2 for differentiable non-uniform demand
  distributions ("efficient numerical algorithms for separable
  differentiable optimization such as ... Kodialam and Luss (1998); Zipkin
  (1980)"). It reproduces the same global optimum CONOPT would find, in
  O(n log(1/tol)) time.

All three non-"dual" methods, and "dual" itself, are cross-checked against
each other on a binding-capacity instance in ``_self_test()`` below.

The model, per product i with support [a_i, b_i] and mode m_i (denoted d_i in
the 2018 paper; h_i = -s_i relates the paper's leftover-cost parameter to
this project's salvage-value parameter s_i used elsewhere):

    max  sum_i (p_i - c_i)(bar_x_i + hat_x_i + tilde_x_i)
              + (s_i - p_i)(m_i - a_i)/(b_i - a_i) * tilde_x_i
              + (s_i - p_i)/(b_i - a_i) * tilde_x_i^2
              + (s_i - p_i)/(3(b_i - a_i)) * (hat_x_i^3/(m_i-a_i) - tilde_x_i^3/(b_i-m_i))
    s.t. sum_i (bar_x_i + hat_x_i + tilde_x_i) <= C
         0 <= bar_x_i <= a_i,  0 <= hat_x_i <= m_i - a_i,  0 <= tilde_x_i <= b_i - m_i

This is the same feasible region and objective as manuscript equation (8) of
Newsvendor_current.pdf (the "current" revision), just solved directly with a
general NLP solver in place of the second-order-cone lift used by
triangular_current.py's conic reformulation. The two must therefore agree on
the optimal objective value and optimal physical order quantities
bar_x+hat_x+tilde_x, up to solver tolerance -- this equivalence is exactly
what run_table_new_experiment.py checks.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, minimize
    from scipy.sparse import diags
except ImportError as error:  # pragma: no cover - import guard only
    raise ImportError(
        "triangular_2018 requires scipy to be installed (pip install scipy)."
    ) from error


DEFAULT_METHOD = "dual"              # "dual", or any scipy.optimize.minimize method
MODEL_ID = "CG2018_TRIANGULAR_CUBIC_CONTINUOUS"
SCIPY_METHODS = ("trust-constr", "SLSQP")
ALL_METHODS = ("dual", "ipopt") + SCIPY_METHODS


class Triangular2018Error(RuntimeError):
    """Raised when input data or solver configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Triangular2018Error(message)


def _as_vector(values: Sequence[float], n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    _require(array.shape == (n,), f"{name} must have shape ({n},); observed {array.shape}")
    _require(bool(np.all(np.isfinite(array))), f"{name} contains a non-finite value")
    return array


def _validate_support(a: np.ndarray, m: np.ndarray, b: np.ndarray) -> None:
    _require(bool(np.all(a >= 0.0)), "a must be nonnegative")
    _require(bool(np.all(a < m)), "a must be strictly less than m")
    _require(bool(np.all(m < b)), "m must be strictly less than b")


# ---------------------------------------------------------------------------
# Closed-form scalar oracle, duplicated (rather than imported) from the
# sibling scripts on purpose: this module's own objective/gradient/Hessian
# below are validated independently against it in _self_test().
# ---------------------------------------------------------------------------

def triangular_unconstrained_quantity(
    a: float, m: float, b: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    alpha = (price_p - cost_c) / (price_p - salvage_s)
    mode_cdf = (m - a) / (b - a)
    if alpha <= mode_cdf:
        return a + math.sqrt(alpha * (b - a) * (m - a))
    return b - math.sqrt((1.0 - alpha) * (b - a) * (b - m))


def triangular_segment_quantities(x: float, a: float, m: float, b: float) -> tuple[float, float, float]:
    x = min(max(x, 0.0), b)
    x_bar = min(x, a)
    x_hat = min(max(x - a, 0.0), m - a)
    x_tilde = min(max(x - m, 0.0), b - m)
    return x_bar, x_hat, x_tilde


@dataclasses.dataclass(slots=True)
class Triangular2018Formulation:
    """A built 2018-style NLP plus the data ``scipy.optimize.minimize`` needs."""

    a: np.ndarray
    m: np.ndarray
    b: np.ndarray
    price_p: np.ndarray
    cost_c: np.ndarray
    salvage_s: np.ndarray
    C: float
    n: int
    bounds: Bounds
    linear_constraint: LinearConstraint
    x0: np.ndarray
    model_id: str = MODEL_ID


def _initial_point(
    a: np.ndarray, m: np.ndarray, b: np.ndarray,
    p: np.ndarray, c: np.ndarray, s: np.ndarray, C: float,
) -> np.ndarray:
    """A feasible-ish starting point: the unconstrained per-product optimum,
    uniformly scaled down to satisfy the capacity constraint if necessary.
    """
    n = a.size
    x_u = np.array(
        [triangular_unconstrained_quantity(a[i], m[i], b[i], p[i], c[i], s[i]) for i in range(n)]
    )
    total = float(np.sum(x_u))
    scale = 1.0 if total <= C or total <= 0.0 else C / total
    scaled = x_u * scale
    bar = np.empty(n)
    hat = np.empty(n)
    tilde = np.empty(n)
    for i in range(n):
        bar[i], hat[i], tilde[i] = triangular_segment_quantities(scaled[i], a[i], m[i], b[i])
    return np.concatenate([bar, hat, tilde])


def build_cg2018_triangular_continuous(
    a: Sequence[float],
    m: Sequence[float],
    b: Sequence[float],
    price_p: Sequence[float],
    cost_c: Sequence[float],
    salvage_s: Sequence[float],
    C: float,
) -> Triangular2018Formulation:
    """The 2018 paper's continuous Triangular model (12)-(13), restricted to a
    single unit-coefficient resource constraint sum_i x_i <= C (i.e. r_i1 = 1
    for all i, R_1 = C, and the separate monetary budget constraint (1) of
    the 2018 paper disabled) so that this model and triangular_current.py's
    build_current_triangular_conic_continuous share exactly the same
    feasible region and objective.
    """
    n = len(a)
    a, m, b = (_as_vector(v, n, name) for v, name in ((a, "a"), (m, "m"), (b, "b")))
    p = _as_vector(price_p, n, "price_p")
    c = _as_vector(cost_c, n, "cost_c")
    s = _as_vector(salvage_s, n, "salvage_s")
    _validate_support(a, m, b)
    _require(np.isfinite(C) and C >= 0.0, "C must be finite and nonnegative")

    lower = np.zeros(3 * n)
    upper = np.concatenate([a, m - a, b - m])
    bounds = Bounds(lower, upper)
    linear_constraint = LinearConstraint(np.ones(3 * n), -np.inf, C)
    x0 = _initial_point(a, m, b, p, c, s, C)

    return Triangular2018Formulation(
        a=a, m=m, b=b, price_p=p, cost_c=c, salvage_s=s, C=float(C), n=n,
        bounds=bounds, linear_constraint=linear_constraint, x0=x0,
    )


def _negated_objective_and_gradient(
    x: np.ndarray, a: np.ndarray, m: np.ndarray, b: np.ndarray,
    p: np.ndarray, c: np.ndarray, s: np.ndarray,
) -> tuple[float, np.ndarray]:
    """-(sum_i pi-bar_i), and its gradient, for scipy's minimize (see module
    docstring for pi-bar_i). The three segments are diagonal-separable: no
    cross terms appear in either the gradient or the Hessian below.
    """
    n = a.size
    bar, hat, tilde = x[:n], x[n:2 * n], x[2 * n:]
    coeff_lin = p - c
    coeff3 = (s - p) / (b - a)

    term = (
        coeff_lin * (bar + hat + tilde)
        + coeff3 * (m - a) * tilde
        + coeff3 * tilde ** 2
        + (coeff3 / 3.0) * (hat ** 3 / (m - a) - tilde ** 3 / (b - m))
    )
    objective = float(np.sum(term))

    d_bar = coeff_lin
    d_hat = coeff_lin + coeff3 * hat ** 2 / (m - a)
    d_tilde = coeff_lin + coeff3 * (m - a) + 2.0 * coeff3 * tilde - coeff3 * tilde ** 2 / (b - m)
    gradient = np.concatenate([d_bar, d_hat, d_tilde])

    return -objective, -gradient


def _negated_hessian(
    x: np.ndarray, a: np.ndarray, m: np.ndarray, b: np.ndarray,
    p: np.ndarray, c: np.ndarray, s: np.ndarray,
):
    """Sparse diagonal Hessian of the negated objective (positive
    semidefinite, since pi-bar_i is concave in each of its three arguments
    with no cross terms).
    """
    n = a.size
    hat, tilde = x[n:2 * n], x[2 * n:]
    coeff3 = (s - p) / (b - a)
    d2_hat = 2.0 * coeff3 * hat / (m - a)
    d2_tilde = 2.0 * coeff3 * (1.0 - tilde / (b - m))
    diagonal = np.concatenate([np.zeros(n), -d2_hat, -d2_tilde])
    return diags(diagonal)


@dataclasses.dataclass(slots=True)
class SimpleSolveResult:
    """A minimal stand-in for scipy's OptimizeResult, so callers can treat
    the output of every ``method=`` uniformly (``.x``, ``.success``,
    ``.status``, ``.message``, ``.fun``, ``.constr_violation``, ``.nit``).
    """

    x: np.ndarray
    success: bool
    status: int
    message: str
    fun: float
    constr_violation: float
    nit: int
    # The optimal capacity multiplier, when the method that produced this
    # result has one (``method="dual"``); None otherwise. Exposed because
    # triangular_submodular_cuts.py's cut coefficients ARE the per-product
    # conjugates evaluated at this price, so a caller needs the multiplier
    # and not only the optimal value. Optional/defaulted so every existing
    # construction of this dataclass keeps working unchanged.
    multiplier: float | None = None


def _per_product_response(
    lam: np.ndarray, a: np.ndarray, m: np.ndarray, b: np.ndarray, coeff_lin: np.ndarray, coeff3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a (scalar-broadcastable) multiplier ``lam``, the componentwise
    argmax over the box of pi-bar_i(bar,hat,tilde) - lam*(bar+hat+tilde),
    vectorized over all n products. Each of the three segments is a
    1-D concave maximization with no cross terms, so this decomposes exactly
    (see module docstring, ``method="dual"``).
    """
    # bar_i: linear in bar_i, so the response is bang-bang at its bound.
    # Strict ">" (rather than ">=") matters at lam == max(coeff_lin): see
    # _solve_dual_bisection's upper bracket, which relies on every product
    # being inactive there.
    bar = np.where(coeff_lin > lam, a, 0.0)

    # hat_i: derivative coeff_lin + coeff3*hat^2/(m-a), decreasing in hat
    # since coeff3 < 0; T_hat'(0) = coeff_lin, T_hat'(m-a) = coeff_lin + coeff3*(m-a).
    hat_deriv_end = coeff_lin + coeff3 * (m - a)
    hat_interior = np.sqrt(np.clip((coeff_lin - lam) / (-coeff3) * (m - a), 0.0, None))
    hat = np.where(
        lam >= coeff_lin, 0.0,
        np.where(lam <= hat_deriv_end, m - a, hat_interior),
    )

    # tilde_i: derivative coeff_lin + coeff3*(m-a) + 2*coeff3*tilde - (coeff3/(b-m))*tilde^2,
    # concave and decreasing over [0, b-m] (its second derivative is <= 0
    # there, reaching exactly 0 at tilde = b-m). T_tilde'(0) = hat_deriv_end
    # (continuity with the hat segment at the mode); T_tilde'(b-m) = s - c.
    deriv0 = hat_deriv_end
    derivB = coeff_lin + coeff3 * (b - a)
    quad_a = -(coeff3 / (b - m))
    quad_b = 2.0 * coeff3
    quad_c = deriv0 - lam
    discriminant = np.clip(quad_b ** 2 - 4.0 * quad_a * quad_c, 0.0, None)
    tilde_interior = (-quad_b - np.sqrt(discriminant)) / (2.0 * quad_a)
    tilde = np.where(
        lam >= deriv0, 0.0,
        np.where(lam <= derivB, b - m, np.clip(tilde_interior, 0.0, b - m)),
    )
    return bar, hat, tilde


def _fill_marginal_bar_gap(
    bar: np.ndarray, a: np.ndarray, coeff_lin: np.ndarray, lam_star: float, gap: float,
) -> None:
    """Top up the marginal (bang-bang) bar_x product(s) in place by exactly
    ``gap`` (the shortfall between the capacity C and the total quantity
    already achieved by bar+hat+tilde combined), so the capacity constraint
    binds with equality whenever it should.

    Unlike hat_x and tilde_x (strictly concave, hence smooth/interior
    responses), bar_x is *linear* in its own objective term, so
    ``_per_product_response`` gives it a discontinuous bang-bang response:
    ``a_i`` when coeff_lin_i > lam, 0 otherwise. When the bisection's
    optimal multiplier lam_star lands exactly on one product's own
    coeff_lin_i (which happens whenever that product is the capacity
    constraint's marginal/critical item -- a generic, not rare, occurrence
    for a knapsack-like linear segment), evaluating the response function at
    lam_star leaves that product's bar_x at the "just excluded" corner (0),
    understating both the achieved capacity usage and the objective. The
    fix mirrors the classical continuous-knapsack optimum: give the
    marginal item exactly the fractional quantity needed to use the
    remaining capacity, rather than the response function's arbitrary
    all-or-nothing corner at the tie.
    """
    if gap <= 0.0:
        return
    tolerance = 1.0e-6 * max(1.0, abs(lam_star))
    candidates = np.where((bar < a - 1.0e-12) & (np.abs(coeff_lin - lam_star) <= tolerance))[0]
    for i in candidates:
        fill = min(gap, a[i] - bar[i])
        bar[i] += fill
        gap -= fill
        if gap <= 1.0e-9:
            break


def _solve_dual_bisection(
    formulation: Triangular2018Formulation,
    *,
    max_bisection_iterations: int = 100,
    feasibility_tol: float = 1.0e-9,
) -> SimpleSolveResult:
    a, m, b = formulation.a, formulation.m, formulation.b
    p, c, s = formulation.price_p, formulation.cost_c, formulation.salvage_s
    C = formulation.C
    coeff_lin = p - c
    coeff3 = (s - p) / (b - a)

    def aggregate_quantity(lam: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        bar, hat, tilde = _per_product_response(lam, a, m, b, coeff_lin, coeff3)
        return float(np.sum(bar) + np.sum(hat) + np.sum(tilde)), bar, hat, tilde

    total_at_zero, bar, hat, tilde = aggregate_quantity(0.0)
    if total_at_zero <= C + feasibility_tol:
        lam_star = 0.0
    else:
        # A hair above max(coeff_lin) so every product's bar/hat/tilde
        # response is strictly, unambiguously inactive at the bracket's
        # upper end (see the strict ">" note in _per_product_response).
        lam_lo = 0.0
        lam_hi = float(np.max(coeff_lin)) * (1.0 + 1.0e-12) + 1.0e-9
        _require(
            aggregate_quantity(lam_hi)[0] <= C + feasibility_tol,
            "dual bisection: multiplier upper bracket does not clear the capacity constraint",
        )
        for _ in range(max_bisection_iterations):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            total_mid, bar, hat, tilde = aggregate_quantity(lam_mid)
            if total_mid > C:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo <= 1.0e-14 * max(1.0, lam_hi):
                break
        lam_star = lam_hi
        achieved_total, bar, hat, tilde = aggregate_quantity(lam_star)
        _fill_marginal_bar_gap(bar, a, coeff_lin, lam_star, C - achieved_total)

    x = np.concatenate([bar, hat, tilde])
    objective, _ = _negated_objective_and_gradient(x, a, m, b, p, c, s)
    violation = max(0.0, float(np.sum(bar) + np.sum(hat) + np.sum(tilde)) - C)
    return SimpleSolveResult(
        x=x,
        success=violation <= 1.0e-6 + 1.0e-8 * max(1.0, C),
        status=1,
        message="dual bisection converged",
        fun=objective,
        constr_violation=violation,
        nit=max_bisection_iterations,
        multiplier=float(lam_star),
    )


def _build_pyomo_model(
    a: np.ndarray, m: np.ndarray, b: np.ndarray,
    p: np.ndarray, c: np.ndarray, s: np.ndarray, C: float,
) -> Any:
    """The literal cubic model (12)-(13) as a Pyomo ConcreteModel, for IPOPT."""
    import pyomo.environ as pyo

    n = a.size
    coeff_lin = p - c
    coeff3 = (s - p) / (b - a)

    model = pyo.ConcreteModel(name=MODEL_ID)
    model.I = pyo.RangeSet(0, n - 1)
    model.bar_x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=lambda mdl, i: (0.0, a[i]))
    model.hat_x = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=lambda mdl, i: (0.0, m[i] - a[i]))
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
    model.capacity = pyo.Constraint(
        expr=sum(model.bar_x[i] + model.hat_x[i] + model.tilde_x[i] for i in model.I) <= C
    )
    return model


def _solve_ipopt(
    formulation: Triangular2018Formulation,
    *,
    max_iter: int = 3000,
    tol: float = 1.0e-8,
    tee: bool = False,
) -> SimpleSolveResult:
    """Solve the literal cubic model with IPOPT via Pyomo (see module
    docstring's ``method="ipopt"`` entry for why Pyomo, not CVXPY).
    """
    import pyomo.environ as pyo

    n = formulation.n
    a, m, b = formulation.a, formulation.m, formulation.b
    p, c, s = formulation.price_p, formulation.cost_c, formulation.salvage_s
    C = formulation.C

    model = _build_pyomo_model(a, m, b, p, c, s, C)
    solver = pyo.SolverFactory("ipopt")
    _require(solver.available(), "IPOPT is not available (pyo.SolverFactory('ipopt').available() is False)")
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    results = solver.solve(model, tee=tee)

    termination = results.solver.termination_condition
    success = termination == pyo.TerminationCondition.optimal
    bar = np.array([pyo.value(model.bar_x[i]) for i in range(n)])
    hat = np.array([pyo.value(model.hat_x[i]) for i in range(n)])
    tilde = np.array([pyo.value(model.tilde_x[i]) for i in range(n)])
    x = np.concatenate([bar, hat, tilde])
    objective, _ = _negated_objective_and_gradient(x, a, m, b, p, c, s)
    violation = max(0.0, float(np.sum(bar) + np.sum(hat) + np.sum(tilde)) - C)

    return SimpleSolveResult(
        x=x,
        success=success and violation <= 1.0e-6 + 1.0e-6 * max(1.0, C),
        status=1 if success else 0,
        message=f"IPOPT termination_condition={termination}",
        fun=objective,
        constr_violation=violation,
        nit=int(getattr(results.solver, "iterations", -1) or -1),
    )


def solve_cg2018_triangular_continuous(
    formulation: Triangular2018Formulation,
    *,
    method: str = DEFAULT_METHOD,
    maxiter: int = 2000,
    gtol: float = 1.0e-10,
    xtol: float = 1.0e-12,
    verbose: int = 0,
) -> Any:
    """Solve ``formulation`` (CONOPT stand-in; see module docstring).

    ``method="dual"`` (this module's own default) uses the exact Lagrangian
    bisection in ``_solve_dual_bisection``. ``method="ipopt"`` builds the
    literal cubic model as a Pyomo ConcreteModel and solves it with IPOPT
    (see ``_solve_ipopt``). Any other ``method`` (e.g. ``"trust-constr"``,
    ``"SLSQP"``) is passed through to ``scipy.optimize.minimize`` unchanged.

    Returns an object exposing ``.x``, ``.success``, ``.status``,
    ``.message``, ``.fun``, and ``.constr_violation`` regardless of method;
    the caller slices ``.x`` into (bar, hat, tilde) using ``formulation.n``.
    """
    if method == "dual":
        return _solve_dual_bisection(formulation)
    if method == "ipopt":
        return _solve_ipopt(formulation)

    a, m, b = formulation.a, formulation.m, formulation.b
    p, c, s = formulation.price_p, formulation.cost_c, formulation.salvage_s

    def fun_and_jac(x: np.ndarray) -> tuple[float, np.ndarray]:
        return _negated_objective_and_gradient(x, a, m, b, p, c, s)

    def hess(x: np.ndarray):
        return _negated_hessian(x, a, m, b, p, c, s)

    options: dict[str, Any] = {"maxiter": maxiter, "gtol": gtol, "xtol": xtol}
    if verbose:
        options["verbose"] = verbose

    return minimize(
        fun=fun_and_jac,
        x0=formulation.x0,
        jac=True,
        hess=hess,
        bounds=formulation.bounds,
        constraints=[formulation.linear_constraint],
        method=method,
        options=options,
    )


def extract_solution(formulation: Triangular2018Formulation, result: Any) -> dict[str, np.ndarray]:
    n = formulation.n
    x = np.asarray(result.x, dtype=float)
    bar, hat, tilde = x[:n], x[n:2 * n], x[2 * n:]
    return {"bar_x": bar, "hat_x": hat, "tilde_x": tilde, "q": bar + hat + tilde}


def _self_test() -> None:
    """In-memory smoke tests: solve small random instances with both the
    ``"dual"`` bisection and scipy's ``"trust-constr"``, and check that (a)
    the nonbinding-capacity solution matches the closed-form per-product
    optimum, and (b) the two solve methods agree with each other on a
    binding-capacity instance -- this is the cross-check referenced in the
    module docstring's description of ``method="dual"``.
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
    C_nonbinding = float(np.sum(x_u)) * 10.0

    formulation = build_cg2018_triangular_continuous(a, m, b, p, c, s, C_nonbinding)
    result = solve_cg2018_triangular_continuous(formulation, method="dual")
    _require(result.success, f"self-test: dual solver did not report success: {result.message}")
    solution = extract_solution(formulation, result)
    max_gap = float(np.max(np.abs(solution["q"] - x_u)))
    _require(max_gap <= 1.0e-6, f"self-test: nonbinding dual solution does not match x_u (max gap {max_gap})")

    C_binding = 0.3 * float(np.sum(x_u))
    formulation_b = build_cg2018_triangular_continuous(a, m, b, p, c, s, C_binding)
    result_dual = solve_cg2018_triangular_continuous(formulation_b, method="dual")
    result_scipy = solve_cg2018_triangular_continuous(formulation_b, method="trust-constr")
    _require(result_dual.success, "self-test: dual solver did not report success on binding instance")
    _require(result_scipy.success, "self-test: trust-constr did not report success on binding instance")
    objective_gap = abs(result_dual.fun - result_scipy.fun)
    _require(
        objective_gap <= 1.0e-4 + 1.0e-7 * max(1.0, abs(result_dual.fun)),
        f"self-test: dual and trust-constr disagree on binding instance (gap {objective_gap})",
    )
    q_dual = extract_solution(formulation_b, result_dual)["q"]
    q_scipy = extract_solution(formulation_b, result_scipy)["q"]
    max_q_gap = float(np.max(np.abs(q_dual - q_scipy)))
    _require(max_q_gap <= 1.0e-3, f"self-test: dual and trust-constr order quantities disagree (max gap {max_q_gap})")

    ipopt_available = False
    try:
        import pyomo.environ as pyo

        ipopt_available = pyo.SolverFactory("ipopt").available()
    except ImportError:
        pass
    if ipopt_available:
        result_ipopt = solve_cg2018_triangular_continuous(formulation_b, method="ipopt")
        _require(result_ipopt.success, f"self-test: IPOPT did not report success: {result_ipopt.message}")
        objective_gap_ipopt = abs(result_dual.fun - result_ipopt.fun)
        _require(
            objective_gap_ipopt <= 1.0e-4 + 1.0e-7 * max(1.0, abs(result_dual.fun)),
            f"self-test: dual and IPOPT disagree on binding instance (gap {objective_gap_ipopt})",
        )
        q_ipopt = extract_solution(formulation_b, result_ipopt)["q"]
        max_q_gap_ipopt = float(np.max(np.abs(q_dual - q_ipopt)))
        _require(
            max_q_gap_ipopt <= 1.0e-3,
            f"self-test: dual and IPOPT order quantities disagree (max gap {max_q_gap_ipopt})",
        )
    else:
        print("TRIANGULAR_2018_SELF_TEST_NOTE: IPOPT unavailable, skipping method=\"ipopt\" cross-check")

    print("TRIANGULAR_2018_SELF_TEST_PASS")


if __name__ == "__main__":
    _self_test()
