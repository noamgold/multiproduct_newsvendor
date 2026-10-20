"""Submodular-cut (Nemhauser-Wolsey) solution of the Triangular and
Uniform fixed-cost multiproduct newsvendor, via a GUROBI lazy-constraint
callback.

The master, the cut families and the separation below are entirely
distribution-independent: the demand distribution enters only through the
f-oracle (TriangularCapacityOracle / UniformCapacityOracle), so both
families run the identical separation code.

Background (submodularity_1.pdf, Sections 1-2). Once the set of stocked
product types S = {i : z_i = 1} is fixed, what remains is the
capacity-constrained *continuous* multiproduct newsvendor on S:

    f(S) = max { sum_{i in S} pi_i(x_i) : sum_{i in S} x_i <= C,
                 0 <= x_i <= x^U_i },      f({}) = 0.                    (1)

f is monotone nondecreasing and submodular (Proposition 1), so by
Corollary 1 the whole fixed-cost problem (manuscript (9)/(12)) is exactly

    max { f(S) : sum_{i in S} k_i <= K },

a monotone submodular maximization under a knapsack constraint. That is
solved here as a small pure-binary master

    max  y
    s.t. y <= <cut>,  sum_i k_i z_i <= K,  z in {0,1}^n                  (6)

with the cuts generated lazily, at integer nodes only (Remark 6: at an
integer point the most violated cut of the family is the one generated
from S = {i : z*_i = 1}, obtained by a single evaluation of f -- which is
what makes the cut loop exact). The continuous x variables are projected
out entirely: unlike the MISOCP formulations in triangular_current.py,
this master has n binaries, one continuous variable, and no cones.

Cut families (``cut_family``)
-----------------------------
"dual" (default, and the reason this is cheap). Lemma 1 writes f as a
pointwise minimum of *modular* set functions indexed by the capacity
price nu:

    f(S) = min_{nu >= 0} { nu*C + sum_{i in S} pi*_i(nu) },
    pi*_i(nu) = max_{0 <= x <= x^U_i} { pi_i(x) - nu*x }.                (2,3)

Weak duality therefore makes, for ANY nu >= 0 and EVERY T,
f(T) <= nu*C + sum_{i in T} pi*_i(nu), i.e. the linear inequality

    y <= nu*C + sum_{i in N} pi*_i(nu) z_i                               (*)

is valid for the whole master, and it is *tight* at the S whose optimal
multiplier is nu. So one bisection for f(S) yields both the cut's
constant (nu*C) and all n of its coefficients (pi*_i(nu), each a
closed-form evaluation) -- no per-item re-solves. Two further
consequences worth noting:

  * (*) is valid for any nu >= 0 whatsoever, optimal or not. Bisection
    accuracy therefore affects only how tight the cut is (and hence
    convergence speed), never correctness -- a cut can never remove the
    true optimum.
  * (*) is the Nemhauser-Wolsey inequality (6b) with the exact marginal
    rho_i(S) replaced by the dual upper bound pi*_i(nu_S) >= rho_i(S)
    (evaluate (3) for S+{i} at nu_S and subtract f(S)). Overestimating
    the marginals keeps validity and costs only tightness.

"nemhauser-wolsey". The document's (6b) verbatim, with exact marginals
rho_i(S) = f(S+{i}) - f(S) and rho_i(N\\{i}) = f(N) - f(N\\{i}):

    y <= f(S) + sum_{i in S} rho_i(N\\{i})(z_i - 1) + sum_{i not in S} rho_i(S) z_i

Tighter than (*) away from S, but it costs |N\\S| extra evaluations of f
per cut (plus n once, up front, for the rho_i(N\\{i}) terms) instead of
zero, i.e. O(n) bisections per callback rather than one. Offered for
comparison; not recommended at the larger n.

The f-oracle itself reuses triangular_2018.py's exact dual bisection
machinery (``_per_product_response`` for the componentwise argmax at a
given multiplier, ``_fill_marginal_bar_gap`` for the bar_x flat-segment
tie that occurs when the bisection lands exactly on a product's own
p_i - c_i), restricted to the subset S.

Usage: run_triangular_strongly_correlated.py and
run_uniform_strongly_correlated.py both run this as one of their methods
(``--methods cuts``, or by default alongside the two MISOCPs). This
module's own ``python submodular_cuts.py`` runs a self-test for both
distributions (oracle vs. the closed-form profit oracle and vs.
triangular_2018.py's independent dual solve; whole-MIP vs. brute-force
enumeration on tiny instances).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Sequence

import numpy as np

import triangular_2018 as t2018

MODEL_ID = "SUBMODULAR_CUTS_FIXED_COST"
TRIANGULAR_MODEL_ID = "SUBMODULAR_CUTS_TRIANGULAR_FIXED_COST"
UNIFORM_MODEL_ID = "SUBMODULAR_CUTS_UNIFORM_FIXED_COST"
DISTRIBUTION_FREE_MODEL_ID = "SUBMODULAR_CUTS_DISTRIBUTION_FREE_FIXED_COST"
CUT_FAMILY_CHOICES = ("dual", "nemhauser-wolsey")
DEFAULT_CUT_FAMILY = "dual"

# How f(S) is evaluated. "cg2018" calls the 2018 paper's own continuous
# solver (triangular_2018.solve_cg2018_triangular_continuous, method
# "dual") on the subset -- unmodified provenance, one formulation build
# per call. "inline" runs the same dual bisection masked in place, which
# avoids that per-call build; the two agree to ~2e-15 relative over 180
# random subsets spanning n=100..500.
ORACLE_BACKEND_CHOICES = ("cg2018", "inline")
DEFAULT_ORACLE_BACKEND = "cg2018"

# A cut is only added when the incumbent's y exceeds the set's true value
# by more than this (scaled) amount; also the acceptance test that makes
# the loop terminate. Must comfortably exceed the oracle's own duality
# gap, which the bisection drives to ~1e-12 relative.
DEFAULT_CUT_VIOLATION_ABS_TOL = 1.0e-6
DEFAULT_CUT_VIOLATION_REL_TOL = 1.0e-9


class SubmodularCutError(RuntimeError):
    """Raised when input data or solver configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubmodularCutError(message)


@dataclasses.dataclass(slots=True)
class CapacityValue:
    """One evaluation of f(S): its value, the optimal capacity price nu_S
    attaining the minimum in (3), and the full-length primal order
    quantities (zero outside S).
    """

    value: float
    multiplier: float
    quantities: np.ndarray


class CapacityOracle:
    """Everything in (1)-(3) that does NOT depend on the demand
    distribution: the bisection for f(S), the conjugates the cuts are
    built from, and the marginal-fill tie-break.

    Subclasses supply only the two distribution-specific pieces --
    ``response`` (the componentwise argmax of pi_i(x) - nu*x, returned as
    its segment decomposition) and ``per_product_profit`` (pi_i evaluated
    at that decomposition) -- and must set ``a`` (the linear segment's
    upper bound), ``coeff_lin`` (= p - c, the slope on that segment),
    ``n``, ``C``, ``backend``, ``max_bisection_iterations`` and
    ``feasibility_tol``.

    Assumption 1 of the document (pi_i continuous, concave, pi_i(0) = 0)
    is all Proposition 1 needs, so the submodularity -- and hence every
    cut below -- holds identically for the uniform and triangular models;
    only these two methods change.

    All operations are vectorized over the full product set and masked to
    the subset, so evaluating f(S) costs one bisection of O(n) numpy work
    regardless of |S|.
    """

    def response(self, multiplier: float) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def per_product_profit(self, *segments: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _evaluate_via_reference_solver(self, mask: np.ndarray) -> CapacityValue:
        raise NotImplementedError(
            f"{type(self).__name__} has no reference-solver backend; use backend='inline'"
        )

    def validate_unconstrained(self, x_u: Sequence[float]) -> None:
        """Guard that the oracle and the instance agree about the model:
        at nu = 0 the response must reproduce the instance's own
        unconstrained optimum. Catches a mismatched or misordered
        parameter vector before any cut is built from it.

        Overridable because "agree" is family-specific: for the uniform
        and triangular oracles the response is closed-form and matches
        x^U to machine precision, so the positions are compared directly.
        """
        unconstrained = sum(self.response(0.0))
        _require(
            bool(np.allclose(unconstrained, np.asarray(x_u, dtype=float), rtol=1.0e-7, atol=1.0e-7)),
            "oracle's nu=0 response does not reproduce the instance's x_u",
        )

    def conjugates(self, multiplier: float) -> np.ndarray:
        """pi*_i(nu) for every product -- definition (2). Exact (the
        maximizer is closed-form), which is what makes the cut (*) valid
        regardless of how accurately ``multiplier`` was found.
        """
        segments = self.response(multiplier)
        quantity = sum(segments)
        return self.per_product_profit(*segments) - multiplier * quantity

    def evaluate(self, mask: np.ndarray) -> CapacityValue:
        """f(S) for S = {i : mask[i]}, with its optimal capacity price and
        primal quantities, by bisecting the capacity price until the
        selected products' total quantity meets C (Lemma 1). The value
        returned is the PRIMAL objective at the recovered primal point,
        not the dual bound -- there is no duality gap here (Slater holds),
        and the self-test asserts the two agree.
        """
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return CapacityValue(0.0, 0.0, np.zeros(self.n))
        if self.backend != "inline":
            return self._evaluate_via_reference_solver(mask)

        segments = self.response(0.0)
        total_at_zero = float(sum(segments)[mask].sum())
        if total_at_zero <= self.C + self.feasibility_tol:
            multiplier = 0.0
        else:
            low = 0.0
            # A hair above max(coeff_lin) over S so every selected product
            # is strictly inactive at the bracket's upper end (see the
            # strict ">" note in triangular_2018._per_product_response).
            high = float(np.max(self.coeff_lin[mask])) * (1.0 + 1.0e-12) + 1.0e-9
            for _ in range(self.max_bisection_iterations):
                middle = 0.5 * (low + high)
                segments = self.response(middle)
                if float(sum(segments)[mask].sum()) > self.C:
                    low = middle
                else:
                    high = middle
                if high - low <= 1.0e-14 * max(1.0, high):
                    break
            multiplier = high
            segments = self.response(multiplier)
            achieved = float(sum(segments)[mask].sum())
            self._fill_marginal_gap(segments[0], multiplier, self.C - achieved, mask)

        segments = tuple(np.where(mask, segment, 0.0) for segment in segments)
        value = float(self.per_product_profit(*segments).sum())
        return CapacityValue(value, float(multiplier), sum(segments))

    def _fill_marginal_gap(
        self, linear_segment: np.ndarray, multiplier: float, gap: float, mask: np.ndarray
    ) -> None:
        """triangular_2018._fill_marginal_bar_gap, restricted to S and
        applied to whichever segment is linear in its own objective term
        (bar_x in both the triangular and uniform models): top it up on
        the marginal product(s) so the capacity binds with equality.

        Without this, an S whose critical product sits exactly at
        nu = p_i - c_i has its f(S) understated, and the cut loop would
        re-cut the same point forever.
        """
        if gap <= 0.0:
            return
        tolerance = 1.0e-6 * max(1.0, abs(multiplier))
        candidates = np.where(
            mask
            & (linear_segment < self.a - 1.0e-12)
            & (np.abs(self.coeff_lin - multiplier) <= tolerance)
        )[0]
        for index in candidates:
            fill = min(gap, self.a[index] - linear_segment[index])
            linear_segment[index] += fill
            gap -= fill
            if gap <= 1.0e-9:
                break


class TriangularCapacityOracle(CapacityOracle):
    """Triangular demand on [a_i, b_i] with mode m_i -- manuscript (7)/(8).
    Segments are (bar, hat, tilde); the per-multiplier response and the
    optional reference-solver backend both come from triangular_2018.py.
    """

    def __init__(
        self,
        a: Sequence[float],
        m: Sequence[float],
        b: Sequence[float],
        price_p: Sequence[float],
        cost_c: Sequence[float],
        salvage_s: Sequence[float],
        C: float,
        *,
        backend: str = DEFAULT_ORACLE_BACKEND,
        max_bisection_iterations: int = 100,
        feasibility_tol: float = 1.0e-9,
    ) -> None:
        _require(
            backend in ORACLE_BACKEND_CHOICES,
            f"backend must be one of {ORACLE_BACKEND_CHOICES}",
        )
        self.backend = backend
        self.a = np.asarray(a, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.price_p = np.asarray(price_p, dtype=float)
        self.cost_c = np.asarray(cost_c, dtype=float)
        self.salvage_s = np.asarray(salvage_s, dtype=float)
        self.n = self.a.size
        self.C = float(C)
        self.max_bisection_iterations = max_bisection_iterations
        self.feasibility_tol = feasibility_tol

        _require(bool(np.all(self.a < self.m)), "a must be strictly less than m")
        _require(bool(np.all(self.m < self.b)), "m must be strictly less than b")
        _require(self.C >= 0.0 and np.isfinite(self.C), "C must be finite and nonnegative")

        # Same coefficient names/definitions as triangular_2018.py and
        # triangular_current.py, so the three modules' profit expressions
        # are literally the same algebra.
        self.coeff_lin = self.price_p - self.cost_c
        self.coeff3 = (self.salvage_s - self.price_p) / (self.b - self.a)

    def response(self, multiplier: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Componentwise argmax of pi_i(x) - multiplier*x over the box, as
        the (bar, hat, tilde) segment decomposition, for every product.
        """
        return t2018._per_product_response(
            multiplier, self.a, self.m, self.b, self.coeff_lin, self.coeff3
        )

    def per_product_profit(
        self, bar: np.ndarray, hat: np.ndarray, tilde: np.ndarray
    ) -> np.ndarray:
        """pi_i evaluated at the segment decomposition -- manuscript (8),
        the same expression build_triangular_2018_fixed_cost() gives Pyomo
        and _triangular_objective() gives CVXPY.
        """
        return (
            self.coeff_lin * (bar + hat + tilde)
            + self.coeff3 * (self.m - self.a) * tilde
            + self.coeff3 * tilde**2
            + (self.coeff3 / 3.0)
            * (hat**3 / (self.m - self.a) - tilde**3 / (self.b - self.m))
        )

    def _evaluate_via_reference_solver(self, mask: np.ndarray) -> CapacityValue:
        """f(S) by calling the 2018 paper's own continuous solver on the
        subset, then scattering its (bar, hat, tilde) solution back to
        full length. Costs one formulation build + one solve per call.
        """
        index = np.where(mask)[0]
        formulation = t2018.build_cg2018_triangular_continuous(
            self.a[index], self.m[index], self.b[index],
            self.price_p[index], self.cost_c[index], self.salvage_s[index],
            self.C,
        )
        result = t2018.solve_cg2018_triangular_continuous(formulation, method="dual")
        _require(
            result.multiplier is not None,
            "triangular_2018's dual solve did not report a multiplier; the cut "
            "coefficients are the conjugates at that price, so it is required",
        )
        size = index.size
        subset_bar, subset_hat, subset_tilde = (
            result.x[:size], result.x[size : 2 * size], result.x[2 * size :]
        )
        bar = np.zeros(self.n)
        hat = np.zeros(self.n)
        tilde = np.zeros(self.n)
        bar[index], hat[index], tilde[index] = subset_bar, subset_hat, subset_tilde
        return CapacityValue(
            float(-result.fun), float(result.multiplier), bar + hat + tilde
        )


class UniformCapacityOracle(CapacityOracle):
    """Uniform demand on [a_i, b_i] -- manuscript (3):

        pi_i(x) = (p-c)x                                     for x <= a_i
                = (p-c)x + (s-p)(x-a_i)^2 / (2(b_i-a_i))     for a_i < x

    Concave with pi_i(0) = 0, so Assumption 1 holds and Proposition 1
    applies verbatim -- the cut machinery in CapacityOracle is shared with
    the triangular case unchanged.

    Segments are (bar, hat): bar = min(x, a_i) on the linear piece, hat =
    (x - a_i)_+ on the quadratic piece. Setting pi_i'(x) = nu on the
    quadratic piece gives hat = (b-a)((p-c) - nu)/(p-s); on the linear
    piece the slope is the constant (p-c), so bar is bang-bang at nu =
    p_i - c_i -- exactly the tie CapacityOracle._fill_marginal_gap
    resolves. At nu = 0 the response reproduces x^U_i = a + (b-a)(p-c)/(p-s).
    """

    def __init__(
        self,
        a: Sequence[float],
        b: Sequence[float],
        price_p: Sequence[float],
        cost_c: Sequence[float],
        salvage_s: Sequence[float],
        C: float,
        *,
        backend: str = "inline",
        max_bisection_iterations: int = 100,
        feasibility_tol: float = 1.0e-9,
    ) -> None:
        _require(
            backend == "inline",
            "UniformCapacityOracle has no reference-solver backend (there is no "
            "uniform counterpart to triangular_2018.py); use backend='inline'",
        )
        self.backend = backend
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.price_p = np.asarray(price_p, dtype=float)
        self.cost_c = np.asarray(cost_c, dtype=float)
        self.salvage_s = np.asarray(salvage_s, dtype=float)
        self.n = self.a.size
        self.C = float(C)
        self.max_bisection_iterations = max_bisection_iterations
        self.feasibility_tol = feasibility_tol

        _require(bool(np.all(self.a < self.b)), "a must be strictly less than b")
        _require(bool(np.all(self.price_p > self.cost_c)), "price must exceed cost")
        _require(bool(np.all(self.price_p > self.salvage_s)), "price must exceed salvage")
        _require(self.C >= 0.0 and np.isfinite(self.C), "C must be finite and nonnegative")

        self.coeff_lin = self.price_p - self.cost_c
        self.span = self.b - self.a
        self.underage_over_range = self.coeff_lin / (self.price_p - self.salvage_s)

    def response(self, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
        # Strict ">" mirrors triangular_2018._per_product_response: at
        # multiplier == coeff_lin the product must come out inactive so the
        # bisection's upper bracket is unambiguous, with _fill_marginal_gap
        # putting back exactly the quantity the capacity constraint needs.
        bar = np.where(self.coeff_lin > multiplier, self.a, 0.0)
        hat_interior = self.span * (self.coeff_lin - multiplier) / (self.price_p - self.salvage_s)
        hat = np.where(
            multiplier >= self.coeff_lin, 0.0, np.clip(hat_interior, 0.0, self.span)
        )
        return bar, hat

    def per_product_profit(self, *segments: np.ndarray) -> np.ndarray:
        bar, hat = segments
        return self.coeff_lin * (bar + hat) + (self.salvage_s - self.price_p) * hat**2 / (
            2.0 * self.span
        )


class DistributionFreeCapacityOracle(CapacityOracle):
    """Distribution-free (Scarf) demand known only through mu_i, sigma_i --
    manuscript (11)/(12), written as the profit form Assumption 1 needs.

    The manuscript minimizes a cost, g_i(x) = theta_i(x) + lambda (p-s)^2
    sigma^2 x^2 with

        theta_i(x) = (c-s)x + (p-s)[sqrt(sigma^2 + (x-mu)^2) - x + mu]/2,

    and not stocking i still costs g_i(0) > 0. Subtracting that constant
    gives the profit form

        pi_i(x) = g_i(0) - g_i(x),

    with pi_i(0) = 0 and pi_i concave (g_i is strictly convex: g_i'' =
    (p-s) sigma^2 / 2 (sigma^2+(x-mu)^2)^{3/2} + 2 lambda (p-s)^2 sigma^2
    > 0). So Assumption 1 holds, Proposition 1 applies, and the master
    and cuts are the shared ones -- exactly as for the uniform and
    triangular families.

    Two differences from those two oracles, both consequences of g_i being
    strictly convex rather than piecewise-polynomial:

      * There is no linear segment and hence no bang-bang tie, so ``a`` is
        the zero vector and CapacityOracle._fill_marginal_gap is a no-op
        by construction. The response is continuous in the multiplier, so
        the outer bisection reaches capacity equality on its own.
      * The per-multiplier response has no closed form (the lambda term
        makes pi_i'(x) = nu transcendental), so it is solved by an inner
        vectorized bisection on the monotone g_i'. That is accurate to
        full double precision, but -- unlike the closed-form uniform and
        triangular conjugates -- it is not exact in principle. An
        underestimated pi*_i(nu) would make a cut too tight; the error
        here is ~1e-16 relative, eleven orders of magnitude below the
        callback's own violation tolerance, so it cannot affect which
        cuts are added.

    ``coeff_lin`` is pi_i'(0) = -g_i'(0), the marginal profit of the first
    unit -- the same role (p-c) plays in the other two oracles, and the
    multiplier at or above which product i orders nothing.
    """

    def __init__(
        self,
        mu: Sequence[float],
        sigma: Sequence[float],
        lambda_: float,
        price_p: Sequence[float],
        cost_c: Sequence[float],
        salvage_s: Sequence[float],
        x_u: Sequence[float],
        C: float,
        *,
        backend: str = "inline",
        max_bisection_iterations: int = 100,
        feasibility_tol: float = 1.0e-9,
        response_iterations: int = 64,
    ) -> None:
        _require(
            backend == "inline",
            "DistributionFreeCapacityOracle has no reference-solver backend; "
            "use backend='inline'",
        )
        self.backend = backend
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.lambda_ = float(lambda_)
        self.price_p = np.asarray(price_p, dtype=float)
        self.cost_c = np.asarray(cost_c, dtype=float)
        self.salvage_s = np.asarray(salvage_s, dtype=float)
        self.x_u = np.asarray(x_u, dtype=float)
        self.n = self.mu.size
        self.C = float(C)
        self.max_bisection_iterations = max_bisection_iterations
        self.feasibility_tol = feasibility_tol
        self.response_iterations = response_iterations

        _require(bool(np.all(self.sigma > 0.0)), "sigma must be strictly positive")
        _require(bool(np.all(self.mu > 0.0)), "mu must be strictly positive")
        _require(bool(np.all(self.price_p > self.cost_c)), "price must exceed cost")
        _require(bool(np.all(self.price_p > self.salvage_s)), "price must exceed salvage")
        _require(bool(np.all(self.x_u >= 0.0)), "x_u must be nonnegative")
        _require(self.lambda_ >= 0.0, "lambda must be nonnegative")
        _require(self.C >= 0.0 and np.isfinite(self.C), "C must be finite and nonnegative")

        self.spread = self.price_p - self.salvage_s
        # Coefficient of x in g'(x), i.e. the derivative of the variance penalty.
        self.penalty_slope = 2.0 * self.lambda_ * self.spread**2 * self.sigma**2
        self.cost_zero = self._cost(np.zeros(self.n))
        # No linear segment: disables the marginal-gap tie-break, which
        # this family does not need (see the class docstring).
        self.a = np.zeros(self.n)
        self.coeff_lin = -self._cost_derivative(np.zeros(self.n))

    def _cost(self, x: np.ndarray) -> np.ndarray:
        """g_i(x), vectorized."""
        radius = np.sqrt(self.sigma**2 + (x - self.mu) ** 2)
        theta = (self.cost_c - self.salvage_s) * x + self.spread * (radius - x + self.mu) / 2.0
        return theta + self.lambda_ * self.spread**2 * self.sigma**2 * x**2

    def _cost_derivative(self, x: np.ndarray) -> np.ndarray:
        """g_i'(x), vectorized. Strictly increasing in x."""
        radius = np.sqrt(self.sigma**2 + (x - self.mu) ** 2)
        return (
            (self.cost_c - self.salvage_s)
            + self.spread / 2.0 * ((x - self.mu) / radius - 1.0)
            + self.penalty_slope * x
        )

    def response(self, multiplier: float) -> tuple[np.ndarray]:
        """argmax_{0<=x<=x^U} {pi_i(x) - nu x} = argmin {g_i(x) + nu x},
        i.e. the root of g_i'(x) = -nu, clipped to [0, x^U_i].

        g_i' is strictly increasing and g_i'(x^U_i) = 0 >= -nu for every
        nu >= 0, so the root never exceeds x^U_i and a bisection on
        [0, x^U_i] is well posed. Products whose first unit is already
        unprofitable at this capacity price (coeff_lin <= nu) order
        nothing; the weak inequality mirrors the strict ">" convention in
        the other two oracles, keeping the outer bracket unambiguous.
        """
        price = float(multiplier)
        low = np.zeros(self.n)
        high = self.x_u.copy()
        for _ in range(self.response_iterations):
            middle = 0.5 * (low + high)
            below = self._cost_derivative(middle) < -price
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        quantity = 0.5 * (low + high)
        return (np.where(self.coeff_lin <= price, 0.0, quantity),)

    def per_product_profit(self, *segments: np.ndarray) -> np.ndarray:
        (quantity,) = segments
        return self.cost_zero - self._cost(quantity)

    def validate_unconstrained(self, x_u: Sequence[float]) -> None:
        """Compare the nu=0 response to x^U by OBJECTIVE VALUE, not by
        position -- the base class's positional test is the wrong
        instrument for this family.

        g_i is extremely flat near its minimum for some products (its
        curvature there is (p-s) sigma^2 / 2 (sigma^2+(x-mu)^2)^{3/2},
        tiny when sigma is large relative to the distance from mu), and
        the Table 3 instances' x_u column comes from a golden-section
        minimizer validated to a scaled DERIVATIVE residual of 1e-7
        (generate_manuscript_table_instances.DF_DERIVATIVE_TOLERANCE),
        not to a position tolerance. Over all 30 replications at n=100
        and a sample at n=700, this oracle's root of g'(x) = 0 and the
        stored x_u differ by up to 1.1e-5 in position (2.2e-7 relative)
        -- which the base class's rtol/atol of 1e-7 rejects -- while
        their profits differ by at most 5.6e-13 relative, in both
        directions. Both points are optimal to within floating point;
        the position is simply not pinned down by the data.

        Profit agreement is the invariant that actually matters (it is
        what every cut is built from) and it still fails loudly on a
        genuine parameter mismatch, so it is what is checked here, with
        a loose positional bound retained to catch gross errors.
        """
        target = np.asarray(x_u, dtype=float)
        (unconstrained,) = self.response(0.0)
        at_response = self.per_product_profit(unconstrained)
        at_target = self.per_product_profit(target)
        scale = np.maximum(1.0, np.abs(at_target))
        _require(
            bool(np.all(np.abs(at_response - at_target) <= 1.0e-9 * scale)),
            "oracle's nu=0 response does not reproduce the instance's x_u profit",
        )
        _require(
            bool(np.allclose(unconstrained, target, rtol=1.0e-4, atol=1.0e-4)),
            "oracle's nu=0 response is far from the instance's x_u",
        )


def dual_price_cut(
    oracle: TriangularCapacityOracle, mask: np.ndarray
) -> tuple[float, np.ndarray, CapacityValue]:
    """The cut (*): y <= constant + sum_i coefficients[i] * z_i, valid for
    every z and tight at ``mask``. Returns (constant, coefficients, the
    f(S) evaluation it was built from).
    """
    capacity_value = oracle.evaluate(mask)
    coefficients = oracle.conjugates(capacity_value.multiplier)
    constant = capacity_value.multiplier * oracle.C
    return constant, coefficients, capacity_value


def nemhauser_wolsey_cut(
    oracle: TriangularCapacityOracle,
    mask: np.ndarray,
    marginal_at_full_set: np.ndarray,
) -> tuple[float, np.ndarray, CapacityValue]:
    """Inequality (6b) verbatim, with exact marginals. Rearranged to the
    same ``y <= constant + coefficients @ z`` shape as dual_price_cut by
    folding the -sum_{i in S} rho_i(N\\{i}) term into the constant.

    Costs |N\\S| evaluations of f (one per candidate addition); see the
    module docstring on why "dual" is the default.
    """
    mask = np.asarray(mask, dtype=bool)
    capacity_value = oracle.evaluate(mask)
    coefficients = np.zeros(oracle.n)
    coefficients[mask] = marginal_at_full_set[mask]
    for index in np.where(~mask)[0]:
        augmented = mask.copy()
        augmented[index] = True
        coefficients[index] = oracle.evaluate(augmented).value - capacity_value.value
    constant = capacity_value.value - float(marginal_at_full_set[mask].sum())
    return constant, coefficients, capacity_value


def marginals_at_full_set(oracle: TriangularCapacityOracle) -> np.ndarray:
    """rho_i(N\\{i}) = f(N) - f(N\\{i}) for every i -- the middle-sum
    coefficients of (6b). n evaluations of f, computed once.
    """
    full = np.ones(oracle.n, dtype=bool)
    full_value = oracle.evaluate(full).value
    marginals = np.zeros(oracle.n)
    for index in range(oracle.n):
        without = full.copy()
        without[index] = False
        marginals[index] = full_value - oracle.evaluate(without).value
    return marginals


@dataclasses.dataclass(slots=True)
class SubmodularCutSolveResult:
    """Outcome of one master solve, in the shape the runners report."""

    status: str
    objective: float | None
    selected: np.ndarray | None
    quantities: np.ndarray | None
    node_count: int | None
    solve_seconds: float
    cut_count: int
    oracle_evaluations: int
    mip_gap: float | None
    best_bound: float | None


def solve_triangular_fixed_cost_submodular_cuts(
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
    oracle_backend: str = DEFAULT_ORACLE_BACKEND,
    **kwargs: Any,
) -> SubmodularCutSolveResult:
    """Triangular-demand entry point: builds the oracle and delegates to
    solve_fixed_cost_submodular_cuts.
    """
    oracle = TriangularCapacityOracle(
        a, m, b, price_p, cost_c, salvage_s, C, backend=oracle_backend
    )
    return solve_fixed_cost_submodular_cuts(oracle, x_u, k, K, **kwargs)


def solve_uniform_fixed_cost_submodular_cuts(
    a: Sequence[float],
    b: Sequence[float],
    price_p: Sequence[float],
    cost_c: Sequence[float],
    salvage_s: Sequence[float],
    x_u: Sequence[float],
    k: Sequence[float],
    C: float,
    K: float,
    **kwargs: Any,
) -> SubmodularCutSolveResult:
    """Uniform-demand entry point (manuscript Table 1's model). Identical
    master and cuts as the triangular case -- see UniformCapacityOracle.
    """
    oracle = UniformCapacityOracle(a, b, price_p, cost_c, salvage_s, C)
    return solve_fixed_cost_submodular_cuts(oracle, x_u, k, K, **kwargs)


def solve_distribution_free_fixed_cost_submodular_cuts(
    mu: Sequence[float],
    sigma: Sequence[float],
    lambda_: float,
    price_p: Sequence[float],
    cost_c: Sequence[float],
    salvage_s: Sequence[float],
    x_u: Sequence[float],
    k: Sequence[float],
    C: float,
    K: float,
    **kwargs: Any,
) -> SubmodularCutSolveResult:
    """Distribution-free entry point (manuscript Table 3's model (11)/(12)).
    Identical master and cuts as the other two families -- see
    DistributionFreeCapacityOracle.

    NOTE ON SCALE: the returned ``objective`` is the PROFIT sum_i pi_i(x_i)
    = sum_i g_i(0) - sum_i g_i(x_i), not the cost that (11a)/(12)
    minimize. The two differ by the constant sum_i g_i(0), so they have
    the same optimizer but different relative gaps; the caller is
    responsible for converting before comparing objectives or gaps
    against a MISOCP solve (distribution_free_table3.zero_quantity_cost
    provides the constant).
    """
    oracle = DistributionFreeCapacityOracle(
        mu, sigma, lambda_, price_p, cost_c, salvage_s, x_u, C
    )
    return solve_fixed_cost_submodular_cuts(oracle, x_u, k, K, **kwargs)


def solve_fixed_cost_submodular_cuts(
    oracle: CapacityOracle,
    x_u: Sequence[float],
    k: Sequence[float],
    K: float,
    *,
    time_limit_seconds: float = 3600.0,
    mip_gap: float = 1.0e-3,
    threads: int = 1,
    verbose: bool = False,
    cut_family: str = DEFAULT_CUT_FAMILY,
    violation_abs_tol: float = DEFAULT_CUT_VIOLATION_ABS_TOL,
    violation_rel_tol: float = DEFAULT_CUT_VIOLATION_REL_TOL,
) -> SubmodularCutSolveResult:
    """Solve max{f(S) : sum_{i in S} k_i <= K} with GUROBI + lazy
    submodular cuts, for whichever demand distribution ``oracle``
    implements. ``x_u`` is accepted for interface symmetry with the other
    builders and validated against the oracle's own nu=0 response; the
    bound x_i <= x^U_i is itself redundant (Proposition 1 / Section 1).
    """
    import gurobipy as gp
    from gurobipy import GRB

    _require(cut_family in CUT_FAMILY_CHOICES, f"cut_family must be one of {CUT_FAMILY_CHOICES}")
    C = oracle.C
    n = oracle.n
    k = np.asarray(k, dtype=float)
    _require(k.shape == (n,), f"k must have shape ({n},)")
    _require(bool(np.all(k >= 0.0)), "k must be nonnegative")
    _require(np.isfinite(K) and K >= 0.0, "K must be finite and nonnegative")

    # The unconstrained single-product optima are exactly the nu=0
    # response; a mismatch means the instance data and this oracle
    # disagree about the model, which would silently corrupt every cut.
    oracle.validate_unconstrained(x_u)

    full_set_marginals = (
        marginals_at_full_set(oracle) if cut_family == "nemhauser-wolsey" else None
    )
    evaluation_counter = {"count": 0}

    def build_cut(mask: np.ndarray) -> tuple[float, np.ndarray, CapacityValue]:
        evaluation_counter["count"] += 1
        if cut_family == "dual":
            return dual_price_cut(oracle, mask)
        evaluation_counter["count"] += int((~mask).sum())
        return nemhauser_wolsey_cut(oracle, mask, full_set_marginals)

    model = gp.Model(MODEL_ID)
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_limit_seconds
    model.Params.MIPGap = mip_gap
    model.Params.Threads = threads

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    y = model.addVar(lb=0.0, ub=GRB.INFINITY, name="y")
    model.setObjective(y, GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(float(k[i]) * z[i] for i in range(n)) <= float(K), name="budget")

    z_list = [z[i] for i in range(n)]

    def add_cut_as_constraint(mask: np.ndarray) -> None:
        constant, coefficients, _ = build_cut(mask)
        model.addConstr(y <= constant + gp.LinExpr(coefficients.tolist(), z_list))

    # Seed cuts. S = {} gives nu = 0, i.e. y <= sum_i pi_i(x^U_i) z_i (the
    # uncapacitated bound); S = N bounds y by f(N) and makes the master
    # bounded from the start.
    add_cut_as_constraint(np.zeros(n, dtype=bool))
    add_cut_as_constraint(np.ones(n, dtype=bool))

    state = {"cuts": 2}

    def callback(cb_model: Any, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        z_values = np.asarray(cb_model.cbGetSolution(z_list), dtype=float)
        y_value = float(cb_model.cbGetSolution(y))
        mask = z_values > 0.5
        constant, coefficients, capacity_value = build_cut(mask)
        tolerance = violation_abs_tol + violation_rel_tol * max(1.0, abs(capacity_value.value))
        if y_value > capacity_value.value + tolerance:
            # Sparsify: pi*_i(nu) >= 0 always, and products worth nothing
            # at this capacity price contribute nothing to the cut.
            support = np.where(np.abs(coefficients) > 0.0)[0]
            cb_model.cbLazy(
                y
                <= constant
                + gp.LinExpr(
                    coefficients[support].tolist(), [z_list[i] for i in support]
                )
            )
            state["cuts"] += 1

    started = time.perf_counter()
    model.optimize(callback)
    elapsed = time.perf_counter() - started

    status_names = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.INTERRUPTED: "interrupted",
    }
    status = status_names.get(model.Status, str(model.Status))

    selected = None
    quantities = None
    objective = None
    if model.SolCount > 0:
        z_values = np.array([var.X for var in z_list], dtype=float)
        selected = z_values > 0.5
        # Report the true achievable profit of the selected set, recomputed
        # from the oracle (the master's y is only an upper bound until the
        # cut for this set has been added).
        final = oracle.evaluate(selected)
        objective = final.value
        quantities = final.quantities

    return SubmodularCutSolveResult(
        status=status,
        objective=objective,
        selected=selected,
        quantities=quantities,
        node_count=int(model.NodeCount),
        solve_seconds=elapsed,
        cut_count=state["cuts"],
        oracle_evaluations=evaluation_counter["count"],
        mip_gap=float(model.MIPGap) if model.SolCount > 0 else None,
        best_bound=float(model.ObjBound) if model.SolCount > 0 else None,
    )


def _random_instance(seed: int, n: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 50.0, size=n)
    spans = rng.uniform(80.0, 150.0, size=n)
    b = a + spans
    m = a + rng.uniform(0.1, 0.9, size=n) * spans
    c = rng.uniform(1.0, 5.0, size=n)
    p = c * rng.uniform(1.2, 2.0, size=n)
    s = rng.uniform(-0.5, 0.5, size=n) * c
    x_u = np.array(
        [t2018.triangular_unconstrained_quantity(a[i], m[i], b[i], p[i], c[i], s[i]) for i in range(n)]
    )
    k = rng.uniform(1.0, 5.0, size=n)
    return {
        "a": a, "m": m, "b": b, "price_p": p, "cost_c": c, "salvage_s": s,
        "x_u": x_u, "k": k,
        "C": 0.3 * float(np.sum(x_u)), "K": 0.5 * float(np.sum(k)),
    }


def _self_test() -> None:
    """Checks, in increasing order of strength:

    1. the vectorized per-product profit reproduces the scalar closed-form
       oracle the rest of the project validates against;
    2. the nu=0 response reproduces the unconstrained optima x^U;
    3. f(S) agrees with triangular_2018.py's independent dual solve built
       on the subset alone, and its duality gap is ~0;
    4. f is in fact monotone and submodular on random subsets;
    5. the full master + lazy cuts reproduces brute-force enumeration over
       all knapsack-feasible subsets for tiny n.
    """
    import itertools

    import run_triangular_table2_experiment as rte

    data = _random_instance(seed=0, n=6)
    a, m, b = data["a"], data["m"], data["b"]
    p, c, s = data["price_p"], data["cost_c"], data["salvage_s"]
    x_u, k, C, K = data["x_u"], data["k"], data["C"], data["K"]
    oracle = TriangularCapacityOracle(a, m, b, p, c, s, C)
    n = oracle.n

    # 1. vectorized profit vs. the project's scalar closed form.
    probe = 0.5 * (a + b)
    bar = np.minimum(probe, a)
    hat = np.clip(probe - a, 0.0, m - a)
    tilde = np.clip(probe - m, 0.0, b - m)
    vectorized = oracle.per_product_profit(bar, hat, tilde)
    for i in range(n):
        expected = rte.triangular_expected_profit(probe[i], a[i], m[i], b[i], p[i], c[i], s[i])
        _require(
            abs(vectorized[i] - expected) <= 1.0e-9 * max(1.0, abs(expected)),
            f"self-test: vectorized profit disagrees at product {i}",
        )

    # 2. nu = 0 response == unconstrained optima.
    _require(
        bool(np.allclose(sum(oracle.response(0.0)), x_u, rtol=1.0e-9, atol=1.0e-9)),
        "self-test: nu=0 response does not reproduce x_u",
    )

    # 3. f(S) vs. triangular_2018's independent dual solve on the subset,
    #    and zero duality gap against (3).
    for subset in ((0, 2, 4), (1,), tuple(range(n))):
        mask = np.zeros(n, dtype=bool)
        mask[list(subset)] = True
        evaluated = oracle.evaluate(mask)
        index = list(subset)
        reference_formulation = t2018.build_cg2018_triangular_continuous(
            a[index], m[index], b[index], p[index], c[index], s[index], C
        )
        reference = t2018.solve_cg2018_triangular_continuous(reference_formulation, method="dual")
        _require(
            abs(evaluated.value - (-reference.fun)) <= 1.0e-6 * max(1.0, abs(evaluated.value)),
            f"self-test: f({subset}) disagrees with triangular_2018's dual solve "
            f"({evaluated.value} vs {-reference.fun})",
        )
        dual_bound = evaluated.multiplier * C + float(oracle.conjugates(evaluated.multiplier)[mask].sum())
        _require(
            dual_bound >= evaluated.value - 1.0e-6,
            f"self-test: weak duality violated for {subset}",
        )
        _require(
            dual_bound - evaluated.value <= 1.0e-6 * max(1.0, abs(evaluated.value)),
            f"self-test: duality gap too large for {subset} ({dual_bound - evaluated.value})",
        )

    # 4. monotonicity and submodularity on random subset pairs.
    rng = np.random.default_rng(1)
    for _ in range(50):
        left = rng.random(n) < 0.5
        right = rng.random(n) < 0.5
        f_left = oracle.evaluate(left).value
        f_right = oracle.evaluate(right).value
        f_union = oracle.evaluate(left | right).value
        f_inter = oracle.evaluate(left & right).value
        scale = max(1.0, abs(f_left), abs(f_right))
        _require(
            f_union + f_inter <= f_left + f_right + 1.0e-6 * scale,
            "self-test: submodularity violated",
        )
        _require(f_left <= f_union + 1.0e-6 * scale, "self-test: monotonicity violated")

    # 5. whole master vs. brute force over knapsack-feasible subsets.
    best = 0.0
    for size in range(n + 1):
        for subset in itertools.combinations(range(n), size):
            if float(k[list(subset)].sum()) > K + 1.0e-9:
                continue
            mask = np.zeros(n, dtype=bool)
            mask[list(subset)] = True
            best = max(best, oracle.evaluate(mask).value)

    # Both oracle backends must agree exactly on f(S) before either is
    # trusted to generate cuts.
    inline_oracle = TriangularCapacityOracle(a, m, b, p, c, s, C, backend="inline")
    cg2018_oracle = TriangularCapacityOracle(a, m, b, p, c, s, C, backend="cg2018")
    for _ in range(25):
        probe_mask = rng.random(n) < 0.5
        if not probe_mask.any():
            continue
        left = inline_oracle.evaluate(probe_mask)
        right = cg2018_oracle.evaluate(probe_mask)
        _require(
            abs(left.value - right.value) <= 1.0e-9 * max(1.0, abs(right.value)),
            f"self-test: oracle backends disagree ({left.value} vs {right.value})",
        )

    for cut_family, backend in (
        ("dual", "cg2018"), ("dual", "inline"), ("nemhauser-wolsey", "cg2018")
    ):
        result = solve_triangular_fixed_cost_submodular_cuts(
            a, m, b, p, c, s, x_u, k, C, K,
            time_limit_seconds=60.0, mip_gap=0.0, cut_family=cut_family,
            oracle_backend=backend,
        )
        _require(
            result.status == "optimal",
            f"self-test: {cut_family}/{backend} master did not solve to optimality ({result.status})",
        )
        _require(
            abs(result.objective - best) <= 1.0e-6 * max(1.0, abs(best)),
            f"self-test: {cut_family} optimum {result.objective} != brute force {best}",
        )
        # The reported quantities must be feasible and reproduce the value
        # through the project's own independent profit oracle.
        _require(
            float(result.quantities.sum()) <= C + 1.0e-6,
            f"self-test: {cut_family} solution violates capacity",
        )
        _require(
            float(k[result.selected].sum()) <= K + 1.0e-9,
            f"self-test: {cut_family} solution violates the fixed-cost budget",
        )
        recomputed = sum(
            rte.triangular_expected_profit(
                result.quantities[i], a[i], m[i], b[i], p[i], c[i], s[i]
            )
            for i in range(n)
        )
        _require(
            abs(recomputed - result.objective) <= 1.0e-6 * max(1.0, abs(result.objective)),
            f"self-test: {cut_family} reported objective {result.objective} does not match "
            f"the independently recomputed {recomputed}",
        )
        print(
            f"  {cut_family}: optimum={result.objective:.6f} cuts={result.cut_count} "
            f"oracle_evals={result.oracle_evaluations} nodes={result.node_count}"
        )

    _uniform_self_test()
    _distribution_free_self_test()
    print("SUBMODULAR_CUTS_SELF_TEST_PASS")


def _uniform_self_test() -> None:
    """The uniform counterpart: vectorized profit vs. the project's own
    scalar uniform oracle, nu=0 response vs. x^U, monotonicity and
    submodularity, and the whole master vs. brute-force enumeration.
    """
    import itertools

    import run_uniform_table1_experiment as rue

    rng = np.random.default_rng(5)
    n = 6
    a = rng.uniform(0.0, 50.0, size=n)
    b = a + rng.uniform(80.0, 150.0, size=n)
    c = rng.uniform(1.0, 5.0, size=n)
    p = c * rng.uniform(1.2, 2.0, size=n)
    s = rng.uniform(-0.5, 0.5, size=n) * c
    x_u = np.array(
        [rue.uniform_unconstrained_quantity(a[i], b[i], p[i], c[i], s[i]) for i in range(n)]
    )
    k = rng.uniform(1.0, 5.0, size=n)
    C = 0.3 * float(np.sum(x_u))
    K = 0.5 * float(np.sum(k))
    oracle = UniformCapacityOracle(a, b, p, c, s, C)

    # 1. vectorized profit vs. the project's scalar uniform closed form,
    #    probed on both sides of the a_i kink.
    for probe in (0.5 * a, 0.5 * (a + b), np.minimum(x_u, b)):
        bar = np.minimum(probe, a)
        hat = np.clip(probe - a, 0.0, b - a)
        vectorized = oracle.per_product_profit(bar, hat)
        for i in range(n):
            expected = rue.uniform_expected_profit(probe[i], a[i], b[i], p[i], c[i], s[i])
            _require(
                abs(vectorized[i] - expected) <= 1.0e-9 * max(1.0, abs(expected)),
                f"uniform self-test: profit disagrees at product {i} "
                f"({vectorized[i]} vs {expected})",
            )

    # 2. nu = 0 response == unconstrained optima.
    _require(
        bool(np.allclose(sum(oracle.response(0.0)), x_u, rtol=1.0e-9, atol=1.0e-9)),
        "uniform self-test: nu=0 response does not reproduce x_u",
    )

    # 3. zero duality gap at the recovered multiplier.
    for subset in ((0, 2, 4), (1,), tuple(range(n))):
        mask = np.zeros(n, dtype=bool)
        mask[list(subset)] = True
        evaluated = oracle.evaluate(mask)
        dual_bound = evaluated.multiplier * C + float(
            oracle.conjugates(evaluated.multiplier)[mask].sum()
        )
        _require(
            dual_bound - evaluated.value <= 1.0e-6 * max(1.0, abs(evaluated.value)),
            f"uniform self-test: duality gap too large for {subset}",
        )

    # 4. monotonicity and submodularity.
    for _ in range(50):
        left = rng.random(n) < 0.5
        right = rng.random(n) < 0.5
        f_left = oracle.evaluate(left).value
        f_right = oracle.evaluate(right).value
        f_union = oracle.evaluate(left | right).value
        f_inter = oracle.evaluate(left & right).value
        scale = max(1.0, abs(f_left), abs(f_right))
        _require(
            f_union + f_inter <= f_left + f_right + 1.0e-6 * scale,
            "uniform self-test: submodularity violated",
        )
        _require(f_left <= f_union + 1.0e-6 * scale, "uniform self-test: monotonicity violated")

    # 5. whole master vs. brute force.
    best = 0.0
    for size in range(n + 1):
        for subset in itertools.combinations(range(n), size):
            if float(k[list(subset)].sum()) > K + 1.0e-9:
                continue
            mask = np.zeros(n, dtype=bool)
            mask[list(subset)] = True
            best = max(best, oracle.evaluate(mask).value)

    for cut_family in CUT_FAMILY_CHOICES:
        result = solve_uniform_fixed_cost_submodular_cuts(
            a, b, p, c, s, x_u, k, C, K,
            time_limit_seconds=60.0, mip_gap=0.0, cut_family=cut_family,
        )
        _require(
            result.status == "optimal",
            f"uniform self-test: {cut_family} did not solve to optimality ({result.status})",
        )
        _require(
            abs(result.objective - best) <= 1.0e-6 * max(1.0, abs(best)),
            f"uniform self-test: {cut_family} optimum {result.objective} != brute force {best}",
        )
        _require(
            float(result.quantities.sum()) <= C + 1.0e-6
            and float(k[result.selected].sum()) <= K + 1.0e-9,
            f"uniform self-test: {cut_family} solution violates capacity or budget",
        )
        recomputed = sum(
            rue.uniform_expected_profit(result.quantities[i], a[i], b[i], p[i], c[i], s[i])
            for i in range(n)
        )
        _require(
            abs(recomputed - result.objective) <= 1.0e-6 * max(1.0, abs(result.objective)),
            f"uniform self-test: {cut_family} objective does not match the independent "
            f"recomputation ({result.objective} vs {recomputed})",
        )
        print(
            f"  uniform/{cut_family}: optimum={result.objective:.6f} cuts={result.cut_count} "
            f"oracle_evals={result.oracle_evaluations} nodes={result.node_count}"
        )


def _distribution_free_self_test() -> None:
    """The distribution-free counterpart: vectorized profit vs. the
    generator's own scalar g, the inner root-find's stationarity, nu=0
    response vs. x^U, monotonicity and submodularity, and the whole
    master vs. brute-force enumeration.
    """
    import itertools

    import generate_manuscript_table_instances as gmti

    rng = np.random.default_rng(11)
    n = 6
    lambda_ = gmti.DF_LAMBDA
    mu = rng.uniform(100.0, 500.0, size=n)
    sigma = np.array([rng.uniform(m / 6.0, m / 2.0) for m in mu])
    c = rng.uniform(1.0, 9.0, size=n)
    p = c * (1.1 + rng.uniform(0.0, 1.4, size=n))
    s = rng.uniform(-0.9, 0.9, size=n) * c
    x_u = np.array(
        [
            gmti.distribution_free_unconstrained_quantity(
                mu[i], sigma[i], lambda_, p[i], c[i], s[i]
            )[0]
            for i in range(n)
        ]
    )
    k = rng.uniform(1.0, 5.0, size=n)
    C = 0.3 * float(np.sum(x_u))
    K = 0.5 * float(np.sum(k))
    oracle = DistributionFreeCapacityOracle(mu, sigma, lambda_, p, c, s, x_u, C)

    def scalar_profit(index: int, quantity: float) -> float:
        args = (mu[index], sigma[index], lambda_, p[index], c[index], s[index])
        return gmti.df_g(0.0, *args) - gmti.df_g(quantity, *args)

    # 1. vectorized profit vs. the generator's own scalar g.
    for probe in (0.25 * x_u, 0.75 * x_u, x_u):
        vectorized = oracle.per_product_profit(probe)
        for i in range(n):
            expected = scalar_profit(i, probe[i])
            _require(
                abs(vectorized[i] - expected) <= 1.0e-9 * max(1.0, abs(expected)),
                f"distribution-free self-test: profit disagrees at product {i} "
                f"({vectorized[i]} vs {expected})",
            )

    # 2. nu = 0 response == unconstrained optima, and the inner root-find
    #    really solves g'(x) = -nu wherever it lands strictly inside.
    _require(
        bool(np.allclose(sum(oracle.response(0.0)), x_u, rtol=1.0e-7, atol=1.0e-7)),
        "distribution-free self-test: nu=0 response does not reproduce x_u",
    )
    for price in (0.0, 0.05, 0.2, 0.5):
        (quantity,) = oracle.response(price)
        interior = (quantity > 1.0e-8) & (quantity < x_u - 1.0e-8)
        residual = np.abs(oracle._cost_derivative(quantity) + price)[interior]
        if residual.size:
            _require(
                float(residual.max()) <= 1.0e-9 * max(1.0, price),
                f"distribution-free self-test: stationarity residual {residual.max()} "
                f"too large at nu={price}",
            )

    # 3. zero duality gap at the recovered multiplier.
    for subset in ((0, 2, 4), (1,), tuple(range(n))):
        mask = np.zeros(n, dtype=bool)
        mask[list(subset)] = True
        evaluated = oracle.evaluate(mask)
        dual_bound = evaluated.multiplier * C + float(
            oracle.conjugates(evaluated.multiplier)[mask].sum()
        )
        _require(
            dual_bound - evaluated.value <= 1.0e-6 * max(1.0, abs(evaluated.value)),
            f"distribution-free self-test: duality gap too large for {subset}",
        )

    # 4. monotonicity and submodularity.
    for _ in range(50):
        left = rng.random(n) < 0.5
        right = rng.random(n) < 0.5
        f_left = oracle.evaluate(left).value
        f_right = oracle.evaluate(right).value
        f_union = oracle.evaluate(left | right).value
        f_inter = oracle.evaluate(left & right).value
        scale = max(1.0, abs(f_left), abs(f_right))
        _require(
            f_union + f_inter <= f_left + f_right + 1.0e-6 * scale,
            "distribution-free self-test: submodularity violated",
        )
        _require(
            f_left <= f_union + 1.0e-6 * scale,
            "distribution-free self-test: monotonicity violated",
        )

    # 5. whole master vs. brute force.
    best = 0.0
    for size in range(n + 1):
        for subset in itertools.combinations(range(n), size):
            if float(k[list(subset)].sum()) > K + 1.0e-9:
                continue
            mask = np.zeros(n, dtype=bool)
            mask[list(subset)] = True
            best = max(best, oracle.evaluate(mask).value)

    for cut_family in CUT_FAMILY_CHOICES:
        result = solve_distribution_free_fixed_cost_submodular_cuts(
            mu, sigma, lambda_, p, c, s, x_u, k, C, K,
            time_limit_seconds=60.0, mip_gap=0.0, cut_family=cut_family,
        )
        _require(
            result.status == "optimal",
            f"distribution-free self-test: {cut_family} did not solve to optimality "
            f"({result.status})",
        )
        _require(
            abs(result.objective - best) <= 1.0e-6 * max(1.0, abs(best)),
            f"distribution-free self-test: {cut_family} optimum {result.objective} "
            f"!= brute force {best}",
        )
        _require(
            float(result.quantities.sum()) <= C + 1.0e-6
            and float(k[result.selected].sum()) <= K + 1.0e-9,
            f"distribution-free self-test: {cut_family} solution violates capacity or budget",
        )
        recomputed = sum(scalar_profit(i, result.quantities[i]) for i in range(n))
        _require(
            abs(recomputed - result.objective) <= 1.0e-6 * max(1.0, abs(result.objective)),
            f"distribution-free self-test: {cut_family} objective does not match the "
            f"independent recomputation ({result.objective} vs {recomputed})",
        )
        print(
            f"  distribution-free/{cut_family}: optimum={result.objective:.6f} "
            f"cuts={result.cut_count} oracle_evals={result.oracle_evaluations} "
            f"nodes={result.node_count}"
        )


if __name__ == "__main__":
    _self_test()
