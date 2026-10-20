#!/usr/bin/env python3
"""Compare the Chernonog and Goldberg (2018) Triangular model (solved with a
general local NLP solver, standing in for CONOPT -- see triangular_2018.py)
against the "current" (Newsvendor_current.pdf) conic/perspective Triangular
model with zero fixed costs (triangular_current.py's
build_current_triangular_conic_continuous), which has no binary/indicator
variables since there is no display/fixed-cost budget constraint to enforce.

For each n in {100, 200, 300, 400, 500}, this script generates 30 random
instances following the 2018 paper's own generation scheme (Section 6):

    a_i          ~ discrete uniform on {0, ..., 100}
    b_i - a_i    ~ discrete uniform on {100, ..., 500}
    d_i (mode)   ~ discrete uniform on {a_i + 1, ..., b_i - 1}
    c_i          ~ continuous uniform (0, 10)
    w_i          ~ discrete uniform {-1, +1}                (per product)
    tau          ~ continuous uniform (0, 1)                (shared per instance)
    h_i = c_i * w_i * tau,  so the salvage value s_i = -h_i
    eta          ~ continuous uniform (0, 1)                (shared per instance)
    v_i = c_i * (1.1 + 1.4 * eta)                            (this project's p_i)

(The 2018 paper's text subscripts a_i, (b_i-a_i), d_i, c_i and w_i per
product i, but not tau or eta, which is why those two are drawn once per
instance here rather than once per product.)

Both models here use a single unit-coefficient capacity constraint
sum_i x_i <= C (i.e. the 2018 paper's resource constraint (2) with r_i1 = 1
and R_1 = C, with its separate monetary budget constraint (1) disabled) so
that they share exactly the same feasible region and objective -- this is
what makes their optimal objective values directly comparable. C is drawn as
C ~ continuous uniform (0, sum_i x_U_i), matching the 2018 paper's scheme for
generating resource right-hand sides from the unconstrained optimum.

Generated instances are saved as plain-text files under an instances
subdirectory so they can be inspected or reused without regenerating.

For every instance this script solves both models, independently recomputes
each one's economic objective from its own returned order quantities using
the shared closed-form Triangular profit function (so the comparison does
not depend on either model's own reported/internal objective value or
normalization constant), and checks the two for equality within a small
tolerance whenever both solves report success. It then prints a summary,
per n, of each solver's running time (average, maximum, standard deviation)
and the number of runs in which that solver failed to report an optimal
solution.

The 2018 model is always solved as the actual cubic NLP (12)-(13), never the
exact-but-differently-structured Lagrangian-bisection shortcut also offered
by triangular_2018.py (that remains available for cross-checking only, via
--nlp-method dual). --nlp-method picks which NLP backend runs that cubic
model:

    --nlp-method ipopt          (default) Pyomo + IPOPT, a large-scale
                                 sparse interior-point NLP solver -- the
                                 practical CONOPT substitute. Solves n=500 in
                                 well under a second per instance.
    --nlp-method trust-constr   scipy.optimize.minimize -- the most literal
                                 CONOPT stand-in (a plain general-purpose
                                 local NLP solver), but it does not exploit
                                 this problem's sparsity and becomes
                                 impractically slow past a few hundred
                                 products (n=100 took 1-20 minutes per
                                 instance during development, and some
                                 instances did not converge within 2000
                                 iterations at all).
    --nlp-method SLSQP           Another scipy method; similarly slow.

Note: IPOPT is driven through Pyomo, not CVXPY -- CVXPY has no IPOPT backend
(it only targets DCP problems that reduce to conic/QP form and hands those
to solvers like MOSEK/CPLEX/CLARABEL/SCS; IPOPT never appears in its
installed_solvers()). Pyomo's SolverFactory("ipopt") is the standard way to
drive IPOPT from Python, so that is what triangular_2018.py's method="ipopt"
uses. See triangular_2018.py's module docstring for the full comparison and
the empirical timings behind these recommendations.

Usage:

    python run_table_new_experiment.py --self-test
    python run_table_new_experiment.py --n-values 100 --replications 1-3
    python run_table_new_experiment.py --n-values all --replications all
    python run_table_new_experiment.py --nlp-method trust-constr --n-values 100 --replications 1
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# EDITABLE DEFAULT CONFIGURATION
# ---------------------------------------------------------------------------

N_VALUES = (100, 200, 300, 400, 500)
REPLICATIONS = 30
MASTER_SEED = 20260831
GENERATION_VERSION = "TABLE_NEW_2018_VS_CONIC_V1"

DEFAULT_CONIC_SOLVER = "MOSEK"          # CPLEX, MOSEK, CLARABEL, or SCS
DEFAULT_NLP_METHOD = "ipopt"            # runs the actual 2018 cubic model.
                                         # "ipopt": Pyomo + IPOPT (fast,
                                         # practical CONOPT substitute).
                                         # "trust-constr"/"SLSQP": scipy's
                                         # generic solvers (the most literal
                                         # CONOPT stand-in, but impractically
                                         # slow past a few hundred products;
                                         # see triangular_2018.py). "dual" is
                                         # also accepted but is a *different*
                                         # (exact) solution method, not the
                                         # cubic NLP itself -- opt-in only,
                                         # never the default here.

# Objective-equality tolerance between the two solvers' independently
# recomputed economic objectives, for runs where both report success. The
# conic model's tower-of-variables SOCP lift (see triangular_current.py)
# carries inherently larger residual than a plain quadratic SOCP even at
# solver-reported optimality, so the relative tolerance here is looser than
# a typical numerical-equality check; run_triangular_table2_experiment.py's
# development notes have the same observation.
DEFAULT_OBJECTIVE_ABS_TOL = 1.0e-2
DEFAULT_OBJECTIVE_REL_TOL = 1.0e-4

DEFAULT_INSTANCES_DIR = "table_new_instances"
DEFAULT_RESULTS_DIR = "table_new_results"


class TableNewError(RuntimeError):
    """Raised when input, configuration, or validation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TableNewError(message)


# ---------------------------------------------------------------------------
# Closed-form Triangular oracle, independent of both models under test, used
# only to recompute a common economic objective from each solver's own
# returned order quantities (see module docstring).
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
    if x > b:
        return (salvage_s - cost_c) * x + (price_p - salvage_s) * (a + m + b) / 3.0
    x_bar = min(x, a)
    x_hat = min(max(x - a, 0.0), m - a)
    x_tilde = min(max(x - m, 0.0), b - m)
    return (
        (price_p - cost_c) * (x_bar + x_hat + x_tilde)
        + (salvage_s - price_p) * (m - a) * x_tilde / (b - a)
        + (salvage_s - price_p) * x_tilde ** 2 / (b - a)
        + (salvage_s - price_p)
        * (x_hat ** 3 / (m - a) - x_tilde ** 3 / (b - m))
        / (3.0 * (b - a))
    )


def recompute_economic_objective(
    a: Sequence[float], m: Sequence[float], b: Sequence[float],
    p: Sequence[float], c: Sequence[float], s: Sequence[float],
    q: Sequence[float],
) -> float:
    return sum(
        triangular_expected_profit(qi, ai, mi, bi, pi, ci, si)
        for qi, ai, mi, bi, pi, ci, si in zip(q, a, m, b, p, c, s, strict=True)
    )


# ---------------------------------------------------------------------------
# Instance generation, following Chernonog and Goldberg (2018), Section 6.
# ---------------------------------------------------------------------------

def derive_instance_seed(n: int, replication: int) -> int:
    return MASTER_SEED * 1_000_000 + n * 1_000 + replication


@dataclasses.dataclass(frozen=True, slots=True)
class TableNewInstance:
    n: int
    replication: int
    instance_seed: int
    a: tuple[float, ...]
    m: tuple[float, ...]
    b: tuple[float, ...]
    cost_c: tuple[float, ...]
    price_p: tuple[float, ...]
    salvage_s: tuple[float, ...]
    x_u: tuple[float, ...]
    tau: float
    eta: float
    C: float


def generate_2018_triangular_instance(n: int, replication: int) -> TableNewInstance:
    seed = derive_instance_seed(n, replication)
    rng = random.Random(seed)

    a_values = [float(rng.randint(0, 100)) for _ in range(n)]
    spans = [float(rng.randint(100, 500)) for _ in range(n)]
    b_values = [a_values[i] + spans[i] for i in range(n)]
    # d_i (2018 notation) ~ discrete uniform on {a_i+1, ..., b_i-1}; a_i, b_i
    # are themselves integers here, so this draw is a genuine discrete
    # uniform over the strictly-interior integers of [a_i, b_i].
    m_values = [
        float(rng.randint(int(a_values[i]) + 1, int(b_values[i]) - 1)) for i in range(n)
    ]
    cost_values = [rng.uniform(0.0, 10.0) for _ in range(n)]
    w_values = [rng.choice((-1.0, 1.0)) for _ in range(n)]
    tau = rng.uniform(0.0, 1.0)
    eta = rng.uniform(0.0, 1.0)
    price_values = [cost_values[i] * (1.1 + 1.4 * eta) for i in range(n)]
    # h_i = c_i * w_i * tau (2018 leftover-cost parameter); this project's
    # salvage value is s_i = -h_i.
    salvage_values = [-(cost_values[i] * w_values[i] * tau) for i in range(n)]

    for i in range(n):
        _require(price_values[i] > cost_values[i], f"product {i + 1}: price must exceed cost")
        _require(price_values[i] > salvage_values[i], f"product {i + 1}: price must exceed salvage")
        _require(0.0 <= a_values[i] < m_values[i] < b_values[i], f"product {i + 1}: invalid a<m<b")

    x_u = [
        triangular_unconstrained_quantity(
            a_values[i], m_values[i], b_values[i], price_values[i], cost_values[i], salvage_values[i]
        )
        for i in range(n)
    ]
    r3 = rng.uniform(0.0, 1.0)
    C = r3 * sum(x_u)

    return TableNewInstance(
        n=n,
        replication=replication,
        instance_seed=seed,
        a=tuple(a_values),
        m=tuple(m_values),
        b=tuple(b_values),
        cost_c=tuple(cost_values),
        price_p=tuple(price_values),
        salvage_s=tuple(salvage_values),
        x_u=tuple(x_u),
        tau=tau,
        eta=eta,
        C=C,
    )


def instance_filename(n: int, replication: int) -> str:
    return f"table_new_n{n:04d}_rep{replication:03d}.txt"


def write_instance_txt(path: Path, instance: TableNewInstance) -> None:
    lines = [
        "# Table-New Triangular instance (Chernonog and Goldberg 2018 generation scheme)",
        f"# generation_version={GENERATION_VERSION}",
        f"# master_seed={MASTER_SEED}",
        f"# n={instance.n}",
        f"# replication={instance.replication}",
        f"# instance_seed={instance.instance_seed}",
        f"# tau={instance.tau!r}",
        f"# eta={instance.eta!r}",
        f"# C={instance.C!r}",
        "# columns: product_id a m b cost_c price_p salvage_s x_u",
    ]
    for i in range(instance.n):
        lines.append(
            f"{i + 1} {instance.a[i]!r} {instance.m[i]!r} {instance.b[i]!r} "
            f"{instance.cost_c[i]!r} {instance.price_p[i]!r} {instance.salvage_s[i]!r} {instance.x_u[i]!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_instance_txt(path: Path) -> TableNewInstance:
    header: dict[str, str] = {}
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# columns:"):
            continue
        if line.startswith("#"):
            if "=" in line:
                key, _, value = line[1:].strip().partition("=")
                header[key.strip()] = value.strip()
            continue
        rows.append(line.split())

    n = int(header["n"])
    _require(len(rows) == n, f"{path}: expected {n} product rows, observed {len(rows)}")
    a, m, b, cost_c, price_p, salvage_s, x_u = ([] for _ in range(7))
    for index, row in enumerate(rows, start=1):
        _require(int(row[0]) == index, f"{path}: product_id sequence is not exactly 1..n")
        a.append(float(row[1]))
        m.append(float(row[2]))
        b.append(float(row[3]))
        cost_c.append(float(row[4]))
        price_p.append(float(row[5]))
        salvage_s.append(float(row[6]))
        x_u.append(float(row[7]))

    return TableNewInstance(
        n=n,
        replication=int(header["replication"]),
        instance_seed=int(header["instance_seed"]),
        a=tuple(a), m=tuple(m), b=tuple(b),
        cost_c=tuple(cost_c), price_p=tuple(price_p), salvage_s=tuple(salvage_s),
        x_u=tuple(x_u),
        tau=float(header["tau"]),
        eta=float(header["eta"]),
        C=float(header["C"]),
    )


def load_or_generate_instance(
    instances_dir: Path, n: int, replication: int, *, regenerate: bool
) -> TableNewInstance:
    path = instances_dir / instance_filename(n, replication)
    if path.is_file() and not regenerate:
        return read_instance_txt(path)
    instance = generate_2018_triangular_instance(n, replication)
    instances_dir.mkdir(parents=True, exist_ok=True)
    write_instance_txt(path, instance)
    return instance


# ---------------------------------------------------------------------------
# Solving both models and recording a comparable result per instance.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class SolveOutcome:
    solver_label: str
    success: bool
    status_text: str
    elapsed_seconds: float
    economic_objective: float | None
    q: tuple[float, ...] | None


def solve_2018(instance: TableNewInstance, *, method: str) -> SolveOutcome:
    import triangular_2018 as t2018

    formulation = t2018.build_cg2018_triangular_continuous(
        instance.a, instance.m, instance.b,
        instance.price_p, instance.cost_c, instance.salvage_s, instance.C,
    )
    started = time.perf_counter()
    result = t2018.solve_cg2018_triangular_continuous(formulation, method=method)
    elapsed = time.perf_counter() - started

    # A light sanity backstop only -- triangular_2018.py's own result.success
    # already applies a method-appropriate scaled feasibility check (1e-6
    # relative for "ipopt", 1e-8 relative for "dual"); re-deriving a
    # *stricter* bound here would just reject genuinely-successful solves
    # for no reason, so this uses the more lenient of the two.
    constraint_violation = float(getattr(result, "constr_violation", 0.0) or 0.0)
    violation_limit = 1.0e-6 + 1.0e-6 * max(1.0, instance.C)
    success = (
        bool(result.success)
        and constraint_violation <= violation_limit
        and int(getattr(result, "status", -1)) in (1, 2)
    )
    q = None
    economic_objective = None
    if success:
        solution = t2018.extract_solution(formulation, result)
        q = tuple(float(value) for value in solution["q"])
        economic_objective = recompute_economic_objective(
            instance.a, instance.m, instance.b,
            instance.price_p, instance.cost_c, instance.salvage_s, q,
        )
    return SolveOutcome(
        solver_label=f"2018-NLP({method})",
        success=success,
        status_text=str(getattr(result, "message", "")),
        elapsed_seconds=elapsed,
        economic_objective=economic_objective,
        q=q,
    )


def solve_conic(instance: TableNewInstance, *, solver: str) -> SolveOutcome:
    import cvxpy as cp
    import triangular_current as tc

    bundle = tc.build_current_triangular_conic_continuous(
        instance.a, instance.m, instance.b,
        instance.price_p, instance.cost_c, instance.salvage_s, instance.C,
    )
    started = time.perf_counter()
    bundle.problem.solve(solver=solver)
    elapsed = time.perf_counter() - started

    success = bundle.problem.status == cp.OPTIMAL
    q = None
    economic_objective = None
    if success:
        q = tuple(float(value) for value in bundle.variables["q"].value)
        economic_objective = recompute_economic_objective(
            instance.a, instance.m, instance.b,
            instance.price_p, instance.cost_c, instance.salvage_s, q,
        )
    return SolveOutcome(
        solver_label=f"Conic({solver})",
        success=success,
        status_text=str(bundle.problem.status),
        elapsed_seconds=elapsed,
        economic_objective=economic_objective,
        q=q,
    )


@dataclasses.dataclass(slots=True)
class InstanceComparison:
    n: int
    replication: int
    outcome_2018: SolveOutcome
    outcome_conic: SolveOutcome
    both_succeeded: bool
    objective_absolute_difference: float | None
    objective_relative_difference: float | None
    objectives_match: bool | None


def compare_instance(
    instance: TableNewInstance,
    *,
    nlp_method: str,
    conic_solver: str,
    objective_abs_tol: float,
    objective_rel_tol: float,
) -> InstanceComparison:
    outcome_2018 = solve_2018(instance, method=nlp_method)
    outcome_conic = solve_conic(instance, solver=conic_solver)

    both_succeeded = outcome_2018.success and outcome_conic.success
    absolute = relative = None
    objectives_match = None
    if both_succeeded:
        left = outcome_2018.economic_objective
        right = outcome_conic.economic_objective
        absolute = abs(left - right)
        relative = absolute / max(1.0, abs(left), abs(right))
        limit = objective_abs_tol + objective_rel_tol * max(1.0, abs(left), abs(right))
        objectives_match = absolute <= limit

    return InstanceComparison(
        n=instance.n,
        replication=instance.replication,
        outcome_2018=outcome_2018,
        outcome_conic=outcome_conic,
        both_succeeded=both_succeeded,
        objective_absolute_difference=absolute,
        objective_relative_difference=relative,
        objectives_match=objectives_match,
    )


# ---------------------------------------------------------------------------
# Summary reporting.
# ---------------------------------------------------------------------------

def _stats(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    mean = statistics.fmean(values)
    maximum = max(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, maximum, stdev


def summarize(comparisons: Sequence[InstanceComparison]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_n: dict[int, list[InstanceComparison]] = {}
    for comparison in comparisons:
        by_n.setdefault(comparison.n, []).append(comparison)

    for n in sorted(by_n):
        group = by_n[n]
        times_2018 = [c.outcome_2018.elapsed_seconds for c in group]
        times_conic = [c.outcome_conic.elapsed_seconds for c in group]
        failures_2018 = sum(1 for c in group if not c.outcome_2018.success)
        failures_conic = sum(1 for c in group if not c.outcome_conic.success)
        joint = [c for c in group if c.both_succeeded]
        mismatches = sum(1 for c in joint if not c.objectives_match)
        max_rel_diff = max((c.objective_relative_difference for c in joint), default=None)

        mean_2018, max_2018, std_2018 = _stats(times_2018)
        mean_conic, max_conic, std_conic = _stats(times_conic)

        rows.append({
            "n": n,
            "runs": len(group),
            "2018_time_avg_sec": mean_2018,
            "2018_time_max_sec": max_2018,
            "2018_time_std_sec": std_2018,
            "2018_failed_runs": failures_2018,
            "conic_time_avg_sec": mean_conic,
            "conic_time_max_sec": max_conic,
            "conic_time_std_sec": std_conic,
            "conic_failed_runs": failures_conic,
            "both_succeeded_runs": len(joint),
            "objective_mismatch_runs": mismatches,
            "max_objective_relative_difference": max_rel_diff,
        })
    return rows


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    header = (
        f"{'n':>5} {'runs':>5} | "
        f"{'2018 avg':>10} {'2018 max':>10} {'2018 std':>10} {'2018 fail':>9} | "
        f"{'conic avg':>10} {'conic max':>10} {'conic std':>10} {'conic fail':>10} | "
        f"{'both ok':>8} {'mismatch':>8} {'max rel diff':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n']:>5} {row['runs']:>5} | "
            f"{row['2018_time_avg_sec']:>10.3f} {row['2018_time_max_sec']:>10.3f} "
            f"{row['2018_time_std_sec']:>10.3f} {row['2018_failed_runs']:>9} | "
            f"{row['conic_time_avg_sec']:>10.3f} {row['conic_time_max_sec']:>10.3f} "
            f"{row['conic_time_std_sec']:>10.3f} {row['conic_failed_runs']:>10} | "
            f"{row['both_succeeded_runs']:>8} {row['objective_mismatch_runs']:>8} "
            f"{'' if row['max_objective_relative_difference'] is None else f'{row['max_objective_relative_difference']:.3e}':>13}"
        )


def write_summary_txt(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "n,runs,2018_time_avg_sec,2018_time_max_sec,2018_time_std_sec,2018_failed_runs,"
        "conic_time_avg_sec,conic_time_max_sec,conic_time_std_sec,conic_failed_runs,"
        "both_succeeded_runs,objective_mismatch_runs,max_objective_relative_difference"
    ]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in (
            "n", "runs", "2018_time_avg_sec", "2018_time_max_sec", "2018_time_std_sec",
            "2018_failed_runs", "conic_time_avg_sec", "conic_time_max_sec", "conic_time_std_sec",
            "conic_failed_runs", "both_succeeded_runs", "objective_mismatch_runs",
            "max_objective_relative_difference",
        )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

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
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            _require(start <= stop, f"invalid descending {label} range: {token!r}")
            selected.extend(value for value in allowed if start <= value <= stop)
        else:
            selected.append(int(token))
    _require(bool(selected), f"no {label} values selected")
    invalid = sorted(set(selected) - allowed_set)
    _require(not invalid, f"unsupported {label} values: {invalid}; allowed={list(allowed)}")
    return tuple(sorted(set(selected)))


def run_self_tests() -> None:
    def test_generation_determinism() -> None:
        first = generate_2018_triangular_instance(10, 1)
        second = generate_2018_triangular_instance(10, 1)
        _require(first == second, "instance generation is not deterministic")

    def test_generation_discrimination() -> None:
        a = generate_2018_triangular_instance(10, 1)
        b = generate_2018_triangular_instance(10, 2)
        c = generate_2018_triangular_instance(20, 1)
        _require(a != b and a != c, "n/replication must change the generated instance")

    def test_txt_roundtrip(tmp_dir: Path) -> None:
        instance = generate_2018_triangular_instance(6, 3)
        path = tmp_dir / instance_filename(6, 3)
        write_instance_txt(path, instance)
        reloaded = read_instance_txt(path)
        _require(reloaded.n == instance.n, "roundtrip n mismatch")
        _require(reloaded.replication == instance.replication, "roundtrip replication mismatch")
        for field in ("a", "m", "b", "cost_c", "price_p", "salvage_s", "x_u"):
            original = getattr(instance, field)
            restored = getattr(reloaded, field)
            _require(
                all(abs(o - r) <= 1.0e-9 for o, r in zip(original, restored, strict=True)),
                f"roundtrip mismatch in field {field!r}",
            )
        _require(abs(reloaded.C - instance.C) <= 1.0e-9, "roundtrip C mismatch")

    def test_small_solve_and_compare(tmp_dir: Path) -> None:
        instance = generate_2018_triangular_instance(8, 1)
        comparison = compare_instance(
            instance,
            nlp_method=DEFAULT_NLP_METHOD,
            conic_solver=_first_available_conic_solver(),
            objective_abs_tol=DEFAULT_OBJECTIVE_ABS_TOL,
            objective_rel_tol=DEFAULT_OBJECTIVE_REL_TOL,
        )
        _require(comparison.outcome_2018.success, "2018 solve did not succeed on self-test instance")
        _require(comparison.outcome_conic.success, "conic solve did not succeed on self-test instance")
        _require(comparison.objectives_match, "objectives should match on a well-conditioned instance")

    tests: list[tuple[str, Any]] = [
        ("Instance generation is deterministic", test_generation_determinism),
        ("Instance generation discriminates on n/replication", test_generation_discrimination),
        ("Instance text roundtrip", None),
        ("Small end-to-end solve and compare", None),
    ]
    import tempfile
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        callables = [
            test_generation_determinism,
            test_generation_discrimination,
            lambda: test_txt_roundtrip(tmp_path),
            lambda: test_small_solve_and_compare(tmp_path),
        ]
        names = [
            "Instance generation is deterministic",
            "Instance generation discriminates on n/replication",
            "Instance text roundtrip",
            "Small end-to-end solve and compare",
        ]
        for index, (name, function) in enumerate(zip(names, callables), start=1):
            try:
                function()
                print(f"SELF-TEST {index:02d} PASS - {name}")
            except Exception as error:
                failures += 1
                print(f"SELF-TEST {index:02d} FAIL - {name}: {error}")
    print(f"Self-tests: {len(names) - failures} PASS, {failures} FAIL")
    if failures:
        raise TableNewError("one or more self-tests failed")


def _first_available_conic_solver() -> str:
    import cvxpy as cp

    installed = set(cp.installed_solvers())
    for candidate in (DEFAULT_CONIC_SOLVER, "MOSEK", "CPLEX", "CLARABEL", "SCS"):
        if candidate in installed:
            return candidate
    raise TableNewError("no usable CVXPY conic solver is installed")


def make_parser() -> argparse.ArgumentParser:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare the 2018 Triangular NLP model against the current "
        "conic/perspective Triangular model (zero fixed costs, no binary variables)."
    )
    parser.add_argument("--self-test", action="store_true", help="run tests on a tiny instance and exit")
    parser.add_argument("--n-values", default="all", help="all or comma/range selection of n in {100,...,500}")
    parser.add_argument("--replications", default="all", help="all or comma/range selection of replication in 1..30")
    parser.add_argument(
        "--nlp-method",
        default=DEFAULT_NLP_METHOD,
        choices=["ipopt", "trust-constr", "SLSQP", "dual"],
        help="how to solve the 2018 cubic model: 'ipopt' (default; Pyomo + IPOPT, "
        "the practical CONOPT substitute), a scipy.optimize.minimize method "
        "('trust-constr' or 'SLSQP', the most literal but impractically slow CONOPT "
        "stand-in), or 'dual' for the exact but non-NLP Lagrangian-bisection shortcut "
        "in triangular_2018.py (opt-in cross-check only, not the cubic model itself)",
    )
    parser.add_argument("--conic-solver", default=DEFAULT_CONIC_SOLVER, help="CVXPY conic solver for the current model")
    parser.add_argument("--objective-abs-tol", type=float, default=DEFAULT_OBJECTIVE_ABS_TOL)
    parser.add_argument("--objective-rel-tol", type=float, default=DEFAULT_OBJECTIVE_REL_TOL)
    parser.add_argument("--instances-dir", type=Path, default=script_directory / DEFAULT_INSTANCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=script_directory / DEFAULT_RESULTS_DIR)
    parser.add_argument("--regenerate", action="store_true", help="regenerate instance files even if they already exist")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_tests()
        print("TABLE_NEW_SELF_TESTS_PASS")
        return 0

    n_values = parse_int_selection(args.n_values, N_VALUES, "n")
    replications = parse_int_selection(args.replications, tuple(range(1, REPLICATIONS + 1)), "replication")

    import cvxpy as cp

    installed = set(cp.installed_solvers())
    _require(args.conic_solver in installed, f"CVXPY does not report {args.conic_solver!r} as installed: {sorted(installed)}")

    if args.nlp_method == "ipopt":
        import pyomo.environ as pyo

        _require(
            pyo.SolverFactory("ipopt").available(),
            "--nlp-method ipopt requires Pyomo with a working IPOPT installation "
            "(pyo.SolverFactory('ipopt').available() is False); install IPOPT or "
            "pass --nlp-method trust-constr/SLSQP/dual instead",
        )

    args.instances_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[InstanceComparison] = []
    total = len(n_values) * len(replications)
    done = 0
    for n in n_values:
        for replication in replications:
            instance = load_or_generate_instance(
                args.instances_dir, n, replication, regenerate=args.regenerate
            )
            comparison = compare_instance(
                instance,
                nlp_method=args.nlp_method,
                conic_solver=args.conic_solver,
                objective_abs_tol=args.objective_abs_tol,
                objective_rel_tol=args.objective_rel_tol,
            )
            comparisons.append(comparison)
            done += 1
            match_text = (
                "n/a" if comparison.objectives_match is None
                else ("MATCH" if comparison.objectives_match else "MISMATCH")
            )
            print(
                f"[{done}/{total}] n={n} rep={replication}: "
                f"2018={'OK' if comparison.outcome_2018.success else 'FAIL'} "
                f"({comparison.outcome_2018.elapsed_seconds:.2f}s), "
                f"conic={'OK' if comparison.outcome_conic.success else 'FAIL'} "
                f"({comparison.outcome_conic.elapsed_seconds:.2f}s), objective={match_text}",
                flush=True,
            )

    rows = summarize(comparisons)
    print()
    print_summary(rows)
    summary_path = args.output_dir / "table_new_summary.csv"
    write_summary_txt(summary_path, rows)
    print(f"\nSummary written to {summary_path}")
    print("TABLE_NEW_RESULTS_READY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TableNewError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
