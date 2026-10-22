"""The Distribution-Free (Table 3) model layer: instance dataclass, CSV
reader/validator, the closed-form economics, and the two CVXPY MISOCP
builders for the manuscript's formulations (11) and (12).

This is the Distribution-Free analogue of the model/parse layer inside
run_uniform_table1_experiment.py (Table 1) and
run_triangular_table2_experiment.py (Table 2). Those two scripts also
carry a large official-run repair/validation/logging apparatus; this
module deliberately carries only the model and the reader, because the
experiment that uses it (run_distribution_free_strongly_correlated.py) is
an exploratory probe in the same spirit as the other two
*_strongly_correlated.py runners.

The model (Newsvendor_current.pdf, Section 7 and equations (11)-(12))
-------------------------------------------------------------------
Demand for product i is known only through its mean mu_i and standard
deviation sigma_i. Scarf's bound gives the worst-case expected shortfall
in closed form, and the manuscript writes the resulting problem as a
*cost minimization* (unlike Tables 1 and 2, which maximize profit):

    theta_i(x) = (c_i - s_i) x
                 + (p_i - s_i) [ sqrt(sigma_i^2 + (x - mu_i)^2) - x + mu_i ] / 2
    g_i(x)     = theta_i(x) + lambda (p_i - s_i)^2 sigma_i^2 x^2

with the lambda term a variance penalty (lambda = 1e-9 in the Table 3
instances). g_i is strictly convex, and x^U_i is its unconstrained
minimizer. with_portfolio_variance_bound switches an instance to the
portfolio variance bound, weights (p_i - s_i)^2 sigma_i^2 / mu_i^2. Formulation (11), big-M:

    min  sum_i theta_i + lambda sum_i (p_i - s_i)^2 sigma_i^2 x_i^2   (11a)
    s.t. (c_i-s_i)x_i + (p_i-s_i)(nu_i - x_i + mu_i)/2 <= theta_i     (11b')
         ||(sigma_i, x_i - mu_i)||_2 <= nu_i                          (11b'')
         x_i <= x^U_i z_i                                             (11c)
         sum_i x_i <= C,  sum_i k_i z_i <= K,  x,theta >= 0, z binary (2d-2f)

where (11b) has been split into the pair (11b')/(11b'') exactly as the
manuscript prescribes, to make it a second-order cone program. The
perspective reformulation (12) keeps (11b)-(11d) verbatim and changes
only how the quadratic penalty is lifted: the objective uses u_i in place
of x_i^2, under the perspective cone x_i^2 <= u_i z_i. This module
implements the big-M variant as the same lifted u_i under the plain
rotated cone x_i^2 <= u_i * 1, so the two formulations differ in exactly
one coefficient -- which is what (12) actually says, and it matches how
run_uniform_table1_experiment.py handles the analogous pair (4)/(6).

Profit vs. cost, and the constant offset
----------------------------------------
Not stocking product i still costs g_i(0) > 0 in (11a) -- the worst-case
shortfall cost of having none of it. So the cost objective and the
profit objective used everywhere else in this project differ by a
per-instance constant:

    pi_i(x) = g_i(0) - g_i(x),       pi_i(0) = 0, pi_i concave
    sum_i g_i(x_i) = sum_i g_i(0) - sum_i pi_i(x_i)

Both directions are provided (distribution_free_cost /
distribution_free_profit). The fixed_cost_score column of the Table 3
instances is pi_i(x^U_i) = g_i(0) - g_i(x^U_i), i.e. the profit form; see
generate_manuscript_table_instances.py line ~833. pi_i satisfies
Assumption 1 of submodularity_1.pdf (continuous, concave, pi_i(0) = 0),
which is what lets submodular_cuts.py solve this family too.

Beware when comparing objective values across methods: the cost scale
carries the whole sum_i g_i(0) constant (tens of thousands on these
instances) while the decisions only move the profit part (thousands), so
a relative tolerance means something much weaker on the cost scale than
on the profit scale. run_distribution_free_strongly_correlated.py reports
both and says which one its PASS/FAIL test uses.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import generate_manuscript_table_instances as gmti
import run_triangular_table2_experiment as rte

# Distribution-independent utilities, re-exported rather than re-written
# so this module, Table 1's and Table 2's all apply the identical
# tolerance/parsing/solver-introspection rules.
require = rte.require
close_enough = rte.close_enough
scaled_tolerance = rte.scaled_tolerance
relative_difference = rte.relative_difference
parse_int_selection = rte.parse_int_selection
finite_vector = rte.finite_vector
extract_native_stats = rte.extract_native_stats
import_optimization_stack = rte.import_optimization_stack

BIGM_MODEL_ID = "CURRENT_DF_BIGM_FIXED_COST"
PERSPECTIVE_MODEL_ID = "CURRENT_DF_PERSPECTIVE_FIXED_COST"
MODEL_LABELS = {
    BIGM_MODEL_ID: "Big-M (11)",
    PERSPECTIVE_MODEL_ID: "Conic (12)",
}

EXPECTED_FAMILY = "DISTRIBUTION_FREE"
EXPECTED_GENERATION_VERSION = gmti.GENERATION_VERSION
EXPECTED_KAPPA = gmti.KAPPA
EXPECTED_CSV_HEADER = gmti.CSV_HEADERS["DISTRIBUTION_FREE"]
DEFAULT_INSTANCES_CSV = gmti.CSV_FILENAMES["DISTRIBUTION_FREE"]

TABLE3_N_VALUES = (100, 200, 300, 400, 500, 600, 700)
TABLE3_REPLICATIONS = tuple(range(1, 31))


class DistributionFreeError(RuntimeError):
    """Raised when Table 3 input data or model configuration is invalid."""


@dataclasses.dataclass(frozen=True, slots=True)
class DistributionFreeInstance:
    """One Table 3 instance, in the same shape as UniformInstance and
    TriangularInstance so the runners stay interchangeable.
    """

    n: int
    replication: int
    instance_id: str
    master_seed: int
    instance_seed: int
    mu: tuple[float, ...]
    sigma: tuple[float, ...]
    lambda_: float
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
    # Per-product penalty weights w_i in lambda * w_i * x_i^2. None means the
    # stored instances' weights (p_i - s_i)^2 sigma_i^2; see
    # with_portfolio_variance_bound for the portfolio weights.
    variance_weight: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DistributionFreeFormulation:
    problem: Any
    variables: dict[str, Any]
    model_id: str


def df_g(
    x: float, mu: float, sigma: float, lambda_: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    """g_i(x) -- the per-product cost minimized by (11a). Delegated to the
    instance generator so the experiment can never drift from the
    definition the data was built with.
    """
    return gmti.df_g(x, mu, sigma, lambda_, price_p, cost_c, salvage_s)


def df_profit(
    x: float, mu: float, sigma: float, lambda_: float, price_p: float, cost_c: float, salvage_s: float
) -> float:
    """pi_i(x) = g_i(0) - g_i(x): the same model written as a profit, with
    pi_i(0) = 0. This is the form the instances' fixed_cost_score column
    holds (at x = x^U) and the form the submodular cuts need.
    """
    return df_g(0.0, mu, sigma, lambda_, price_p, cost_c, salvage_s) - df_g(
        x, mu, sigma, lambda_, price_p, cost_c, salvage_s
    )


def penalty_weights(instance: DistributionFreeInstance) -> tuple[float, ...]:
    """w_i in the penalty lambda * w_i * x_i^2 of (11a)/(12)."""
    if instance.variance_weight is not None:
        return instance.variance_weight
    return tuple(
        (p - s) ** 2 * sigma**2
        for p, s, sigma in zip(instance.price_p, instance.salvage_s, instance.sigma, strict=True)
    )


def _product_costs(instance: DistributionFreeInstance, q: Sequence[float]) -> list[float]:
    """g_i(q_i) = theta_i(q_i) + lambda * w_i * q_i^2 for every product.
    With the stored weights this is exactly the generator's df_g.
    """
    return [
        gmti.df_theta(quantity, mu, sigma, p, c, s) + instance.lambda_ * weight * quantity**2
        for quantity, mu, sigma, p, c, s, weight in zip(
            q, instance.mu, instance.sigma, instance.price_p, instance.cost_c,
            instance.salvage_s, penalty_weights(instance), strict=True,
        )
    ]


def distribution_free_cost(instance: DistributionFreeInstance, q: Sequence[float]) -> float:
    """sum_i g_i(q_i): the objective (11a)/(12) reports, recomputed
    independently from the order quantities alone.
    """
    return sum(_product_costs(instance, q))


def distribution_free_profit(instance: DistributionFreeInstance, q: Sequence[float]) -> float:
    """sum_i pi_i(q_i) = zero_quantity_cost(instance) - distribution_free_cost."""
    return sum(
        at_zero - at_q
        for at_zero, at_q in zip(
            _product_costs(instance, [0.0] * instance.n), _product_costs(instance, q), strict=True
        )
    )


def zero_quantity_cost(instance: DistributionFreeInstance) -> float:
    """sum_i g_i(0) -- the constant separating the cost and profit scales."""
    return sum(_product_costs(instance, [0.0] * instance.n))


def with_portfolio_variance_bound(
    instance: DistributionFreeInstance, lambda_: float
) -> DistributionFreeInstance:
    """The same instance under the portfolio variance bound: penalty weights
    w_i = (p_i - s_i)^2 sigma_i^2 / mu_i^2 and penalty parameter ``lambda_``.

    Only x^U depends on the penalty, so it is recomputed (it is the big-M
    constant of (11c)); mu, sigma, prices, k, K and C are kept as stored.
    k and fixed_cost_score therefore remain the profit scores of the stored
    model, whose penalty (lambda = 1e-9) is negligible.
    """
    require(lambda_ >= 0.0, "lambda must be nonnegative")
    require(all(mu > 0.0 for mu in instance.mu), f"{instance.instance_id}: mu must be positive")
    weights = tuple(
        (p - s) ** 2 * sigma**2 / mu**2
        for p, s, sigma, mu in zip(
            instance.price_p, instance.salvage_s, instance.sigma, instance.mu, strict=True
        )
    )
    x_u = []
    for mu, sigma, p, c, s, weight in zip(
        instance.mu, instance.sigma, instance.price_p, instance.cost_c,
        instance.salvage_s, weights, strict=True,
    ):
        def derivative(x: float) -> float:
            return (c - s) + (p - s) / 2.0 * ((x - mu) / math.hypot(sigma, x - mu) - 1.0) + 2.0 * lambda_ * weight * x

        if derivative(0.0) >= 0.0:
            x_u.append(0.0)
            continue
        # g_i' is strictly increasing, so bisect on a bracket found by doubling.
        low, high = 0.0, mu + sigma
        while derivative(high) < 0.0:
            high *= 2.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if derivative(middle) < 0.0:
                low = middle
            else:
                high = middle
        x_u.append(0.5 * (low + high))
    return dataclasses.replace(
        instance, lambda_=float(lambda_), variance_weight=weights, x_u=tuple(x_u)
    )


def _scalar_field(rows: Sequence[Mapping[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    require(len(values) == 1, f"instance rows disagree on scalar field {field!r}")
    return next(iter(values))


def parse_distribution_free_table3_instance(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_generation_version: str = EXPECTED_GENERATION_VERSION,
    validate_fixed_costs: Callable[[DistributionFreeInstance], None] | None = None,
) -> DistributionFreeInstance:
    """Parse and fully validate one Table 3 instance group.

    Every economic invariant -- the mu/sigma draw ranges, lambda, the
    price/salvage relationships, that x^U is genuinely the minimizer of
    g (a KKT + local-perturbation check), the fixed_cost_score formula,
    the C formula, and the Distribution-Free candidate-rejection
    bookkeeping -- is checked by calling the generator's OWN authoritative
    validator, gmti.validate_tables_1_to_3_instance, rather than
    reimplementing it here.

    That validator also insists on the manuscript's equation-(13) fixed
    costs and the base generation_version, neither of which holds for the
    strongly-correlated variant. So the k/K/generation_version triple is
    swapped back to its base-equivalent form purely for the duration of
    that call, and the file's actual fixed costs are then checked by
    ``validate_fixed_costs`` (default: equation (13) itself).
    """
    require(bool(rows), "cannot build an instance from zero rows")
    first = rows[0]
    n = gmti._parse_int(first, "n")
    replication = gmti._parse_int(first, "replication")
    instance_id = first["instance_id"]
    require(len(rows) == n, f"{instance_id}: expected {n} rows, observed {len(rows)}")
    require(
        [gmti._parse_int(row, "product_id") for row in rows] == list(range(1, n + 1)),
        f"{instance_id}: product_id sequence is not exactly 1..n",
    )
    for field in (
        "family", "n", "replication", "instance_id", "master_seed", "instance_seed",
        "price_markup_draw", "kappa", "K", "capacity_fraction", "C",
        "candidate_attempt", "candidate_rejection_count", "accepted_max_df_score",
        "rejection_records", "generation_version",
    ):
        _scalar_field(rows, field)

    require(first["family"] == EXPECTED_FAMILY, f"{instance_id}: family is not DISTRIBUTION_FREE")
    require(
        instance_id == gmti.stable_instance_id(EXPECTED_FAMILY, n, replication),
        f"unexpected instance_id {instance_id!r}",
    )
    require(
        first["generation_version"] == expected_generation_version,
        f"{instance_id}: unexpected generation version {first['generation_version']!r}; "
        f"expected {expected_generation_version!r}",
    )
    require(n in TABLE3_N_VALUES, f"{instance_id}: unsupported Table 3 n={n}")
    require(replication in TABLE3_REPLICATIONS, f"{instance_id}: invalid replication")

    parsed = gmti._instance_from_csv_group(EXPECTED_FAMILY, list(rows))

    # Hand the generator's validator a base-equivalent copy: same
    # economics, but with equation (13)'s fixed costs and the base version
    # tag, which are the only two things it checks that a variant file is
    # entitled to differ on.
    scores = [product["fixed_cost_score"] for product in parsed["products"]]
    base_shift = max(scores) / 10.0
    base_products = [dict(product, k=score + base_shift) for product, score in zip(parsed["products"], scores)]
    base_instance = dict(
        parsed,
        products=base_products,
        K=gmti.KAPPA / 101.0 * sum(product["k"] for product in base_products),
        generation_version=gmti.GENERATION_VERSION,
    )
    gmti.validate_tables_1_to_3_instance(base_instance)

    lambda_values = {product["lambda"] for product in parsed["products"]}
    require(len(lambda_values) == 1, f"{instance_id}: products disagree on lambda")

    instance = DistributionFreeInstance(
        n=n,
        replication=replication,
        instance_id=instance_id,
        master_seed=parsed["master_seed"],
        instance_seed=parsed["instance_seed"],
        mu=tuple(product["mu"] for product in parsed["products"]),
        sigma=tuple(product["sigma"] for product in parsed["products"]),
        lambda_=next(iter(lambda_values)),
        cost_c=tuple(product["cost_c"] for product in parsed["products"]),
        price_p=tuple(product["price_p"] for product in parsed["products"]),
        salvage_s=tuple(product["salvage_s"] for product in parsed["products"]),
        x_u=tuple(product["x_u"] for product in parsed["products"]),
        fixed_cost_score=tuple(scores),
        k=tuple(product["k"] for product in parsed["products"]),
        kappa=parsed["kappa"],
        K=parsed["K"],
        capacity_fraction=parsed["capacity_fraction"],
        C=parsed["C"],
        generation_version=first["generation_version"],
    )

    if validate_fixed_costs is None:
        validate_fixed_costs = _validate_manuscript_fixed_costs
    validate_fixed_costs(instance)
    return instance


def _validate_manuscript_fixed_costs(instance: DistributionFreeInstance) -> None:
    """Equation (13): k_i = score_i + max_l(score_l)/10, K = kappa/101 sum k."""
    shift = max(instance.fixed_cost_score) / 10.0
    for index, (observed, score) in enumerate(zip(instance.k, instance.fixed_cost_score), start=1):
        require(
            close_enough(observed, score + shift),
            f"{instance.instance_id} product {index}: k formula mismatch",
        )
    require(
        close_enough(instance.K, instance.kappa / 101.0 * sum(instance.k)),
        f"{instance.instance_id}: K formula mismatch",
    )


def read_selected_distribution_free_table3_instances(
    csv_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
    *,
    expected_generation_version: str = EXPECTED_GENERATION_VERSION,
    validate_fixed_costs: Callable[[DistributionFreeInstance], None] | None = None,
) -> Iterator[DistributionFreeInstance]:
    requested = {(n, replication) for n in n_values for replication in replications}
    observed: set[tuple[int, int]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            tuple(reader.fieldnames or ()) == EXPECTED_CSV_HEADER,
            f"{csv_path.name} has an unexpected column layout",
        )
        for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
            rows = list(group)
            key = (gmti._parse_int(rows[0], "n"), gmti._parse_int(rows[0], "replication"))
            if key not in requested:
                continue
            require(key not in observed, f"duplicate selected instance group: {key}")
            observed.add(key)
            yield parse_distribution_free_table3_instance(
                rows,
                expected_generation_version=expected_generation_version,
                validate_fixed_costs=validate_fixed_costs,
            )
    missing = sorted(requested - observed)
    require(not missing, f"{csv_path.name} is missing selected instances: {missing}")


def build_current_manuscript_distribution_free_model(
    instance: DistributionFreeInstance, model_id: str
) -> DistributionFreeFormulation:
    """Formulation (11) (big-M) or (12) (perspective), as a CVXPY MISOCP.

    The two differ in exactly one place: the rotated cone lifting the
    quadratic penalty is x_i^2 <= u_i * 1 for (11) and x_i^2 <= u_i * z_i
    for (12).
    """
    cp, np, _ = import_optimization_stack()
    require(model_id in MODEL_LABELS, f"unsupported model_id: {model_id}")

    mu = np.asarray(instance.mu, dtype=float)
    sigma = np.asarray(instance.sigma, dtype=float)
    cost = np.asarray(instance.cost_c, dtype=float)
    price = np.asarray(instance.price_p, dtype=float)
    salvage = np.asarray(instance.salvage_s, dtype=float)
    x_u = np.asarray(instance.x_u, dtype=float)
    k = np.asarray(instance.k, dtype=float)
    spread = price - salvage

    x = cp.Variable(instance.n, nonneg=True, name="x")
    theta = cp.Variable(instance.n, nonneg=True, name="theta")
    nu = cp.Variable(instance.n, nonneg=True, name="nu")
    u = cp.Variable(instance.n, nonneg=True, name="u")
    z = cp.Variable(instance.n, boolean=True, name="z")

    constraints: list[Any] = [
        # (11b'), the linear half of (11b) once sqrt(...) is named nu_i.
        cp.multiply(cost - salvage, x) + cp.multiply(spread, nu - x + mu) / 2.0 <= theta,
        x <= cp.multiply(x_u, z),                       # (11c)
        cp.sum(x) <= instance.C,                        # (2d)
        k @ z <= instance.K,                            # (2e)
    ]

    for i in range(instance.n):
        # (11b''): ||(sigma_i, x_i - mu_i)||_2 <= nu_i, the Scarf square root.
        constraints.append(cp.SOC(nu[i], cp.hstack([sigma[i], x[i] - mu[i]])))
        # x_i^2 <= u_i * multiplier, via ||(2x, u-m)||_2 <= u+m.
        multiplier = 1.0 if model_id == BIGM_MODEL_ID else z[i]
        constraints.append(
            cp.SOC(u[i] + multiplier, cp.hstack([2.0 * x[i], u[i] - multiplier]))
        )

    penalty = instance.lambda_ * np.asarray(penalty_weights(instance), dtype=float)
    objective_expression = cp.sum(theta) + penalty @ u
    problem = cp.Problem(cp.Minimize(objective_expression), constraints)

    require(problem.is_dcp(), f"{model_id}: CVXPY model is not DCP")
    require(problem.is_mixed_integer(), f"{model_id}: CVXPY model is not mixed-integer")
    return DistributionFreeFormulation(
        problem=problem,
        variables={"x": x, "theta": theta, "nu": nu, "u": u, "z": z},
        model_id=model_id,
    )


def _self_test() -> None:
    """Read one real instance and check the model layer against the data."""
    base = Path(__file__).parent
    instances = list(
        read_selected_distribution_free_table3_instances(
            base / DEFAULT_INSTANCES_CSV, (100,), (1, 2)
        )
    )
    assert len(instances) == 2, f"expected 2 instances, read {len(instances)}"
    for instance in instances:
        # fixed_cost_score is pi_i(x^U_i) product by product.
        for index in range(instance.n):
            expected = df_profit(
                instance.x_u[index], instance.mu[index], instance.sigma[index],
                instance.lambda_, instance.price_p[index], instance.cost_c[index],
                instance.salvage_s[index],
            )
            assert close_enough(instance.fixed_cost_score[index], expected), (
                f"{instance.instance_id} product {index + 1}: profit form mismatch"
            )
        # The cost and profit scales differ by exactly sum_i g_i(0).
        quantities = [0.4 * value for value in instance.x_u]
        cost = distribution_free_cost(instance, quantities)
        profit = distribution_free_profit(instance, quantities)
        assert close_enough(cost + profit, zero_quantity_cost(instance)), (
            f"{instance.instance_id}: cost/profit offset is not sum_i g_i(0)"
        )
        # x^U minimizes g, so no perturbation of it can lower the cost.
        at_optimum = distribution_free_cost(instance, instance.x_u)
        for scale in (0.95, 1.05):
            perturbed = distribution_free_cost(instance, [scale * v for v in instance.x_u])
            assert perturbed >= at_optimum - 1.0e-9, (
                f"{instance.instance_id}: x_u is not the unconstrained minimizer"
            )
        print(
            f"  {instance.instance_id}: n={instance.n} "
            f"sum g(0)={zero_quantity_cost(instance):.3f} "
            f"sum score={sum(instance.fixed_cost_score):.3f} C={instance.C:.2f}"
        )
    print("DISTRIBUTION_FREE_TABLE3_SELF_TEST_PASS")


if __name__ == "__main__":
    _self_test()
