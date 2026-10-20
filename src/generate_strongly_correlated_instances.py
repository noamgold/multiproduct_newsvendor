"""Generate a harder-fixed-cost variant of the frozen Triangular Table 2
instances (triangular_table2_instances.csv), for benchmarking solver
difficulty beyond what the manuscript's own instance set exercises.

Background. The manuscript (Newsvendor_current.pdf, Section 8.1, equation
(13)) builds each product's fixed cost as

    k_i = pi_tilde_i + max_{l in N}(pi_tilde_l) / 10

where pi_tilde_i is fixed_cost_score (product i's own unconstrained,
relaxed profit) -- i.e. k_i tracks product i's own profitability, offset by
a constant shared shift. The manuscript cites this explicitly as following
"a correlated random knapsack instance generation scheme in Pisinger
(2004) ... commonly used to generate knapsack instances that are difficult
to solve in practice." generate_manuscript_table_instances.py implements
(13) verbatim (see its score_shift = max(scores) / 10), and
triangular_table2_instances.csv is that frozen, already-generated dataset.

An empirical A/B probe (strongly_correlated_fixed_cost_variant.py's
predecessor) found that GUROBI's B&B node counts on these instances are
highly sensitive to the size of that additive shift: decoupling k_i from
score_i entirely collapsed every instance to ~1 node (trivial), while
SHRINKING the shift toward the smallest value that keeps every k_i >= 0
(instead of the manuscript's max(score)/10) increased average node counts
3-5x and worst-case node counts up to 14x on n=100/200 test batches --
consistent with Pisinger's finding that "strongly correlated" knapsack
instances (weight = value + a SMALL constant) are the hardest B&B regime,
and that the manuscript's specific constant is a working choice rather than
the theoretical extreme.

This module generates a full n=100..700, 30-replications-per-n sibling of
triangular_table2_instances.csv, identical in every economic parameter
(a, m, b, cost_c, price_p, salvage_s, x_u, fixed_cost_score, kappa,
capacity_fraction, C, master_seed, instance_seed, price_markup_draw,
salvage_multiplier) EXCEPT that k_i (and therefore K, via the same
K = kappa/101 * sum(k) formula) is rebuilt with the minimal
non-negativity-preserving shift instead of max(score)/10.

It does NOT modify triangular_table2_instances.csv or
generate_manuscript_table_instances.py.

Usage:

    python generate_strongly_correlated_instances.py --n-values all --replications all
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_triangular_table2_experiment as rte

STRONGLY_CORRELATED_VERSION = "STRONGLY_CORRELATED_K_V1"
BASE_GENERATION_VERSION = rte.EXPECTED_GENERATION_VERSION
VARIANT_GENERATION_VERSION = f"{STRONGLY_CORRELATED_VERSION}+{BASE_GENERATION_VERSION}"

# Tiny float-safety margin added on top of the minimal shift, so k_i is
# strictly nonnegative even after floating-point subtraction, not merely
# nonnegative in exact arithmetic.
MINIMAL_SHIFT_EPSILON = 1.0e-9

DEFAULT_INPUT_CSV = "triangular_table2_instances.csv"
DEFAULT_OUTPUT_CSV = "triangular_table2_instances_strongly_correlated.csv"


def minimal_nonnegative_shift(scores: Sequence[float]) -> float:
    """The smallest additive shift that keeps every score + shift >= 0."""
    return max(0.0, -min(scores)) + MINIMAL_SHIFT_EPSILON


def rebuild_strongly_correlated_k(
    scores: Sequence[float], kappa: int
) -> tuple[tuple[float, ...], float]:
    """Rebuild (k, K) with k_i = fixed_cost_score_i + minimal_shift, and K
    via the manuscript's own K = kappa/101 * sum(k) formula (equation just
    below (13)). Every product's k differs from its own profitability by
    the exact same constant (Pisinger's "strongly correlated" w_i = v_i +
    delta knapsack structure), with delta as small as feasibility allows
    rather than the manuscript's max(score)/10.
    """
    delta = minimal_nonnegative_shift(scores)
    k = tuple(score + delta for score in scores)
    K = kappa / 101.0 * sum(k)
    return k, K


def _rows_for_group(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Validate one instance's rows against the standard (manuscript)
    reader, then re-derive k/K with the strongly-correlated construction,
    returning full output rows in the same column layout as the input CSV.
    """
    instance = rte.parse_triangular_table2_instance(rows)
    new_k, new_K = rebuild_strongly_correlated_k(instance.fixed_cost_score, instance.kappa)

    output_rows = []
    for row, k_i in zip(rows, new_k, strict=True):
        new_row = dict(row)
        new_row["k"] = repr(k_i)
        new_row["K"] = repr(new_K)
        new_row["generation_version"] = VARIANT_GENERATION_VERSION
        output_rows.append(new_row)
    return output_rows


def generate_strongly_correlated_csv(
    input_path: Path,
    output_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> int:
    """Read the frozen Triangular Table 2 CSV, rebuild k/K for the selected
    (n, replication) instances with the strongly-correlated construction,
    and write them to ``output_path`` in the same column layout, tagged
    with VARIANT_GENERATION_VERSION. Returns the number of instances
    written.
    """
    requested = {(n, r) for n in n_values for r in replications}
    written = 0
    with input_path.open("r", encoding="utf-8", newline="") as in_stream:
        reader = csv.DictReader(in_stream)
        fieldnames = reader.fieldnames
        rte.require(
            tuple(fieldnames or ()) == rte.EXPECTED_CSV_HEADER,
            "triangular_table2_instances.csv has an unexpected column layout",
        )
        with output_path.open("w", encoding="utf-8", newline="") as out_stream:
            writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
            writer.writeheader()
            for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
                rows = list(group)
                key = (rte.parse_int(rows[0], "n"), rte.parse_int(rows[0], "replication"))
                if key not in requested:
                    continue
                writer.writerows(_rows_for_group(rows))
                written += 1
    return written


def read_selected_strongly_correlated_instances(
    csv_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> list[rte.TriangularInstance]:
    """Load instances from a strongly-correlated variant CSV (as produced
    by generate_strongly_correlated_csv), validating the same economic
    invariants as the manuscript reader EXCEPT the k formula, which is
    checked against rebuild_strongly_correlated_k instead of equation (13).
    """
    requested = {(n, r) for n in n_values for r in replications}
    observed: set[tuple[int, int]] = set()
    results: list[rte.TriangularInstance] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rte.require(
            tuple(reader.fieldnames or ()) == rte.EXPECTED_CSV_HEADER,
            "strongly-correlated instances CSV has an unexpected column layout",
        )
        for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
            rows = list(group)
            key = (rte.parse_int(rows[0], "n"), rte.parse_int(rows[0], "replication"))
            if key not in requested:
                continue
            rte.require(key not in observed, f"duplicate selected instance group: {key}")
            instance = _parse_strongly_correlated_instance(rows)
            observed.add(key)
            results.append(instance)
    missing = sorted(requested - observed)
    rte.require(not missing, f"strongly-correlated CSV is missing selected instances: {missing}")
    return results


def _parse_strongly_correlated_instance(rows: Sequence[Mapping[str, str]]) -> rte.TriangularInstance:
    """Mirrors run_triangular_table2_experiment.parse_triangular_table2_instance
    exactly, EXCEPT the k/K construction check, which validates against
    rebuild_strongly_correlated_k (minimal shift) instead of the base
    parser's hard-coded equation (13) (k_i = score_i + max(score)/10). Every
    other invariant -- economic domain, x_u/fixed_cost_score formulas,
    capacity formula, scalar-field consistency -- is unchanged.
    """
    rte.require(bool(rows), "cannot build an instance from zero rows")
    first = rows[0]
    n = rte.parse_int(first, "n")
    replication = rte.parse_int(first, "replication")
    instance_id = first["instance_id"]
    rte.require(len(rows) == n, f"{instance_id}: expected {n} rows, observed {len(rows)}")
    rte.require(
        [rte.parse_int(row, "product_id") for row in rows] == list(range(1, n + 1)),
        f"{instance_id}: product_id sequence is not exactly 1..n",
    )

    scalar_fields = (
        "family", "n", "replication", "instance_id", "master_seed",
        "instance_seed", "price_markup_draw", "kappa", "K",
        "capacity_fraction", "C", "generation_version",
    )
    for field in scalar_fields:
        rte._scalar_field(rows, field)

    rte.require(first["family"] == rte.EXPECTED_FAMILY, f"{instance_id}: family is not TRIANGULAR")
    expected_id = f"TRIANGULAR-N{n:04d}-R{replication:03d}"
    rte.require(instance_id == expected_id, f"unexpected instance_id {instance_id!r}")
    rte.require(
        first["generation_version"] == VARIANT_GENERATION_VERSION,
        f"{instance_id}: unexpected generation version {first['generation_version']!r}; "
        f"expected {VARIANT_GENERATION_VERSION!r}",
    )
    a = tuple(rte.parse_float(row, "a") for row in rows)
    m = tuple(rte.parse_float(row, "m") for row in rows)
    b = tuple(rte.parse_float(row, "b") for row in rows)
    cost = tuple(rte.parse_float(row, "cost_c") for row in rows)
    price = tuple(rte.parse_float(row, "price_p") for row in rows)
    salvage = tuple(rte.parse_float(row, "salvage_s") for row in rows)
    x_u = tuple(rte.parse_float(row, "x_u") for row in rows)
    scores = tuple(rte.parse_float(row, "fixed_cost_score") for row in rows)
    k = tuple(rte.parse_float(row, "k") for row in rows)
    kappa = rte.parse_int(first, "kappa")
    budget_K = rte.parse_float(first, "K")
    capacity_fraction = rte.parse_float(first, "capacity_fraction")
    capacity_C = rte.parse_float(first, "C")

    rte.require(n in rte.TABLE2_N_VALUES, f"{instance_id}: unsupported Table 2 n={n}")
    rte.require(replication in rte.TABLE2_REPLICATIONS, f"{instance_id}: invalid replication")
    rte.require(kappa == rte.EXPECTED_KAPPA, f"{instance_id}: kappa must equal 50")
    rte.require(1.0 / 5.0 <= capacity_fraction <= 1.0 / 3.0, f"{instance_id}: bad C fraction")

    for index, values in enumerate(zip(a, m, b, cost, price, salvage, x_u, scores), start=1):
        ai, mi, bi, ci, pi, si, xui, score = values
        rte.require(0.0 <= ai < mi < bi, f"{instance_id} product {index}: invalid a<m<b")
        rte.require(pi > ci, f"{instance_id} product {index}: price must exceed cost")
        rte.require(pi > si, f"{instance_id} product {index}: price must exceed salvage")
        expected_xu = rte.triangular_unconstrained_quantity(ai, mi, bi, pi, ci, si)
        rte.require(
            rte.close_enough(xui, expected_xu),
            f"{instance_id} product {index}: x_u formula mismatch",
        )
        expected_score = rte.triangular_expected_profit(xui, ai, mi, bi, pi, ci, si)
        rte.require(
            rte.close_enough(score, expected_score),
            f"{instance_id} product {index}: fixed_cost_score mismatch",
        )

    expected_k, expected_K = rebuild_strongly_correlated_k(scores, kappa)
    for index, (observed, expected) in enumerate(zip(k, expected_k), start=1):
        rte.require(
            rte.close_enough(observed, expected),
            f"{instance_id} product {index}: strongly-correlated k formula mismatch",
        )
    rte.require(
        rte.close_enough(budget_K, expected_K),
        f"{instance_id}: strongly-correlated K formula mismatch",
    )
    rte.require(
        rte.close_enough(capacity_C, capacity_fraction * sum(x_u)),
        f"{instance_id}: C formula mismatch",
    )

    return rte.TriangularInstance(
        n=n,
        replication=replication,
        instance_id=instance_id,
        master_seed=rte.parse_int(first, "master_seed"),
        instance_seed=rte.parse_int(first, "instance_seed"),
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


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="all")
    parser.add_argument("--replications", default="all")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    n_values = rte.parse_int_selection(args.n_values, rte.TABLE2_N_VALUES, "n")
    replications = rte.parse_int_selection(args.replications, rte.TABLE2_REPLICATIONS, "replication")
    base = Path(__file__).parent
    written = generate_strongly_correlated_csv(
        base / args.input_csv, base / args.output_csv, n_values, replications
    )
    print(f"Wrote {written} strongly-correlated instances to {base / args.output_csv}")
    # Round-trip sanity check: read back and validate every written instance.
    read_back = read_selected_strongly_correlated_instances(
        base / args.output_csv, n_values, replications
    )
    print(f"Round-trip validated {len(read_back)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
