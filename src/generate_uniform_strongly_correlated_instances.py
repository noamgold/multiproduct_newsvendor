"""Generate a harder-fixed-cost variant of the frozen Uniform Table 1
instances (uniform_table1_instances.csv) -- the Uniform sibling of
generate_strongly_correlated_instances.py.

See that module's docstring for the full background. In brief: the
manuscript (Newsvendor_current.pdf, equation (13)) sets

    k_i = pi_tilde_i + max_{l in N}(pi_tilde_l) / 10

where pi_tilde_i is fixed_cost_score (product i's own unconstrained,
relaxed expected profit), citing Pisinger (2004)'s correlated random
knapsack scheme. Pisinger's hardest regime is the "strongly correlated"
one, weight = value + a SMALL constant; shrinking that additive shift to
the smallest value that keeps every k_i >= 0 moves the instances further
along that same axis. On the Triangular Table 2 set this raised GUROBI
B&B node counts 3-5x on average (up to 14x worst case).

This module applies the identical construction to the Uniform Table 1
set, reusing generate_strongly_correlated_instances' own
minimal_nonnegative_shift / rebuild_strongly_correlated_k so the two
families cannot drift apart. Every economic parameter (a, b, cost_c,
price_p, salvage_s, x_u, fixed_cost_score, kappa, capacity_fraction, C,
master_seed, instance_seed, price_markup_draw, salvage_multiplier) is
copied through unchanged; only k_i, K (via the same
K = kappa/101 * sum(k) formula) and generation_version differ.

It does NOT modify uniform_table1_instances.csv or
generate_manuscript_table_instances.py.

Usage:

    python generate_uniform_strongly_correlated_instances.py --n-values all --replications all
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate_strongly_correlated_instances as gsc
import run_uniform_table1_experiment as rue

# The k/K construction itself is distribution-independent, so it is
# imported rather than re-implemented: the Uniform and Triangular variant
# sets are then guaranteed to use the same definition of "strongly
# correlated".
minimal_nonnegative_shift = gsc.minimal_nonnegative_shift
rebuild_strongly_correlated_k = gsc.rebuild_strongly_correlated_k

STRONGLY_CORRELATED_VERSION = gsc.STRONGLY_CORRELATED_VERSION
BASE_GENERATION_VERSION = rue.EXPECTED_GENERATION_VERSION
VARIANT_GENERATION_VERSION = f"{STRONGLY_CORRELATED_VERSION}+{BASE_GENERATION_VERSION}"

DEFAULT_INPUT_CSV = "uniform_table1_instances.csv"
DEFAULT_OUTPUT_CSV = "uniform_table1_instances_strongly_correlated.csv"


def _rows_for_group(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Validate one instance's rows against the standard (manuscript)
    Uniform reader, then re-derive k/K with the strongly-correlated
    construction, returning full output rows in the input column layout.
    """
    instance = rue.parse_uniform_table1_instance(rows)
    new_k, new_K = rebuild_strongly_correlated_k(instance.fixed_cost_score, instance.kappa)

    output_rows = []
    for row, k_i in zip(rows, new_k, strict=True):
        new_row = dict(row)
        new_row["k"] = repr(k_i)
        new_row["K"] = repr(new_K)
        new_row["generation_version"] = VARIANT_GENERATION_VERSION
        output_rows.append(new_row)
    return output_rows


def generate_strongly_correlated_uniform_csv(
    input_path: Path,
    output_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> int:
    """Read the frozen Uniform Table 1 CSV, rebuild k/K for the selected
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
        rue.require(
            tuple(fieldnames or ()) == rue.EXPECTED_CSV_HEADER,
            "uniform_table1_instances.csv has an unexpected column layout",
        )
        with output_path.open("w", encoding="utf-8", newline="") as out_stream:
            writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
            writer.writeheader()
            for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
                rows = list(group)
                key = (rue.parse_int(rows[0], "n"), rue.parse_int(rows[0], "replication"))
                if key not in requested:
                    continue
                writer.writerows(_rows_for_group(rows))
                written += 1
    return written


def read_selected_strongly_correlated_uniform_instances(
    csv_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> list[rue.UniformInstance]:
    """Load instances from a strongly-correlated Uniform variant CSV (as
    produced by generate_strongly_correlated_uniform_csv), validating the
    same economic invariants as the manuscript reader EXCEPT the k
    formula, which is checked against rebuild_strongly_correlated_k
    instead of equation (13).
    """
    requested = {(n, r) for n in n_values for r in replications}
    observed: set[tuple[int, int]] = set()
    results: list[rue.UniformInstance] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rue.require(
            tuple(reader.fieldnames or ()) == rue.EXPECTED_CSV_HEADER,
            "strongly-correlated Uniform instances CSV has an unexpected column layout",
        )
        for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
            rows = list(group)
            key = (rue.parse_int(rows[0], "n"), rue.parse_int(rows[0], "replication"))
            if key not in requested:
                continue
            rue.require(key not in observed, f"duplicate selected instance group: {key}")
            instance = _parse_strongly_correlated_uniform_instance(rows)
            observed.add(key)
            results.append(instance)
    missing = sorted(requested - observed)
    rue.require(
        not missing,
        f"strongly-correlated Uniform CSV is missing selected instances: {missing}",
    )
    return results


def _parse_strongly_correlated_uniform_instance(
    rows: Sequence[Mapping[str, str]]
) -> rue.UniformInstance:
    """Mirrors run_uniform_table1_experiment.parse_uniform_table1_instance
    exactly, EXCEPT the k/K construction check, which validates against
    rebuild_strongly_correlated_k (minimal shift) instead of the base
    parser's hard-coded equation (13) (k_i = score_i + max(score)/10).
    Every other invariant -- economic domain, x_u/fixed_cost_score
    formulas, capacity formula, scalar-field consistency -- is unchanged.
    """
    rue.require(bool(rows), "cannot build an instance from zero rows")
    first = rows[0]
    n = rue.parse_int(first, "n")
    replication = rue.parse_int(first, "replication")
    instance_id = first["instance_id"]
    rue.require(len(rows) == n, f"{instance_id}: expected {n} rows, observed {len(rows)}")
    rue.require(
        [rue.parse_int(row, "product_id") for row in rows] == list(range(1, n + 1)),
        f"{instance_id}: product_id sequence is not exactly 1..n",
    )

    scalar_fields = (
        "family", "n", "replication", "instance_id", "master_seed",
        "instance_seed", "price_markup_draw", "kappa", "K",
        "capacity_fraction", "C", "generation_version",
    )
    for field in scalar_fields:
        rue._scalar_field(rows, field)

    rue.require(first["family"] == rue.EXPECTED_FAMILY, f"{instance_id}: family is not UNIFORM")
    expected_id = f"UNIFORM-N{n:04d}-R{replication:03d}"
    rue.require(instance_id == expected_id, f"unexpected instance_id {instance_id!r}")
    rue.require(
        first["generation_version"] == VARIANT_GENERATION_VERSION,
        f"{instance_id}: unexpected generation version {first['generation_version']!r}; "
        f"expected {VARIANT_GENERATION_VERSION!r}",
    )
    a = tuple(rue.parse_float(row, "a") for row in rows)
    b = tuple(rue.parse_float(row, "b") for row in rows)
    cost = tuple(rue.parse_float(row, "cost_c") for row in rows)
    price = tuple(rue.parse_float(row, "price_p") for row in rows)
    salvage = tuple(rue.parse_float(row, "salvage_s") for row in rows)
    x_u = tuple(rue.parse_float(row, "x_u") for row in rows)
    scores = tuple(rue.parse_float(row, "fixed_cost_score") for row in rows)
    k = tuple(rue.parse_float(row, "k") for row in rows)
    kappa = rue.parse_int(first, "kappa")
    budget_K = rue.parse_float(first, "K")
    capacity_fraction = rue.parse_float(first, "capacity_fraction")
    capacity_C = rue.parse_float(first, "C")

    rue.require(n in rue.TABLE1_N_VALUES, f"{instance_id}: unsupported Table 1 n={n}")
    rue.require(replication in rue.TABLE1_REPLICATIONS, f"{instance_id}: invalid replication")
    rue.require(kappa == rue.EXPECTED_KAPPA, f"{instance_id}: kappa must equal 50")
    rue.require(1.0 / 5.0 <= capacity_fraction <= 1.0 / 3.0, f"{instance_id}: bad C fraction")

    for index, values in enumerate(zip(a, b, cost, price, salvage, x_u, scores), start=1):
        ai, bi, ci, pi, si, xui, score = values
        rue.require(0.0 <= ai < bi, f"{instance_id} product {index}: invalid [a,b]")
        rue.require(pi > ci, f"{instance_id} product {index}: price must exceed cost")
        rue.require(pi > si, f"{instance_id} product {index}: price must exceed salvage")
        expected_xu = rue.uniform_unconstrained_quantity(ai, bi, pi, ci, si)
        rue.require(
            rue.close_enough(xui, expected_xu),
            f"{instance_id} product {index}: x_u formula mismatch",
        )
        expected_score = rue.uniform_expected_profit(xui, ai, bi, pi, ci, si)
        rue.require(
            rue.close_enough(score, expected_score),
            f"{instance_id} product {index}: fixed_cost_score mismatch",
        )

    expected_k, expected_K = rebuild_strongly_correlated_k(scores, kappa)
    for index, (observed, expected) in enumerate(zip(k, expected_k), start=1):
        rue.require(
            rue.close_enough(observed, expected),
            f"{instance_id} product {index}: strongly-correlated k formula mismatch",
        )
    rue.require(
        rue.close_enough(budget_K, expected_K),
        f"{instance_id}: strongly-correlated K formula mismatch",
    )
    rue.require(
        rue.close_enough(capacity_C, capacity_fraction * sum(x_u)),
        f"{instance_id}: C formula mismatch",
    )

    return rue.UniformInstance(
        n=n,
        replication=replication,
        instance_id=instance_id,
        master_seed=rue.parse_int(first, "master_seed"),
        instance_seed=rue.parse_int(first, "instance_seed"),
        a=a,
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
    n_values = rue.parse_int_selection(args.n_values, rue.TABLE1_N_VALUES, "n")
    replications = rue.parse_int_selection(
        args.replications, rue.TABLE1_REPLICATIONS, "replication"
    )
    base = Path(__file__).parent
    written = generate_strongly_correlated_uniform_csv(
        base / args.input_csv, base / args.output_csv, n_values, replications
    )
    print(f"Wrote {written} strongly-correlated Uniform instances to {base / args.output_csv}")
    # Round-trip sanity check: read back and validate every written instance.
    read_back = read_selected_strongly_correlated_uniform_instances(
        base / args.output_csv, n_values, replications
    )
    print(f"Round-trip validated {len(read_back)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
