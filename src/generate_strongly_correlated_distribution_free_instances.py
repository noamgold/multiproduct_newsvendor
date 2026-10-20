"""Generate a harder-fixed-cost variant of the frozen Distribution-Free
Table 3 instances (distribution_free_table3_instances.csv) -- the third
sibling of generate_strongly_correlated_instances.py (Triangular) and
generate_strongly_correlated_uniform_instances.py (Uniform).

See generate_strongly_correlated_instances.py's module docstring for the
full background. In brief: the manuscript's equation (13) sets

    k_i = pi_tilde_i + max_{l in N}(pi_tilde_l) / 10

citing Pisinger (2004)'s correlated random knapsack scheme, whose hardest
regime is the "strongly correlated" one (weight = value + a SMALL
constant). Shrinking that additive shift to the smallest value keeping
every k_i >= 0 moves the instances further along the same axis; on the
Triangular set it raised GUROBI B&B node counts 3-5x on average.

This module applies the identical construction to the Distribution-Free
set, reusing generate_strongly_correlated_instances' own
minimal_nonnegative_shift / rebuild_strongly_correlated_k so the three
families cannot drift apart. Every other column -- mu, sigma, lambda,
cost_c, price_p, salvage_s, x_u, fixed_cost_score, kappa,
capacity_fraction, C, the seeds, and the Distribution-Free
candidate-rejection bookkeeping -- is copied through unchanged; only k,
K and generation_version differ.

For the Distribution-Free family fixed_cost_score is pi_i(x^U_i) =
g_i(0) - g_i(x^U_i), the profit form of the manuscript's cost model (see
distribution_free_table3.py), so "correlated with the item's profit"
means the same thing here as in Tables 1 and 2.

It does NOT modify distribution_free_table3_instances.csv or
generate_manuscript_table_instances.py.

Usage:

    python generate_strongly_correlated_distribution_free_instances.py --n-values all --replications all
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import distribution_free_table3 as dft
import generate_strongly_correlated_instances as gsc

# The k/K construction itself is distribution-independent, so it is
# imported rather than re-implemented: all three variant sets are then
# guaranteed to use the same definition of "strongly correlated".
minimal_nonnegative_shift = gsc.minimal_nonnegative_shift
rebuild_strongly_correlated_k = gsc.rebuild_strongly_correlated_k

STRONGLY_CORRELATED_VERSION = gsc.STRONGLY_CORRELATED_VERSION
BASE_GENERATION_VERSION = dft.EXPECTED_GENERATION_VERSION
VARIANT_GENERATION_VERSION = f"{STRONGLY_CORRELATED_VERSION}+{BASE_GENERATION_VERSION}"

DEFAULT_INPUT_CSV = dft.DEFAULT_INSTANCES_CSV
DEFAULT_OUTPUT_CSV = "distribution_free_table3_instances_strongly_correlated.csv"


def validate_strongly_correlated_fixed_costs(instance: dft.DistributionFreeInstance) -> None:
    """The k/K check for a variant file: k_i = score_i + minimal_shift and
    K = kappa/101 sum(k), in place of equation (13). Passed to
    distribution_free_table3's parser, which checks every other invariant
    with the generator's own authoritative validator.
    """
    expected_k, expected_K = rebuild_strongly_correlated_k(
        instance.fixed_cost_score, instance.kappa
    )
    for index, (observed, expected) in enumerate(zip(instance.k, expected_k), start=1):
        dft.require(
            dft.close_enough(observed, expected),
            f"{instance.instance_id} product {index}: strongly-correlated k formula mismatch",
        )
    dft.require(
        dft.close_enough(instance.K, expected_K),
        f"{instance.instance_id}: strongly-correlated K formula mismatch",
    )


def _rows_for_group(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Validate one instance's rows against the standard (manuscript)
    Distribution-Free reader, then re-derive k/K with the
    strongly-correlated construction, returning full output rows in the
    input column layout.
    """
    instance = dft.parse_distribution_free_table3_instance(rows)
    new_k, new_K = rebuild_strongly_correlated_k(instance.fixed_cost_score, instance.kappa)

    output_rows = []
    for row, k_i in zip(rows, new_k, strict=True):
        new_row = dict(row)
        new_row["k"] = repr(k_i)
        new_row["K"] = repr(new_K)
        new_row["generation_version"] = VARIANT_GENERATION_VERSION
        output_rows.append(new_row)
    return output_rows


def generate_strongly_correlated_distribution_free_csv(
    input_path: Path,
    output_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> int:
    """Read the frozen Table 3 CSV, rebuild k/K for the selected
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
        dft.require(
            tuple(fieldnames or ()) == dft.EXPECTED_CSV_HEADER,
            f"{input_path.name} has an unexpected column layout",
        )
        with output_path.open("w", encoding="utf-8", newline="") as out_stream:
            writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
            writer.writeheader()
            for instance_id, group in itertools.groupby(reader, key=lambda row: row["instance_id"]):
                rows = list(group)
                key = (int(rows[0]["n"]), int(rows[0]["replication"]))
                if key not in requested:
                    continue
                writer.writerows(_rows_for_group(rows))
                written += 1
    return written


def read_selected_strongly_correlated_distribution_free_instances(
    csv_path: Path,
    n_values: Sequence[int],
    replications: Sequence[int],
) -> list[dft.DistributionFreeInstance]:
    """Load instances from a strongly-correlated Distribution-Free variant
    CSV, validating the same economic invariants as the manuscript reader
    EXCEPT the k formula, which is checked against
    rebuild_strongly_correlated_k instead of equation (13).
    """
    return list(
        dft.read_selected_distribution_free_table3_instances(
            csv_path,
            n_values,
            replications,
            expected_generation_version=VARIANT_GENERATION_VERSION,
            validate_fixed_costs=validate_strongly_correlated_fixed_costs,
        )
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
    n_values = dft.parse_int_selection(args.n_values, dft.TABLE3_N_VALUES, "n")
    replications = dft.parse_int_selection(
        args.replications, dft.TABLE3_REPLICATIONS, "replication"
    )
    base = Path(__file__).parent
    written = generate_strongly_correlated_distribution_free_csv(
        base / args.input_csv, base / args.output_csv, n_values, replications
    )
    print(
        f"Wrote {written} strongly-correlated Distribution-Free instances "
        f"to {base / args.output_csv}"
    )
    # Round-trip sanity check: read back and validate every written instance.
    read_back = read_selected_strongly_correlated_distribution_free_instances(
        base / args.output_csv, n_values, replications
    )
    print(f"Round-trip validated {len(read_back)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
