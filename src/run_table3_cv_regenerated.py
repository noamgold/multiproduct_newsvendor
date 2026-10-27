"""Rerun Table 3 after deriving budgets from the portfolio-CV objective.

The original strongly correlated CSV supplies the same sampled demand and
economic parameters.  For each instance this script recomputes the unconstrained
quantities, profit scores, fixed-cost coefficients, fixed-cost budget, and capacity
from the portfolio-CV objective before calling the existing Table 3 runner.
It writes a product-level manifest so the transformed instances are auditable.

Run ``python src/run_table3_cv_regenerated.py`` from any working directory.
The complete experiment covers 30 replications for each n=100,...,700 and
the Big-M, perspective, and dual-cut methods. Results, manifest, and settings
are written below ``runs/`` by default so the archived reference in ``data/``
is not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "data" / "distribution_free_table3_instances_strongly_correlated.csv"
EXPECTED_SOURCE_SHA256 = "00fcb6146d1384f04f895d6189bfbce287657022cee2b0069d6c6546b25e829f"

import distribution_free_table3 as dft
import generate_strongly_correlated_distribution_free_instances as gscd
import run_distribution_free_strongly_correlated as runner


EPSILON = dft.PORTFOLIO_FIXED_COST_EPSILON
MANIFEST_FIELDS = (
    "instance_id", "n", "replication", "product_id", "mu", "sigma", "price_p",
    "cost_c", "salvage_s", "lambda", "cv_squared", "variance_weight", "x_u",
    "fixed_cost_score", "k", "kappa", "K", "capacity_fraction", "C",
    "source_generation_version", "source_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="all")
    parser.add_argument("--replications", default="all")
    parser.add_argument("--results-csv", default="runs/table3_cv_regenerated_results.csv")
    parser.add_argument("--manifest-csv", default="runs/table3_cv_regenerated_instances.csv")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=5.0e-4)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.lambda_ <= 0.0 or args.time_limit <= 0.0 or args.mip_gap < 0.0:
        parser.error("lambda and time limit must be positive; mip gap must be nonnegative")
    results = output_path(args.results_csv)
    manifest = output_path(args.manifest_csv)
    settings_path = results.with_suffix(".settings.json")
    if results == manifest or results == settings_path or manifest == settings_path:
        parser.error("results, manifest, and settings paths must be distinct")
    if results.exists() and not args.resume:
        parser.error(f"{results} already exists; use --resume to append missing solves")

    source_digest = sha256(SOURCE_CSV)
    if source_digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Table 3 source CSV changed: {source_digest}")
    settings = {
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)),
        "source_sha256": source_digest,
        "instance_derivation": "portfolio_cv_recompute_xu_score_k_K_C_v1",
        "n_values": args.n_values,
        "replications": args.replications,
        "lambda": args.lambda_,
        "epsilon": EPSILON,
        "time_limit_seconds": args.time_limit,
        "mip_gap": args.mip_gap,
        "threads": 1,
        "solver": "GUROBI",
        "methods": ["bigm", "perspective", "cuts"],
        "cut_family": "dual",
        "results_csv": str(results.relative_to(ROOT)) if results.is_relative_to(ROOT) else str(results),
        "manifest_csv": str(manifest.relative_to(ROOT)) if manifest.is_relative_to(ROOT) else str(manifest),
    }
    if args.resume and settings_path.exists():
        recorded = json.loads(settings_path.read_text(encoding="utf-8"))
        if recorded != settings:
            raise ValueError(f"{settings_path} does not match this run's settings")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    transformed_count = 0
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        n_values = dft.parse_int_selection(args.n_values, dft.TABLE3_N_VALUES, "n")
        replications = dft.parse_int_selection(args.replications, dft.TABLE3_REPLICATIONS, "replication")
        for instance in gscd.read_selected_strongly_correlated_distribution_free_instances(
            SOURCE_CSV, n_values, replications
        ):
            cv_instance = dft.with_portfolio_variance_bound(instance, args.lambda_)
            if cv_instance.generation_version != dft.PORTFOLIO_CV_REGENERATED_VERSION:
                raise ValueError(f"{instance.instance_id}: portfolio budgets were not regenerated")
            weights = dft.penalty_weights(cv_instance)
            for i in range(instance.n):
                writer.writerow({
                    "instance_id": instance.instance_id,
                    "n": instance.n,
                    "replication": instance.replication,
                    "product_id": i + 1,
                    "mu": instance.mu[i],
                    "sigma": instance.sigma[i],
                    "price_p": instance.price_p[i],
                    "cost_c": instance.cost_c[i],
                    "salvage_s": instance.salvage_s[i],
                    "lambda": args.lambda_,
                    "cv_squared": (instance.sigma[i] / instance.mu[i]) ** 2,
                    "variance_weight": weights[i],
                    "x_u": cv_instance.x_u[i],
                    "fixed_cost_score": cv_instance.fixed_cost_score[i],
                    "k": cv_instance.k[i],
                    "kappa": instance.kappa,
                    "K": cv_instance.K,
                    "capacity_fraction": instance.capacity_fraction,
                    "C": cv_instance.C,
                    "source_generation_version": instance.generation_version,
                    "source_sha256": source_digest,
                })
            stream.flush()
            transformed_count += 1

    print(f"Regenerated instance data from {SOURCE_CSV}", flush=True)
    print(f"Source SHA256: {source_digest}", flush=True)
    exit_code = runner.main([
            "--n-values", args.n_values,
            "--replications", args.replications,
            "--variance-bound", "portfolio",
            "--lambda", str(args.lambda_),
            "--methods", "all",
            "--solver", "GUROBI",
            "--time-limit", str(args.time_limit),
            "--mip-gap", str(args.mip_gap),
            "--threads", "1",
            "--cut-family", "dual",
            "--portfolio-budgets", "regenerated",
            "--instances-csv", str(SOURCE_CSV),
            "--results-csv", str(results),
            *(["--resume"] if args.resume else []),
    ])
    print(f"Regenerated {transformed_count} instances in {manifest}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
