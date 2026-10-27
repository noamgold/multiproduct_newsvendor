# Distribution-free Table 3 reproduction

Run `python src/run_table3_cv_regenerated.py` from any working directory with the
dependencies and a Gurobi license installed. The script reads
`data/distribution_free_table3_instances_strongly_correlated.csv`, verifies its
SHA-256 hash, regenerates the portfolio-CV instance parameters, writes an
instance manifest and settings file under `runs/`, then solves all 210 instances
with Big-M, perspective, and dual cuts. The defaults are lambda = 1e-4, a
3,600-second limit per solve, a 5e-4 relative MIP gap, and one solver thread.
Use `--resume` to skip completed solves in the same results file.

The archived completed run is in `data/table3_cv_regenerated_results.csv`; its
product-level parameters and settings are in
`data/table3_cv_regenerated_instances.csv` and
`data/table3_cv_regenerated_results.settings.json`. The portfolio conversion
recomputes each unconstrained quantity, profit score, fixed cost, total
fixed-cost budget, and capacity. The input CSV retains the earlier sigma-squared
parameters and is not changed. The model and solver implementations are in
`src/distribution_free_table3.py`, `src/run_distribution_free_strongly_correlated.py`,
and `src/submodular_cuts.py`.

The archived `solve_seconds` values are elapsed wall time, not process CPU time.
All 630 archived solves reported optimal status. One Big-M row (n=100,
replication 20) has `soc_ok=False`; its reported independent objective differs
from the solver objective by about 0.058. The runner reports this validation
flag but does not treat it as a solver status error.
