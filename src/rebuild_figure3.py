"""Rebuild Figure 3 of Section 6.2: average test loss of the uniform,
data-driven and distribution-free models when fixed costs are correlated
with profit as in (22), k_i = pi_i + epsilon, and K = kappa/101 * sum_i k_i
for kappa in {20, 25, ..., 60}.

pi_i is the expected profit at the unconstrained optimum of the
distribution-free model with lambda = 0 (Gallego and Moon), estimated from
the training window's mean and standard deviation; the same k_i are used for
all three models.  Models, costs, schemes, solvers and checkpointing are
those of rebuild_figure2.py, which this script reuses.

Typical commands::

    python rebuild_figure3.py
    python rebuild_figure3.py --schemes 20 --levels 40 --max-windows 1
"""

from __future__ import annotations

import sys

import rebuild_figure1 as f1
import rebuild_figure2

if __name__ == "__main__":
    try:
        raise SystemExit(rebuild_figure2.main(figure="3", description=__doc__))
    except f1.Figure1Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
