"""Rebuild Figure 2 of Section 6.2: average test loss of the uniform,
data-driven and distribution-free models for varying K with k_i = 1.

Equivalent to ``python rebuild_figures2_3.py --figure 2``; all other
options of that script are accepted, e.g.::

    python rebuild_figure2.py
    python rebuild_figure2.py --schemes 20 --levels 40 --max-windows 1
"""

from __future__ import annotations

import sys

import rebuild_figure1 as f1
import rebuild_figures2_3

if __name__ == "__main__":
    try:
        raise SystemExit(rebuild_figures2_3.main(["--figure", "2", *sys.argv[1:]]))
    except f1.Figure1Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
