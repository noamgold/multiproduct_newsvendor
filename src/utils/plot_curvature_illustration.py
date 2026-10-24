"""Illustrate why the perspective reformulation strengthens the uniform
formulation (12) more than the triangular formulation (17).

One product i with the same support [a_i, b_i] under both distributions (the
triangular mode m_i at the midpoint) and the same prices; capacity coupling is
ignored, so the figure isolates a single product's contribution.

The figure plots phi_i(z_i), the product's value in the continuous relaxation,
as a function of its relaxed indicator z_i.  The guaranteed segment
(x-bar_i <= a_i z_i) is linear in z_i under every formulation, so its
contribution pi_i(a_i) z_i is subtracted and the result is normalized by the
profit of the units beyond a_i, pi_i(x_i^U) - pi_i(a_i), i.e. the part governed
by the big-M constants x_i^U - a_i (uniform) and min{x_i^U, m_i} - a_i,
(x_i^U - m_i)^+ (triangular).  On this scale the convex hull
z_i pi_i(x_i^U) is the diagonal.

The big-M relaxation lets every segment grow in proportion to z_i and lies
above the diagonal; for uniform demand the gap is
(p_i - s_i)(x_i^U - a_i)^2 z_i(1 - z_i) / (2(b_i - a_i)), i.e. at most 1/4 of
pi_i(x_i^U) - pi_i(a_i) for every product.  The perspective divides the
nonlinear terms by z_i, which restores the hull for the uniform model and for
the triangular segment below the mode, but not for the segment above the mode,
which (17) writes through y_i = b_i - m_i - x~_i and cannot perspectivize.

Usage::

    python utils/plot_curvature_illustration.py            # writes figure_curvature.pdf in src
    python utils/plot_curvature_illustration.py --output path/to/file.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

A, B = 50.0, 350.0
M = 0.5 * (A + B)
PRICE, COST, SALVAGE = 10.0, 4.0, 1.0

UNIFORM_COLOR = "#1f77b4"
TRIANGULAR_COLOR = "#e76f51"
HULL_COLOR = "0.25"
INK = "0.15"
MUTED = "0.45"

ALPHA = (PRICE - COST) / (PRICE - SALVAGE)
D = B - A
K = SALVAGE - PRICE  # negative


def uniform_xu() -> float:
    return A + D * ALPHA


def triangular_xu() -> float:
    if ALPHA <= (M - A) / D:
        return A + np.sqrt(ALPHA * D * (M - A))
    return B - np.sqrt((1.0 - ALPHA) * D * (B - M))


def uniform_value(bar, hat, z=None):
    """pi-bar of (12); with z, the perspective objective of (14)."""
    quad = hat**2 if z is None else hat**2 / z
    return (PRICE - COST) * (bar + hat) + K / (2.0 * D) * quad


def triangular_value(bar, hat, tilde, z=None):
    """pi-bar of (16); with z, the perspective objective of (17) with (13),
    which divides only the hat-segment cube by z**2."""
    hat3 = hat**3 if z is None else hat**3 / z**2
    return (
        (PRICE - COST) * (bar + hat + tilde)
        + K * (M - A) / D * tilde
        + K / D * tilde**2
        + K / (3.0 * D) * (hat3 / (M - A) - tilde**3 / (B - M))
    )


def relaxation_curves(z: np.ndarray) -> dict[str, np.ndarray]:
    """(phi_i(z_i) - pi_i(a_i) z_i) / (pi_i(x_i^U) - pi_i(a_i)) for big-M and perspective."""
    guaranteed = (PRICE - COST) * A
    hat_u = uniform_xu() - A
    scale_u = uniform_value(A, hat_u) - guaranteed
    xu_t = triangular_xu()
    hat_t, tilde_t = min(xu_t, M) - A, max(xu_t - M, 0.0)
    scale_t = triangular_value(A, hat_t, tilde_t) - guaranteed

    def normalized(value, scale):
        return (value - guaranteed * z) / scale

    return {
        "uniform_bigm": normalized(uniform_value(A * z, hat_u * z), scale_u),
        "uniform_persp": normalized(uniform_value(A * z, hat_u * z, z), scale_u),
        "triangular_bigm": normalized(triangular_value(A * z, hat_t * z, tilde_t * z), scale_t),
        "triangular_persp": normalized(triangular_value(A * z, hat_t * z, tilde_t * z, z), scale_t),
    }


def plot(output: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axis = plt.subplots(figsize=(4.2, 3.2))

    z = np.linspace(1e-6, 1.0, 401)
    curves = relaxation_curves(z)
    axis.plot(z, z, color=HULL_COLOR, linewidth=1.2, linestyle="--", label="Convex hull (uniform perspective)")
    axis.plot(z, curves["uniform_bigm"], color=UNIFORM_COLOR, linewidth=2.0, label="Uniform, big-M")
    axis.plot(z, curves["triangular_bigm"], color=TRIANGULAR_COLOR, linewidth=2.0, label="Triangular, big-M")
    axis.plot(z, curves["triangular_persp"], color=TRIANGULAR_COLOR, linewidth=1.4, linestyle=(0, (4, 2)),
              label="Triangular, perspective")
    axis.fill_between(z, z, curves["uniform_bigm"], color=UNIFORM_COLOR, alpha=0.12, linewidth=0)
    axis.fill_between(z, z, curves["triangular_bigm"], color=TRIANGULAR_COLOR, alpha=0.18, linewidth=0)

    gaps = {name: float(np.max(curves[name] - z)) for name in curves}
    for name, color, offset, align in (
        ("uniform_bigm", UNIFORM_COLOR, (-4, 0), "right"),
        ("triangular_bigm", TRIANGULAR_COLOR, (4, 0), "left"),
    ):
        at = int(np.argmax(curves[name] - z))
        axis.annotate("", xy=(z[at], z[at]), xytext=(z[at], curves[name][at]),
                      arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0})
        axis.annotate(f"{gaps[name]:.1%}", (z[at], 0.5 * (z[at] + curves[name][at])),
                      xytext=offset, textcoords="offset points", ha=align, va="center", fontsize=7, color=INK,
                      bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"relaxed indicator $z_i$")
    axis.set_ylabel(r"$\dfrac{\phi_i(z_i) - \pi_i(a_i)\,z_i}{\pi_i(x_i^U) - \pi_i(a_i)}$", fontsize=10, labelpad=10)
    axis.legend(loc="lower right", fontsize=7, frameon=False)
    axis.grid(True, alpha=0.18, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "figure_curvature.pdf")
    args = parser.parse_args()
    gaps = plot(args.output)
    print(f"x_i^U uniform {uniform_xu():.1f}, triangular {triangular_xu():.1f} "
          f"(a_i={A}, m_i={M}, b_i={B}, alpha_i={ALPHA:.3f})")
    for name, gap in gaps.items():
        print(f"max gap above hull, {name}: {gap:.3f} of pi_i(x_i^U) - pi_i(a_i)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
