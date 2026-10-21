"""Figures for the replication (light-surface PNGs; every plotted value is also saved as CSV by run.py).

Palette: the first five slots of a validated categorical palette (lightness band, chroma floor, CVD and
normal-vision separation all pass; contrast to the surface is below 3:1 for aqua, yellow and magenta), so every
series also has its own marker and line style and a legend is always present.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {  # method -> (colour, marker, linestyle)
    "DBS": ("#2a78d6", "o", "-"),
    "SmD": ("#eb6834", "s", "--"),
    "wEffRes": ("#1baf7a", "^", "-."),
    "DispFilt": ("#eda100", "D", ":"),
    "W": ("#e87ba4", "v", (0, (5, 1))),
}
METRIC_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curves(rho_df, si_df, x_metric, path, network="hospital"):
    """rho_C, rho_t and xi^-1_0.5 versus adjusted size X, one line per method (cf. paper Fig. 4)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    panels = [(rho_df, "rho_C", r"$\rho_C$  eigenvector-centrality rank"),
              (si_df, "rho_t", r"$\rho_t$  infection-order rank"),
              (si_df, "xi", r"$\xi^{-1}_{0.5}$  half-infection time ratio")]
    for ax, (df, col, title) in zip(axes, panels):
        for m, (c, mk, ls) in STYLE.items():
            d = df[df.method == m].sort_values("X")
            if col == "rho_C":
                d = d.iloc[:: max(1, len(d) // 40)]  # thin the 1,066-point sweep so markers stay readable
            ax.plot(d.X, d[col], color=c, ls=ls, lw=1.6, marker=mk, ms=3.8, mec=SURFACE, mew=0.6, label=m)
        ax.axvline(x_metric, color=INK2, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("adjusted size X  (0 = tree, 1 = full graph)")
    axes[0].text(x_metric + 0.01, 0.02, "metric backbone", color=INK2, fontsize=8, rotation=90, va="bottom",
                 transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel("preserved (1 = identical)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{network}: fidelity of each sparsification versus network size", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.03)
    _save(fig, path)


def auc_bars(auc_df, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [("AUC_rho_C", r"$\rho_C$"), ("AUC_rho_t", r"$\rho_t$"), ("AUC_xi", r"$\xi^{-1}_{0.5}$")]
    w = 0.16
    for k, m in enumerate(STYLE):
        d = auc_df.set_index("method").loc[m]
        xs = np.arange(len(cols)) + (k - 2) * w
        ax.bar(xs, [d[c] for c, _ in cols], width=w - 0.02, color=STYLE[m][0], label=m, zorder=3,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(cols)), [t for _, t in cols])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("area under the curve (1 = perfect)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, path)


def lambda_curves(lam_df, opt_df, path):
    """DBS parameter study: X and the three fidelities versus lambda, and the optimal lambda per scenario."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    d = lam_df[np.isfinite(lam_df.lam)]
    for col, lab, c, mk in [("X", "X (size)", METRIC_COLOURS[3], "D"), ("rho_C", r"$\rho_C$", METRIC_COLOURS[0], "o"),
                            ("rho_t", r"$\rho_t$", METRIC_COLOURS[1], "s"),
                            ("xi", r"$\xi^{-1}_{0.5}$", METRIC_COLOURS[2], "^")]:
        a.plot(d.lam, d[col], color=c, marker=mk, ms=3.5, lw=1.6, mec=SURFACE, mew=0.5, label=lab)
    a.set_xscale("log")
    for x, t in [(1 / 3, r"$\lambda=1/3$"), (1.0, "metric")]:
        a.axvline(x, color=INK2, lw=0.8, ls=(0, (2, 3)))
        a.text(x * 1.04, 0.02, t, color=INK2, fontsize=8, rotation=90, va="bottom")
    a.set_xlabel(r"DBS parameter $\lambda_D$   (larger = sparser)")
    a.set_ylabel("value")
    a.set_title(r"Backbone $B_\lambda$: size and fidelity")
    a.legend(frameon=False, loc="center left")
    o = opt_df.iloc[::-1]
    b.scatter(o.lam_opt, range(len(o)), s=30, color=METRIC_COLOURS[0], zorder=3, edgecolor=SURFACE, linewidth=0.8)
    b.set_yticks(range(len(o)), o.scenario, fontsize=8)
    b.set_xscale("log")
    for x in (1 / 3, 1.0):
        b.axvline(x, color=INK2, lw=0.8, ls=(0, (2, 3)))
    b.set_xlabel(r"optimal $\lambda_D$   (dotted: 1/3 and the metric backbone at 1)")
    b.set_title(r"Optimal $\lambda_D$ per weighting scenario (Eq. 17)")
    _save(fig, path)


def extension_bars(auc_df, path, title, reference="DBS"):
    """Extension: AUC of rho_t and xi^-1 for W and DispFilt, plain, with a spanning tree (M+T) and with the metric backbone
    first (M+B), against the reference (DBS). `auc_df` has columns method, AUC_rho_t, AUC_xi for one lower limit."""
    a = auc_df.set_index("method")
    bases = ("W", "DispFilt")
    labels = ["plain", "+ spanning tree", "+ metric backbone"]
    shades = [METRIC_COLOURS[3], METRIC_COLOURS[0], METRIC_COLOURS[2]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (col, lab) in zip(axes, [("AUC_rho_t", r"$\rho_t$  infection order"), ("AUC_xi", r"$\xi^{-1}_{0.5}$  timescale")]):
        for g, base in enumerate(bases):
            for k, m in enumerate([base, f"{base}+T", f"{base}+B"]):
                ax.bar(g + (k - 1) * 0.25, a.loc[m, col], width=0.23, color=shades[k], zorder=3, edgecolor=SURFACE,
                       label=labels[k] if g == 0 else None)
        ax.axhline(a.loc[reference, col], color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.text(1.45, a.loc[reference, col] + 0.004, reference, color=INK2, fontsize=8, ha="right")
        ax.set_xticks(range(len(bases)), bases)
        ax.set_title(lab)
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("area under the curve (1 = perfect)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    _save(fig, path)
