"""Main figure: accuracy vs. mean questions per session, one curve per policy family.

One panel per rate multiplier. Base families take fixed categorical slots (validated
palette); exact impact is a solid line with filled markers, coarse a dashed line with
hollow markers, and every family has its own marker shape, so identity never rests on
colour alone. `never` and `always` are neutral reference points.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BASE_STYLE = {  # base family -> (colour slot, marker)
    "ttl": ("#2a78d6", "o"),
    "volatility": ("#eb6834", "s"),
    "sxi": ("#1baf7a", "^"),
    "budget": ("#eda100", "D"),
}
LABELS = {
    "ttl": "TTL(N)", "ttl[blind]": "TTL(N), task-blind",
    "volatility": "Volatility(K)", "volatility[blind]": "Volatility(K), task-blind",
    "sxi[exact]": "Staleness×impact, exact", "sxi[coarse]": "Staleness×impact, coarse",
    "sxi[llm]": "Staleness×impact, LLM",
    "budget[exact]": "Budget(k), exact", "budget[coarse]": "Budget(k), coarse",
    "budget[llm]": "Budget(k), LLM",
}
ORDER = ["ttl", "ttl[blind]", "volatility", "volatility[blind]", "sxi[exact]", "sxi[coarse]",
         "sxi[llm]", "budget[exact]", "budget[coarse]", "budget[llm]"]
MAIN = ["ttl", "volatility", "sxi[exact]", "sxi[coarse]", "budget[exact]", "budget[coarse]"]
VARIANTS = ["ttl", "ttl[blind]", "volatility", "volatility[blind]", "sxi[coarse]", "sxi[llm]",
            "budget[coarse]", "budget[llm]"]
VARIANT = {  # variant -> (line style, marker fill: "full" | "none" | "left")
    None: ("-", "full"), "exact": ("-", "full"), "coarse": ("--", "none"),
    "llm": ("-.", "left"), "blind": (":", "none"),
}


def _style(family: str) -> dict:
    base, _, rest = family.partition("[")
    variant = rest.rstrip("]") or None
    colour, marker = BASE_STYLE[base]
    line, fill = VARIANT[variant]
    return dict(color=colour, marker=marker, linestyle=line, linewidth=2, markersize=6.5,
                markeredgewidth=1.6, fillstyle=fill,
                markerfacecolor=colour if fill != "none" else SURFACE,
                markerfacecoloralt=SURFACE, markeredgecolor=colour)


def main_figure(summary: pd.DataFrame, path: Path, title: str,
                panel_title: str = "rate multiplier {m:g}×",
                families: list[str] | None = None) -> None:
    """Accuracy vs. mean questions, one panel per rate multiplier (sweep settings only)."""
    mults = sorted(summary.rate_multiplier.unique())
    fig, axes = plt.subplots(1, len(mults), figsize=(4.6 * len(mults), 4.4), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    handles = {}
    for ax, m in zip(axes[0], mults):
        s = summary[(summary.rate_multiplier == m) & (summary.role == "sweep")]
        ax.set_facecolor(SURFACE)
        for fam in families or MAIN:
            g = s[s.family == fam].sort_values(["mean_questions", "accuracy"])
            if g.empty:
                continue
            (h,) = ax.plot(g.mean_questions, g.accuracy, label=LABELS[fam], zorder=3,
                           **_style(fam))
            handles[fam] = h
        for name, marker in (("never", "X"), ("always", "P")):
            r = s[s.policy == name]
            if len(r):
                ax.plot(r.mean_questions, r.accuracy, marker=marker, color=INK_2,
                        markersize=9, linestyle="none", zorder=4)
                ax.annotate(name, (r.mean_questions.iloc[0], r.accuracy.iloc[0]),
                            xytext=(0, 9) if name == "always" else (10, 2),
                            ha="center" if name == "always" else "left",
                            textcoords="offset points", color=INK_2, fontsize=9)
        ax.set_title(panel_title.format(m=m), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("mean questions per session", color=INK_2, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    shown_rows = summary[summary.role == "sweep"]
    ymin = shown_rows[shown_rows.family.isin(families or MAIN)
                      | shown_rows.policy.isin(["never", "always"])].accuracy.min()
    axes[0][0].set_ylim(bottom=ymin - 0.03, top=1.045)  # shared y: set once, after plotting
    axes[0][0].set_ylabel("accuracy (exact match)", color=INK_2, fontsize=10)
    shown = [f for f in (families or MAIN) if f in handles]
    fig.legend([handles[f] for f in shown], [LABELS[f] for f in shown], loc="upper center",
               ncol=4 if len(shown) > 6 else 3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.95),
               labelcolor=INK)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
