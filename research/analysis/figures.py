#!/usr/bin/env python
"""All figures, generated from <exp>/results/*.csv. Never enter values by hand.

  python analysis/figures.py --exp experiments/pilot [--ablation experiments/ablation]
Palette: validated categorical slots 1-3 (dataviz reference palette); aqua is < 3:1 on white, so
every line is direct-labelled and every figure has a matching table in tables/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WM_COLOR = {"wam": "#2a78d6", "trustmark": "#eb6834", "dwtdct": "#1baf7a"}
WM_LABEL = {"wam": "WAM", "trustmark": "TrustMark", "dwtdct": "DWT-DCT-SVD"}
ED_LABEL = {"ip2p": "InstructPix2Pix", "sdedit": "SDEdit (s=0.5)"}
METHOD_STYLE = {"baseline": ("-", "o"), "reembed": ("--", "s")}  # others: dotted + triangle
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 5.5, "pdf.fonttype": 42,
})


def style(method):
    return METHOD_STYLE.get(method, (":", "^"))


def save(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def label_end(ax, x, y, text, color):
    ax.__dict__.setdefault("_end_labels", []).append((x, y, text))


def flush_labels(ax, min_gap=0.075):
    """Direct labels at line ends, nudged apart so they never overlap (gap in axis fraction)."""
    labs = sorted(ax.__dict__.pop("_end_labels", []), key=lambda t: t[1])
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    placed = []
    for x, y, text in labs:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(text, (x, y), xytext=(x + 0.12, yy), textcoords="data", va="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if abs(yy - y) > 1e-9 else None)


def retention_curves(sw, tag, method="baseline", value="det_rate"):
    """Fig 2/5/6: retention vs editing round, one panel per editor, one line per watermark."""
    d = sw[sw.method == method]
    eds = list(dict.fromkeys(d.editor))
    fig, axes = plt.subplots(1, len(eds), figsize=(3.3 * len(eds), 2.6), sharey=True, squeeze=False)
    for ax, ed in zip(axes[0], eds):
        for wm, g in d[d.editor == ed].groupby("watermark"):
            g = g.sort_values("level")
            lo, hi = (g.det_ci_lo, g.det_ci_hi) if value == "det_rate" else (g.bit_acc_ci_lo, g.bit_acc_ci_hi)
            c = WM_COLOR.get(wm, INK2)
            ax.fill_between(g.level, lo, hi, color=c, alpha=0.12, linewidth=0)
            ax.plot(g.level, g[value], color=c, marker="o", label=WM_LABEL.get(wm, wm))
            label_end(ax, g.level.iloc[-1], g[value].iloc[-1], WM_LABEL.get(wm, wm), c)
        ax.set_title(ED_LABEL.get(ed, ed), fontsize=9, color=INK)
        ax.set_xlabel("editing round (L0 = watermarked original)")
        ax.set_xticks(range(int(d.level.max()) + 1))
        ax.set_xlim(-0.2, d.level.max() + 1.2)
    axes[0][0].set_ylabel("detection rate @ FPR 1e-3" if value == "det_rate" else "bit accuracy")
    if value == "bit_acc":
        for ax in axes[0]:
            ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    axes[0][0].set_ylim(-0.02 if value == "det_rate" else 0.4, 1.02)
    for ax in axes[0]:
        flush_labels(ax)
    n = int(d.n.min())
    fig.suptitle(f"Watermark retention under sequential editing ({method}, n={n} images; 95% CI)", fontsize=9,
                 color=INK, y=1.06)
    save(fig, f"{tag}_fig2_retention_{value}")


def method_comparison(sw, tag, editor="ip2p", value="det_rate"):
    """Fig 3: methods compared per watermark (one panel per watermark)."""
    d = sw[sw.editor == editor]
    wms = [w for w in WM_COLOR if w in set(d.watermark)]
    fig, axes = plt.subplots(1, len(wms), figsize=(3.0 * len(wms), 2.6), sharey=True, squeeze=False)
    for ax, wm in zip(axes[0], wms):
        for m, g in d[d.watermark == wm].groupby("method"):
            g = g.sort_values("level")
            ls, mk = style(m)
            ax.plot(g.level, g[value], color=WM_COLOR[wm], ls=ls, marker=mk, label=m)
            label_end(ax, g.level.iloc[-1], g[value].iloc[-1], m, WM_COLOR[wm])
        ax.set_title(WM_LABEL[wm], fontsize=9)
        ax.set_xlabel("editing round")
        ax.set_xlim(-0.2, d.level.max() + 1.6)
    axes[0][0].set_ylabel("detection rate @ FPR 1e-3" if value == "det_rate" else "bit accuracy")
    for ax in axes[0]:
        flush_labels(ax)
    fig.suptitle(f"Baseline vs watermark-aware methods ({ED_LABEL.get(editor, editor)})", fontsize=9, y=1.06)
    save(fig, f"{tag}_fig3_methods_{editor}_{value}")


def tradeoff(df, tag, edit_metric="clip_dir"):
    """Fig 4: per (method, editor, watermark) mean edit metric vs detection over rounds 1..R."""
    d = df[(df.level > 0) & (df.watermark != "none")]
    g = d.groupby(["method", "editor", "watermark"]).agg(edit=(edit_metric, "mean"), det=("detected", "mean"),
                                                         n=("id", "nunique")).reset_index()
    ref = df[(df.watermark == "none") & (df.level > 0)].groupby("editor")[edit_metric].mean()
    fig, axes = plt.subplots(1, g.editor.nunique(), figsize=(3.4 * g.editor.nunique(), 2.8), squeeze=False)
    for ax, (ed, ge) in zip(axes[0], g.groupby("editor")):
        if ed in ref:
            ax.axvline(ref[ed], color=INK2, lw=0.8, ls=":")
            ax.annotate("B0 (no watermark)", (ref[ed], 1.0), fontsize=7, color=INK2, rotation=90, va="top", ha="right")
        for r in ge.itertuples():
            ls, mk = style(r.method)
            ax.scatter(r.edit, r.det, s=50, marker=mk, color=WM_COLOR.get(r.watermark, INK2), edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.annotate(f"{WM_LABEL.get(r.watermark, r.watermark)}·{r.method}", (r.edit, r.det), xytext=(5, -3),
                        textcoords="offset points", fontsize=7)
        ax.set_title(ED_LABEL.get(ed, ed), fontsize=9)
        ax.set_xlabel(f"mean {edit_metric} over rounds 1-4 (higher = edit followed)")
        ax.set_ylim(-0.05, 1.05)
    axes[0][0].set_ylabel("mean detection rate, rounds 1-4")
    save(fig, f"{tag}_fig4_tradeoff_{edit_metric}")


def qualitative(exp, df, tag, ids=None, wm="wam", editor="ip2p", methods=("baseline", "reembed")):
    """Fig 7: L0..L4 strips with detection outcome. Ids chosen by rule (first of each of 3 categories)."""
    d = df[(df.watermark == wm) & (df.editor == editor)]
    ids = ids or list(dict.fromkeys(d.sort_values(["category", "id"]).drop_duplicates("category").id))[:3]
    rows = [(i, m) for i in ids for m in methods if m in set(d.method)]
    R = int(d.level.max())
    fig, axes = plt.subplots(len(rows), R + 1, figsize=(1.6 * (R + 1), 1.75 * len(rows)), squeeze=False)
    for (sid, m), axr in zip(rows, axes):
        for r, ax in enumerate(axr):
            p = exp / "images" / ("_wm" if r == 0 else f"{m}/{editor}") / wm / (f"{sid}.png" if r == 0 else f"{sid}/L{r}.png")
            ax.axis("off")
            if not p.exists():
                continue
            ax.imshow(Image.open(p))
            row = d[(d.id == sid) & (d.method == m) & (d.level == r)]
            if len(row):
                row = row.iloc[0]
                ok = bool(row.detected)
                ax.set_title(f"L{r} {'✓' if ok else '✗'} acc {row.bit_acc:.2f}", fontsize=7, color=INK)
        axr[0].text(-0.08, 0.5, f"{m}\n{sid}", transform=axr[0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(f"{WM_LABEL.get(wm, wm)} under {ED_LABEL.get(editor, editor)} (✓ = detected @ FPR 1e-3)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"{tag}_fig7_qualitative_{wm}_{editor}")


def probes_figure(pr, tag):
    """Failure analysis: detection and residual survival per probe, per watermark."""
    order = [p for p in ["vae_roundtrip", "ip2p_identity", "ip2p_imgcfg_4.0", "ip2p_imgcfg_2.5", "ip2p_imgcfg_1.5",
                         "ip2p_imgcfg_1.0", "ip2p_steps_10", "ip2p_steps_25", "sdedit_0.1", "sdedit_0.2",
                         "sdedit_0.3", "sdedit_0.5", "sdedit_0.7"] if p in set(pr.probe)]
    g = pr.groupby(["probe", "watermark"]).agg(det=("detected", "mean"), acc=("bit_acc", "mean"),
                                               corr=("res_corr", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
    y = np.arange(len(order))
    for k, (col, xl) in enumerate([("det", "detection rate @ FPR 1e-3"), ("corr", "residual correlation corr(r0, rT)")]):
        ax = axes[k]
        for j, wm in enumerate([w for w in WM_COLOR if w in set(g.watermark)]):
            gg = g[g.watermark == wm].set_index("probe").reindex(order)
            ax.plot(gg[col], y + (j - 1) * 0.18, "o", color=WM_COLOR[wm], label=WM_LABEL[wm], markersize=6)
        ax.set_xlabel(xl)
        ax.set_xlim(-0.05, 1.05)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    per = pr.groupby("probe").id.nunique()
    n = f"n={per.min()}" if per.min() == per.max() else f"n={per.min()}-{per.max()}"
    fig.suptitle(f"Where is the watermark lost? ({n} images per probe)", fontsize=9)
    save(fig, f"{tag}_fig9_probes")
    # frequency survival
    bands = sorted([c for c in pr.columns if c.endswith("_survival")], key=lambda c: int(c[4:-9]))
    if bands:
        fig, axes = plt.subplots(1, len([w for w in WM_COLOR if w in set(pr.watermark)]), figsize=(9, 2.6), sharey=True)
        for ax, wm in zip(np.atleast_1d(axes), [w for w in WM_COLOR if w in set(pr.watermark)]):
            for p, ls in [("vae_roundtrip", "-"), ("ip2p_identity", "--"), ("sdedit_0.3", ":")]:
                gg = pr[(pr.watermark == wm) & (pr.probe == p)][bands].mean()
                if gg.notna().any():
                    ax.plot(np.arange(len(bands)) / len(bands), gg.values, ls=ls, color=WM_COLOR[wm], label=p)
            e0 = pr[pr.watermark == wm][[b.replace("survival", "energy0") for b in bands]].mean().values
            ax.bar(np.arange(len(bands)) / len(bands), e0, width=0.9 / len(bands), align="edge", color=GRID,
                   label="watermark energy share", zorder=0)
            ax.set_title(WM_LABEL[wm], fontsize=9)
            ax.set_xlabel("radial frequency (fraction of Nyquist)")
            ax.axhline(0, color=INK2, lw=0.6)
        np.atleast_1d(axes)[0].set_ylabel("residual survival per band")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        save(fig, f"{tag}_fig10_frequency_survival")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--probes", default=None)
    a = ap.parse_args()
    exp = ROOT / a.exp
    tag = exp.name
    res = exp / "results"
    sw = pd.read_csv(res / "summary_wm.csv")
    df = pd.read_csv(res / "rows_derived.csv", dtype={"id": str})
    for v in ["det_rate", "bit_acc"]:
        retention_curves(sw, tag, value=v)
        for ed in sw.editor.unique():
            method_comparison(sw, tag, editor=ed, value=v)
    for m in ["clip_dir", "clip_t_gain", "lpips"]:
        if m in df:
            tradeoff(df, tag, m)
    for wm in [w for w in WM_COLOR if w in set(df.watermark)]:
        qualitative(exp, df, tag, wm=wm)
    if a.probes:
        probes_figure(pd.read_csv(ROOT / a.probes / "probes" / "rows.csv", dtype={"id": str}), Path(a.probes).name)


if __name__ == "__main__":
    main()
