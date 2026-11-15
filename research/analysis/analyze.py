#!/usr/bin/env python
"""Aggregate per-image rows into summary tables (CSV + Markdown). Never hand-edit the outputs.

  python analysis/analyze.py --exp experiments/pilot
Writes <exp>/results/{summary_wm.csv, summary_edit.csv, fpr.csv, paired_tests.csv, decay_fits.csv}
and tables/<exp-name>_*.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["method", "editor", "watermark"]
EDIT_METRICS = ["clip_t", "clip_t_gain", "clip_dir", "clip_i", "dino", "lpips", "ssim", "psnr", "bg_psnr",
                "bg_lpips", "edit_success"]


def boot_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "clip_dir" in df:
        df["edit_success"] = ((df.clip_dir > 0) & (df.clip_t_gain > 0)).astype(float)
        df.loc[df.level == 0, "edit_success"] = np.nan
    wm = df[df.watermark != "none"]
    l0 = (wm[wm.level == 0].drop_duplicates(KEYS[1:] + ["id"])  # L0 is shared by all methods
          .set_index(KEYS[1:] + ["id"])["detected"].to_dict())
    df["detected_at_l0"] = [l0.get((e, w, i), np.nan) if w != "none" else np.nan
                            for e, w, i in zip(df.editor, df.watermark, df.id)]
    if "detected" in df:
        df["joint_success"] = (df.detected.astype(float) * df.edit_success).where(df.level > 0)
    return df


def summarize_wm(df):
    out = []
    d = df[df.watermark != "none"]
    for (m, e, w, l), g in d.groupby(KEYS + ["level"]):
        k, n = int(g.detected.sum()), len(g)
        g0 = g[g.detected_at_l0 == 1]
        lo, hi = boot_ci(g.bit_acc.values)
        wl, wh = wilson(k, n)
        out.append(dict(method=m, editor=e, watermark=w, level=l, n=n,
                        bit_acc=g.bit_acc.mean(), bit_acc_sd=g.bit_acc.std(), bit_acc_median=g.bit_acc.median(),
                        bit_acc_ci_lo=lo, bit_acc_ci_hi=hi, det_rate=k / n, det_ci_lo=wl, det_ci_hi=wh,
                        det_rate_given_l0=g0.detected.mean() if len(g0) else np.nan, n_l0=len(g0),
                        exact_payload=g.exact_payload.mean(),
                        joint_success=g.joint_success.mean() if l > 0 else np.nan))
    return pd.DataFrame(out)


def summarize_edit(df):
    out = []
    d = df[df.level > 0]
    for (m, e, w, l), g in d.groupby(KEYS + ["level"]):
        r = dict(method=m, editor=e, watermark=w, level=l, n=len(g))
        for c in EDIT_METRICS:
            if c in g and g[c].notna().any():
                r[c] = g[c].mean()
                r[c + "_sd"] = g[c].std()
        out.append(r)
    return pd.DataFrame(out)


def fpr_table(df):
    d = df[df.watermark == "none"]
    out = []
    for det in sorted({c[:-9] for c in d.columns if c.endswith("_detected")}):
        for (e, l), g in d.groupby(["editor", "level"]):
            k, n = int(g[f"{det}_detected"].sum()), len(g)
            out.append(dict(detector=det, editor=e, level=l, n=n, false_positives=k, fpr=k / n,
                            fpr_ci_hi=wilson(k, n)[1], mean_bit_acc=g[f"{det}_bit_acc"].mean()))
    return pd.DataFrame(out)


def paired_tests(df, ref="baseline"):
    """Every non-reference method vs `ref` on the same (editor, watermark, level, image)."""
    out = []
    d = df[(df.watermark != "none") & (df.level > 0)]
    for (e, w, l), g in d.groupby(["editor", "watermark", "level"]):
        piv = {c: g.pivot_table(index="id", columns="method", values=c) for c in
               ["bit_acc", "detected", "clip_dir", "clip_t_gain", "lpips", "dino"] if c in g}
        for m in g.method.unique():
            if m == ref:
                continue
            r = dict(editor=e, watermark=w, level=l, method=m, reference=ref)
            for c, p in piv.items():
                if m not in p or ref not in p:
                    continue
                pp = p[[m, ref]].dropna()
                diff = pp[m] - pp[ref]
                r[f"{c}_n"] = len(pp)
                r[f"{c}_diff"] = diff.mean()
                lo, hi = boot_ci(diff.values)
                r[f"{c}_diff_ci_lo"], r[f"{c}_diff_ci_hi"] = lo, hi
                if c == "detected":
                    b = int(((pp[m] == 1) & (pp[ref] == 0)).sum())
                    cc = int(((pp[m] == 0) & (pp[ref] == 1)).sum())
                    r[f"{c}_p"] = stats.binomtest(b, b + cc).pvalue if b + cc else 1.0  # exact McNemar
                else:
                    r[f"{c}_p"] = stats.wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) > 5 else np.nan
            out.append(r)
    res = pd.DataFrame(out)
    for c in [c for c in res.columns if c.endswith("_p")]:  # Holm within each metric family
        res[c + "_holm"] = holm(res[c].values)
    return res


def holm(p):
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    vals = p[ok][idx] * (ok.sum() - np.arange(ok.sum()))
    vals = np.minimum(1, np.maximum.accumulate(vals))
    tmp = np.empty_like(vals)
    tmp[idx] = vals
    adj[ok] = tmp
    return adj


def decay_fits(summ):
    """Shape of retention decay: excess bit accuracy e(L) = (acc(L) - .5)/(acc(0) - .5), compare
    linear e = 1 - bL vs exponential e = exp(-kL) by SSE over L = 0..R (both 1-parameter)."""
    from scipy.optimize import minimize_scalar

    out = []
    for (m, e, w), g in summ.groupby(KEYS):
        g = g.sort_values("level")
        L, acc = g.level.values.astype(float), g.bit_acc.values
        if len(L) < 3 or acc[0] <= 0.5:
            continue
        ex = (acc - 0.5) / (acc[0] - 0.5)
        lin = minimize_scalar(lambda b: ((1 - b * L - ex) ** 2).sum(), bounds=(0, 5), method="bounded")
        exp = minimize_scalar(lambda k: ((np.exp(-k * L) - ex) ** 2).sum(), bounds=(0, 20), method="bounded")
        out.append(dict(method=m, editor=e, watermark=w, linear_b=lin.x, linear_sse=lin.fun,
                        exp_k=exp.x, exp_sse=exp.fun, better="exponential" if exp.fun < lin.fun else "linear",
                        first_round_drop=ex[0] - ex[1], mean_later_drop=np.mean(-np.diff(ex)[1:])))
    return pd.DataFrame(out)


def to_md(df: pd.DataFrame, path: Path, floatfmt=".3f"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_markdown(index=False, floatfmt=floatfmt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    exp = (ROOT / a.exp) if not Path(a.exp).is_absolute() else Path(a.exp)
    df = add_derived(pd.read_csv(exp / "results" / "rows.csv", dtype={"id": str}))
    res = exp / "results"
    sw, se, fp = summarize_wm(df), summarize_edit(df), fpr_table(df)
    pt, dc = paired_tests(df), decay_fits(sw)
    df.to_csv(res / "rows_derived.csv", index=False)
    for name, t in [("summary_wm", sw), ("summary_edit", se), ("fpr", fp), ("paired_tests", pt), ("decay_fits", dc)]:
        t.to_csv(res / f"{name}.csv", index=False)
    tag = exp.name
    # Sequential table: Method | L0 .. L4  (detection rate and bit accuracy)
    for val in ["det_rate", "bit_acc"]:
        seq = sw.pivot_table(index=KEYS, columns="level", values=val).reset_index()
        seq.columns = [c if isinstance(c, str) else f"L{c}" for c in seq.columns]
        to_md(seq, ROOT / "tables" / f"{tag}_sequential_{val}.md")
    to_md(fp, ROOT / "tables" / f"{tag}_fpr.md")
    to_md(dc, ROOT / "tables" / f"{tag}_decay.md")
    cols = ["method", "editor", "watermark", "level", "n", "edit_success", "clip_dir", "clip_t_gain", "dino", "lpips"]
    to_md(se[[c for c in cols if c in se]], ROOT / "tables" / f"{tag}_edit.md")
    print(sw.pivot_table(index=KEYS, columns="level", values="det_rate").round(2).to_string())
    print(sw.pivot_table(index=KEYS, columns="level", values="bit_acc").round(3).to_string())
    print(fp.groupby("detector")[["false_positives", "n"]].sum().to_string())


if __name__ == "__main__":
    main()
