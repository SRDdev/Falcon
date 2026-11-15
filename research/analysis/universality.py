#!/usr/bin/env python
"""Does edit magnitude explain watermark loss independently of which editor produced the edit?

Preliminary test of the EWO 'universality' hypothesis on existing rows (no images needed). Edit magnitude
is the cumulative pixel change psnr_vs_orig, a global stand-in for rho_area until EWO maps are computed
on the GPU host. For each watermark we fit logistic(detected ~ magnitude) and compare
  in-editor     fit and evaluate on the same editor (5-fold CV)         -> how much magnitude explains
  leave-one-out fit on the other editors, evaluate on the held-out one  -> does the curve transfer?
  editor-rate   predict the held-out editor with the training editors' base rate (editor identity only)
If leave-one-out AUC ~ in-editor AUC and log-loss beats editor-rate, one magnitude curve explains loss
across editors.

  python analysis/universality.py --exp experiments/stage2
Writes <exp>/results/universality.csv and tables/<exp-name>_universality.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
FEATURE = "psnr_vs_orig"


def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def _fit(X, y):
    return LogisticRegression(C=1e3).fit(X, y)


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.method == "baseline") & (df.watermark != "none") & (df.level >= 1)].dropna(subset=[FEATURE])
    d = d[d.detected_at_l0 == 1] if "detected_at_l0" in d else d
    out = []
    for wm, g in d.groupby("watermark"):
        for held in sorted(g.editor.unique()):
            te, tr = g[g.editor == held], g[g.editor != held]
            X_te, y_te = te[[FEATURE]].to_numpy(), te.detected.astype(int).to_numpy()
            X_tr, y_tr = tr[[FEATURE]].to_numpy(), tr.detected.astype(int).to_numpy()
            row = {"watermark": wm, "held_out_editor": held, "n": len(te), "det_rate": y_te.mean()}
            if 0 < y_tr.mean() < 1:
                p = _fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
                row |= {"loo_auc": _auc(y_te, p), "loo_logloss": log_loss(y_te, p, labels=[0, 1])}
            base = np.clip(np.full(len(y_te), y_tr.mean()), 1e-3, 1 - 1e-3)
            row["editor_rate_logloss"] = log_loss(y_te, base, labels=[0, 1])
            if 0 < y_te.mean() < 1 and min(np.bincount(y_te)) >= 5:
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                p_in = cross_val_predict(LogisticRegression(C=1e3), X_te, y_te, cv=cv, method="predict_proba")[:, 1]
                row |= {"in_auc": _auc(y_te, p_in), "in_logloss": log_loss(y_te, p_in, labels=[0, 1])}
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    exp = ROOT / a.exp if not Path(a.exp).is_absolute() else Path(a.exp)
    src = exp / "results" / "rows_derived.csv"
    res = analyse(pd.read_csv(src if src.exists() else exp / "results" / "rows.csv"))
    res.to_csv(exp / "results" / "universality.csv", index=False)
    md = ROOT / "tables" / f"{exp.name}_universality.md"
    md.write_text(res.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
