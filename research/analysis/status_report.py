#!/usr/bin/env python
"""Generate STATUS.md from whatever results exist (partial runs included, clearly labelled).
All numbers come from experiments/*; nothing is typed by hand.   python analysis/status_report.py"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from figures import probes_figure  # noqa: E402

E = ROOT / "experiments"
out = [f"# Status report (generated {time.strftime('%Y-%m-%d %H:%M')})", "",
       "Every number below is read from `experiments/*`. Partial runs are labelled PARTIAL.", ""]


def md(df, fmt=".3f"):
    return df.to_markdown(index=False, floatfmt=fmt)


# progress
out += ["## Progress", "", "| experiment | edited images on disk | evaluated rows |", "|---|---|---|"]
for d in sorted(E.iterdir()):
    n_img = len(list(d.glob("images/*/*/*/*/L*.png")))
    rows = d / "results" / "rows.csv"
    out.append(f"| {d.name} | {n_img} | {len(pd.read_csv(rows)) if rows.exists() else 'not yet'} |")
out.append("")

# pilot
p = E / "pilot" / "results"
if (p / "summary_wm.csv").exists():
    sw = pd.read_csv(p / "summary_wm.csv")
    out += ["## Pilot (n=10 images, complete) — detection rate @ FPR 1e-3, L0 = watermarked original", ""]
    t = sw.pivot_table(index=["editor", "method", "watermark"], columns="level", values="det_rate").reset_index()
    t.columns = [c if isinstance(c, str) else f"L{c}" for c in t.columns]
    out += [md(t, ".2f"), "", "Bit accuracy:", ""]
    t = sw.pivot_table(index=["editor", "method", "watermark"], columns="level", values="bit_acc").reset_index()
    t.columns = [c if isinstance(c, str) else f"L{c}" for c in t.columns]
    out += [md(t), ""]
    fp = pd.read_csv(p / "fpr.csv").groupby("detector")[["false_positives", "n"]].sum().reset_index()
    out += ["False positives on the unwatermarked track (all editors, levels):", "", md(fp, ".0f"), ""]
    d = pd.read_csv(p / "rows_derived.csv", dtype={"id": str})
    e = d[d.level > 0].groupby(["editor", "method", "watermark"])[
        ["edit_success", "clip_dir", "clip_t_gain", "dino", "lpips"]].mean().reset_index()
    out += ["Edit fidelity, mean over rounds 1–4 (`none` = B0, editor without watermark):", "", md(e), ""]
    q = d[d.level == 0].drop_duplicates(["editor", "watermark", "id"]).groupby("watermark")[["wm_psnr", "wm_ssim"]].mean()
    out += ["Watermark imperceptibility at L0 (vs original):", "", md(q.reset_index().query("watermark != 'none'"), ".2f"), ""]
    out += ["Figures: `figures/pilot_fig2_retention_det_rate.png`, `figures/pilot_fig3_methods_ip2p_det_rate.png`, "
            "`figures/pilot_fig4_tradeoff_clip_dir.png`, `figures/pilot_fig7_qualitative_wam_ip2p.png`", ""]

# probes (partial is fine: per-row json files)
rows = [r for f in sorted((E / "probes" / "probes" / "rows").glob("*.json")) for r in json.loads(f.read_text())]
if rows:
    pr = pd.DataFrame(rows)
    done = pr.groupby("probe").id.nunique()
    tag = "complete" if done.min() >= 27 else "PARTIAL"
    g = pr.groupby(["probe", "watermark"]).agg(n=("id", "nunique"), det=("detected", "mean"), bit_acc=("bit_acc", "mean"),
                                               res_corr=("res_corr", "mean")).reset_index()
    g = g.pivot_table(index=["probe", "n"], columns="watermark", values=["det", "res_corr"]).round(3)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    out += [f"## Failure-mode probes ({tag}; up to 27 images per probe)", "",
            "`det` = detection rate; `res_corr` = correlation of the watermark residual before/after the transform "
            "(1 = residual intact, 0 = destroyed).", "", md(g.reset_index()), ""]
    probes_figure(pr, "probes_partial" if tag == "PARTIAL" else "probes")
    out += [f"Figure: `figures/{'probes_partial' if tag == 'PARTIAL' else 'probes'}_fig9_probes.png`", ""]

(ROOT / "STATUS.md").write_text("\n".join(out))
print("\n".join(out))
