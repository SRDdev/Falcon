#!/usr/bin/env python
"""Edit-Watermark Overlap for every existing baseline edit (GPU host: needs the images on disk, no models).

For each (editor, watermark, image, level) computes
  ewo_area_pix     method-independent edit extent from pixel change vs. the watermarked original
  ewo_energy_pix   energy-weighted overlap with the pixel-change map
  ewo_energy_regen energy-weighted overlap with the watermark-residual regeneration map
  ewo_energy_mask  energy-weighted overlap with the PIE-Bench++ edit mask (round 1 only)
and joins the detection outcome already stored in results/rows.csv. Then reports R(rho) per editor.

  python run_ewo_analysis.py --config configs/stage2.yaml
Writes experiments/<name>/ewo/{rows.csv, curves.csv} and tables/<name>_ewo.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from wmedit.ewo import (ewo_area, ewo_energy, recovery_curve, regeneration_map, soft_edit_map,  # noqa: E402
                        watermark_energy)
from wmedit.pipeline import Experiment, load_config  # noqa: E402

VARIANTS = ["ewo_area_pix", "ewo_energy_pix", "ewo_energy_regen", "ewo_energy_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default="baseline")
    a = ap.parse_args()
    exp = Experiment(load_config(a.config))
    exp.prepare()
    det = pd.read_csv(exp.res_dir / "rows.csv", dtype={"id": str}).set_index(["method", "editor", "watermark", "id", "level"])["detected"]
    rows = []
    for wm_name in [w for w in exp.wm_cfgs if w != "none"]:
        for s in exp.samples:
            l0 = exp.l0_path(wm_name, s.id)
            if not l0.exists():
                continue
            x = np.asarray(exp.images[s.id].convert("RGB"))
            xw = np.asarray(Image.open(l0).convert("RGB").resize(exp.images[s.id].size))
            energy = watermark_energy(x, xw)
            for ed in exp.ed_cfgs:
                for r in range(1, exp.rounds + 1):
                    p = exp.level_path(a.method, ed, wm_name, s.id, r)
                    if not p.exists():
                        break
                    xe = np.asarray(Image.open(p).convert("RGB").resize(exp.images[s.id].size))
                    pix = soft_edit_map(xw, xe)
                    row = {"method": a.method, "editor": ed, "watermark": wm_name, "id": s.id,
                           "category": s.category, "level": r,
                           "ewo_area_pix": ewo_area(pix), "ewo_energy_pix": ewo_energy(energy, pix),
                           "ewo_energy_regen": ewo_energy(energy, regeneration_map(x, xw, xe)),
                           "ewo_energy_mask": np.nan,
                           "detected": det.get((a.method, ed, wm_name, s.id, r), np.nan)}
                    m = exp.masks.get(s.id)
                    if r == 1 and m is not None and 0.02 < m.mean() < 0.98:
                        row["ewo_energy_mask"] = ewo_energy(energy, m.astype(float))
                    rows.append(row)
    out = exp.root / "ewo"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    n_join = int(df.detected.notna().sum()) if len(df) else 0
    print(f"{len(df)} edits scored, {n_join} joined to detection rows")
    if n_join == 0:
        raise SystemExit("no detection rows joined: check ids/levels in results/rows.csv")
    curves = []
    for (ed, wm_name), g in df.dropna(subset=["detected"]).groupby(["editor", "watermark"]):
        for v in VARIANTS:
            gg = g.dropna(subset=[v])
            if len(gg):
                c = recovery_curve(gg[v], gg.detected.astype(float))
                curves.append(c.assign(editor=ed, watermark=wm_name, variant=v))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "curves.csv", index=False)
    (Path(__file__).resolve().parent / "tables" / f"{exp.cfg['name']}_ewo.md").write_text(
        curves.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(curves.pivot_table(index=["variant", "watermark", "editor"], columns="bin", values="rate").round(2))


if __name__ == "__main__":
    main()
