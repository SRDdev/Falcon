#!/usr/bin/env python
"""Region analysis (WAM only — the one localising detector): after round-1 edits, does the watermark survive
inside vs outside the PIE-Bench++ edit mask, and does decoding from the unedited region alone recover
payloads that whole-image decoding loses?

  python run_region_analysis.py --config configs/stage2.yaml
Writes experiments/<name>/region/rows.csv. Uses only images already produced by run_experiment.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from wmedit.pipeline import Experiment, load_config  # noqa: E402
from wmedit.watermarks import detection_threshold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    exp = Experiment(load_config(a.config))
    exp.prepare()
    wm = exp.watermark("wam")
    thr = detection_threshold(wm.nbits, float(exp.cfg.get("evaluation", {}).get("target_fpr", 1e-3)))
    rows = []
    for ed in exp.ed_cfgs:
        for s in exp.samples:
            edit_m = exp.masks[s.id].astype(np.float32)
            if not 0.02 < edit_m.mean() < 0.98:  # whole-image or empty masks carry no region information
                continue
            bits = exp.payload(wm, s.id)
            for r in range(0, exp.rounds + 1):
                p = exp.level_path("baseline", ed, "wam", s.id, r)
                if not p.exists():
                    break
                mask_p, bitp = wm.detect_raw(Image.open(p).convert("RGB"))  # (1,1,h,w), (1,k,h,w) at 256x256
                em = F.interpolate(torch.from_numpy(edit_m)[None, None].to(mask_p), size=mask_p.shape[-2:], mode="nearest")
                row = {"editor": ed, "id": s.id, "category": s.category, "level": r, "edit_area": float(edit_m.mean())}
                for name, reg in [("inside", em), ("outside", 1 - em), ("whole", torch.ones_like(em))]:
                    w = (mask_p > 0.5).float() * reg  # pixels WAM says are watermarked, within the region
                    row[f"wm_frac_{name}"] = float((mask_p > 0.5).float().mul(reg).sum() / reg.sum().clamp_min(1))
                    # decode from region only: average bit logits over predicted-watermarked pixels in the region,
                    # falling back to all region pixels when none are predicted (WAM's own 'semihard' rule)
                    ww = w if w.sum() > 0 else reg
                    logits = (bitp * ww).sum((2, 3)) / ww.sum((2, 3)).clamp_min(1)
                    m = int(((logits[0] > 0).cpu().numpy().astype(np.uint8) == bits).sum())
                    row[f"acc_{name}"], row[f"det_{name}"] = m / wm.nbits, m >= thr
                rows.append(row)
    out = exp.root / "region"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    print(df.groupby(["editor", "level"])[["wm_frac_inside", "wm_frac_outside", "det_whole", "det_outside", "det_inside"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
