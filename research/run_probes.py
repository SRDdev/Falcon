#!/usr/bin/env python
"""Failure-mode probes (see src/wmedit/probes.py). Resumable; writes <exp>/probes/rows.csv.

  python run_probes.py --config configs/probes.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from wmedit.pipeline import Experiment, load_config  # noqa: E402
from wmedit.probes import VAERoundTrip, identity_step, residual_stats  # noqa: E402
from wmedit.utils import stable_seed, write_json  # noqa: E402

log = logging.getLogger("probes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    exp = Experiment(cfg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exp.prepare()
    exp.embed()  # reuses cached L0 images
    pc = cfg["probes"]
    out_dir = exp.root / "probes"
    fpr = float(cfg.get("evaluation", {}).get("target_fpr", 1e-3))
    wms = [w for w in exp.wm_cfgs if w != "none"]

    def record(name, s, fn, img_dir):
        p = out_dir / "rows" / f"{name}__{s.id}.json"
        if p.exists():
            return
        orig = exp.images[s.id]
        t_orig = fn(orig, None)
        img_dir.mkdir(parents=True, exist_ok=True)
        t_orig.save(img_dir / f"{s.id}__clean.png")
        rows = []
        for w in wms:
            wm = exp.watermark(w)
            l0 = Image.open(exp.l0_path(w, s.id)).convert("RGB")
            t = fn(l0, w)
            t.save(img_dir / f"{s.id}__{w}.png")
            r = {"probe": name, "id": s.id, "watermark": w, **wm.score(t, exp.payload(wm, s.id), fpr),
                 **residual_stats(orig, l0, t_orig, t)}
            if w == "wam":
                r["wam_mask_frac"] = wm.mask_fraction(t)
            rows.append(r)
        write_json(p, rows)

    # 1) VAE bottleneck
    vae = VAERoundTrip()
    for s in exp.samples:
        record("vae_roundtrip", s, lambda im, w: vae(im), out_dir / "images" / "vae_roundtrip")
    del vae

    # 2) InstructPix2Pix ladder
    from wmedit.editors import InstructPix2Pix

    ed = InstructPix2Pix(steps=pc.get("ip2p_steps", 50))
    for s in exp.samples:
        seed = stable_seed("edit", exp.seed, s.id, 1)
        record("ip2p_identity", s, lambda im, w: ed.edit(im, identity_step(s.source_caption), seed),
               out_dir / "images" / "ip2p_identity")
        for ic in pc.get("ip2p_image_cfg", []):
            record(f"ip2p_imgcfg_{ic}", s, lambda im, w: ed.edit(im, s.plan[0], seed, image_cfg=ic),
                   out_dir / "images" / f"ip2p_imgcfg_{ic}")
        for n in pc.get("ip2p_steps_sweep", []):
            record(f"ip2p_steps_{n}", s, lambda im, w: ed.edit(im, s.plan[0], seed, steps=n),
                   out_dir / "images" / f"ip2p_steps_{n}")
    del ed

    # 3) SDEdit strength ladder (source caption -> no intended semantic change)
    from wmedit.editors import SDEdit

    sd = SDEdit()
    for s in exp.samples:
        seed = stable_seed("edit", exp.seed, s.id, 1)
        for st in pc.get("sdedit_strengths", []):
            record(f"sdedit_{st}", s, lambda im, w: sd.edit(im, identity_step(s.source_caption), seed, strength=st),
                   out_dir / "images" / f"sdedit_{st}")
    rows = [r for p in sorted((out_dir / "rows").glob("*.json")) for r in json.loads(p.read_text())]
    pd.DataFrame(rows).to_csv(out_dir / "rows.csv", index=False)
    log.info("probes: %d rows", len(rows))


if __name__ == "__main__":
    main()
