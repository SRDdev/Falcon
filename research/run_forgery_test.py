#!/usr/bin/env python
"""Does watermark-guided editing *preserve* a watermark, or *write* one?

Run the guided editor on UNWATERMARKED images with a random target payload (1 round). If the output is
detected as carrying that payload, guidance implants watermarks rather than preserving them, i.e. it is
decoder-gradient re-embedding (and equally usable for forgery). Re-embed is the encoder analogue.

  python run_forgery_test.py --config configs/guided_test.yaml
Writes experiments/<name>/forgery/rows.csv.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wmedit.editors import InstructPix2Pix  # noqa: E402
from wmedit.methods import build_method  # noqa: E402
from wmedit.pipeline import Experiment, load_config  # noqa: E402
from wmedit.utils import stable_seed, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exp = Experiment(load_config(a.config))
    exp.prepare()
    out_dir = exp.root / "forgery"
    ed_cfg = exp.ed_cfgs["ip2p"]
    ed = InstructPix2Pix(**{k: v for k, v in ed_cfg.items() if k != "name"})
    fpr = float(exp.cfg.get("evaluation", {}).get("target_fpr", 1e-3))
    guided = [m for m in exp.methods.values() if m["name"].startswith("guided")]
    for wm_name in [w for w in exp.wm_cfgs if w != "none"]:
        wm = exp.watermark(wm_name)
        for s in exp.samples:
            # payload the clean image never carried
            bits = np.random.default_rng(stable_seed("forge", exp.seed, wm_name, s.id)).integers(0, 2, wm.nbits).astype(np.uint8)
            seed = stable_seed("edit", exp.seed, s.id, 1)
            clean = exp.images[s.id]
            for mcfg in [{"name": "baseline"}, {"name": "reembed_forced"}] + guided:
                p = out_dir / "rows" / f"{mcfg['name']}__{wm_name}__{s.id}.json"
                if p.exists():
                    continue
                if mcfg["name"] == "baseline":
                    out = ed.edit(clean, s.plan[0], seed)
                elif mcfg["name"] == "reembed_forced":
                    out = wm.embed(ed.edit(clean, s.plan[0], seed), bits)
                else:
                    m = build_method({**mcfg, "payload": "oracle"}, ed, wm)
                    out, _ = m.run(clean, s.plan[0], seed, {"oracle_payload": bits})
                img_p = out_dir / "images" / mcfg["name"] / wm_name / f"{s.id}.png"
                img_p.parent.mkdir(parents=True, exist_ok=True)
                out.save(img_p)
                write_json(p, {"method": mcfg["name"], "watermark": wm_name, "id": s.id, **wm.score(out, bits, fpr)})
            logging.info("forgery %s %s done", wm_name, s.id)
    import json

    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "rows").glob("*.json"))]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rows.csv", index=False)
    print(df.groupby(["watermark", "method"])[["detected", "bit_acc"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
