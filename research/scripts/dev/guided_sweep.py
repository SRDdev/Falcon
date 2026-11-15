"""Dev sweep: does any guidance setting rescue the watermark on images where baseline loses it? Not a result."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from PIL import Image
from wmedit.editors import InstructPix2Pix
from wmedit.methods import build_method
from wmedit.pipeline import Experiment, load_config
from wmedit.utils import pil_to_tensor, stable_seed

e = Experiment(load_config("configs/pilot.yaml")); e.prepare()
ed = InstructPix2Pix(steps=50)
wm = e.watermark("wam")
for sid in ["111000000000", "311000000000"]:  # pilot: WAM lost at L1 under IP2P baseline
    s = next(x for x in e.samples if x.id == sid)
    l0 = Image.open(e.l0_path("wam", sid)).convert("RGB"); bits = e.payload(wm, sid)
    seed = stable_seed("edit", e.seed, sid, 1)
    base = ed.edit(l0, s.plan[0], seed); print(sid, s.plan[0].instruction, "| baseline acc", (wm.decode(base) == bits).mean())
    for lam, start, every in [(0.005, 0.4, 1), (0.01, 0.4, 1), (0.02, 0.4, 1), (0.05, 0.4, 1), (0.02, 0.8, 1), (0.05, 0.8, 1), (0.1, 0.9, 1)]:
        m = build_method({"name": "guided", "lam": lam, "start_frac": start, "every": every, "payload": "oracle"}, ed, wm)
        out, info = m.run(l0, s.plan[0], seed, {"oracle_payload": bits})
        d = (pil_to_tensor(out) - pil_to_tensor(base)).abs().mean().item() * 255
        print(f"  lam={lam} start={start} acc={(wm.decode(out)==bits).mean():.3f} maskfrac={wm.mask_fraction(out):.2f} "
              f"|out-base|={d:.1f} loss {info['wm_loss_first']:.3f}->{info['wm_loss_last']:.3f}")
