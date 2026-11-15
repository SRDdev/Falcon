"""Smoke test for guided editing on one image: lam=0 equivalence, memory, effect. Not a result."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, torch
from wmedit.data import load_pie_bench
from wmedit.editors import InstructPix2Pix
from wmedit.methods import build_method
from wmedit.utils import RESEARCH_ROOT, pil_to_tensor
from wmedit.watermarks import build_watermark

s, imgs, _ = load_pie_bench(RESEARCH_ROOT / "datasets" / "hf_cache", ["8_change_background_80"], 1, 1, 1, 42)
s = s[0]; ed = InstructPix2Pix(steps=50)
for wn in ["wam", "trustmark"]:
    wm = build_watermark(wn); bits = np.random.default_rng(0).integers(0, 2, wm.nbits).astype(np.uint8)
    l0 = wm.embed(imgs[s.id], bits)
    base = ed.edit(l0, s.plan[0], 7)
    print(wn, "baseline acc", (wm.decode(base) == bits).mean())
    for lam in [0.0, 0.1, 0.3, 1.0]:
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        m = build_method({"name": "guided", "lam": lam, "start_frac": 0.6, "every": 2}, ed, wm)
        out, info = m.run(l0, s.plan[0], 7, {})
        d = (pil_to_tensor(out) - pil_to_tensor(base)).abs().mean().item() * 255
        print(f"  lam={lam} acc={(wm.decode(out)==bits).mean():.3f} mean|out-base|={d:.2f}/255 "
              f"loss {info['wm_loss_first']}->{info['wm_loss_last']} {time.time()-t:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
