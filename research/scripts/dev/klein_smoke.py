"""Smoke test FLUX.2-klein editing on 3 watermarked stage-2 images: runs, size, speed, WAM/TrustMark survival. Not a result."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from PIL import Image
from wmedit.editors import Flux2Klein
from wmedit.pipeline import Experiment, load_config
from wmedit.utils import stable_seed

e = Experiment(load_config("configs/stage2.yaml")); e.prepare()
ed = Flux2Klein(cpu_offload="--offload" in sys.argv)
out_dir = e.root.parent / "dev_klein"; out_dir.mkdir(exist_ok=True)
for s in [e.samples[i] for i in (0, 4, 8)]:
    for wn in ["wam", "trustmark"]:
        wm = e.watermark(wn); l0 = Image.open(e.l0_path(wn, s.id)).convert("RGB")
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        out = ed.edit(l0, s.plan[0], stable_seed("edit", e.seed, s.id, 1))
        dt = time.time() - t
        out.save(out_dir / f"{s.id}_{wn}.png")
        sc = wm.score(out, e.payload(wm, s.id), 1e-3)
        print(f"{s.id} {s.category:16s} {wn:9s} {out.size} {dt:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB "
              f"acc={sc['bit_acc']:.2f} det={sc['detected']} | {s.plan[0].instruction}")
