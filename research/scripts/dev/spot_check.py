"""Ad-hoc: print bit accuracy per level for finished sequences (* = detected). Not used for results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from PIL import Image
from wmedit.pipeline import Experiment, load_config

cfg, method, editor, wmn = sys.argv[1:5]
e = Experiment(load_config(cfg)); wm = e.watermark(wmn)
for d in sorted((e.img_dir / method / editor / wmn).iterdir()):
    out = []
    for r in range(e.rounds + 1):
        p = e.level_path(method, editor, wmn, d.name, r)
        if p.exists():
            sc = wm.score(Image.open(p).convert("RGB"), e.payload(wm, d.name), 1e-3)
            out.append(f"{sc['bit_acc']:.2f}{'*' if sc['detected'] else ' '}")
    print(d.name, " ".join(out))
