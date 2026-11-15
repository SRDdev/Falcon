"""Configuration-driven, resumable experiment pipeline.

Stages (each skips work whose output already exists, so any interrupted run can be resumed by
re-running the same command):
  1. embed    original -> L0 watermarked image (+ payload)            images/_wm/<wm>/<id>.png
  2. edit     L(r-1) -> L(r), r = 1..R, per (method, editor, wm)     images/<method>/<editor>/<wm>/<id>/L<r>.png
  3. evaluate watermark detection + image metrics for every level     results/rows/*.json -> results/rows.csv

`wm: none` is the unwatermarked control track: its edits are the B0 editor baseline, and decoding
every watermark on them gives the empirical false-positive rate.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from .data import EditStep, Sample, load_pie_bench, write_prompt_manifest
from .utils import RESEARCH_ROOT, environment_manifest, seed_everything, stable_seed, write_json

log = logging.getLogger("wmedit")


class Experiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))
        self.root = RESEARCH_ROOT / cfg.get("output_dir", "experiments") / cfg["name"]
        self.img_dir = self.root / "images"
        self.res_dir = self.root / "results"
        self.rounds = int(cfg["dataset"].get("rounds", 4))
        self.wm_cfgs = {w["name"]: w for w in cfg["watermarks"]}
        self.ed_cfgs = {e["name"]: e for e in cfg["editors"]}
        self.methods = {m["name"]: m for m in cfg.get("methods", [{"name": "baseline"}])}
        self._wm_cache: dict = {}

    # ---------------------------------------------------------------- setup
    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        env = environment_manifest()
        env["config_name"] = self.cfg["name"]
        write_json(self.root / f"manifest_{time.strftime('%Y%m%d-%H%M%S')}.json", env)
        d = self.cfg["dataset"]
        self.samples, self.images, self.masks = load_pie_bench(
            RESEARCH_ROOT / "datasets" / "hf_cache", d.get("subsets"), d.get("per_subset"),
            d.get("max_images"), self.rounds, self.seed, d.get("skip_per_subset", 0))
        if d.get("source") == "generated":  # generated-image track: originals are clean SD-1.5 generations
            self._generate_originals()
        write_prompt_manifest(self.samples, self.root / "prompts.jsonl")
        log.info("prepared %d samples -> %s", len(self.samples), self.root)

    def gen_seed(self, sid: str) -> int:
        return stable_seed("gen", self.seed, sid)

    def _generate_originals(self):
        """Clean generation from the source caption with the same base noise the watermarked ones start from."""
        from .genwm import TreeRing

        gen = None
        for s in self.samples:
            p = self.img_dir / "_orig" / f"{s.id}.png"
            if not p.exists():
                gen = gen or TreeRing()
                p.parent.mkdir(parents=True, exist_ok=True)
                gen.generate_clean(s.source_caption, self.gen_seed(s.id)).save(p)
            self.images[s.id] = Image.open(p).convert("RGB")
        self.masks = {k: np.zeros_like(v) for k, v in self.masks.items()}  # PIE masks don't apply to generations

    def watermark(self, name: str):
        if name == "none":
            return None
        if name not in self._wm_cache:
            from .genwm import GEN_REGISTRY
            from .watermarks import build_watermark

            kw = {k: v for k, v in self.wm_cfgs.get(name, {"name": name}).items() if k != "name"}
            self._wm_cache[name] = GEN_REGISTRY[name](**kw) if name in GEN_REGISTRY else build_watermark(name, **kw)
        return self._wm_cache[name]

    def payload(self, wm, sid: str) -> np.ndarray:
        if wm.nbits == 0:  # zero-bit watermark (Tree-Ring)
            return np.zeros(0, dtype=np.uint8)
        rng = np.random.default_rng(stable_seed("payload", self.seed, wm.name, sid))
        return rng.integers(0, 2, wm.nbits).astype(np.uint8)

    def l0_path(self, wm_name: str, sid: str) -> Path:
        return self.img_dir / "_wm" / wm_name / f"{sid}.png"

    def level_path(self, method: str, editor: str, wm_name: str, sid: str, r: int) -> Path:
        if r == 0:
            return self.l0_path(wm_name, sid)
        return self.img_dir / method / editor / wm_name / sid / f"L{r}.png"

    # ---------------------------------------------------------------- stage 1
    def embed(self):
        for wm_name in self.wm_cfgs:
            wm = self.watermark(wm_name)
            for s in self.samples:
                p = self.l0_path(wm_name, s.id)
                if p.exists():
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                img = self.images[s.id]
                if wm is None:
                    out = img
                elif getattr(wm, "generative", False):
                    out = wm.generate(s.source_caption, self.payload(wm, s.id), self.gen_seed(s.id))
                else:
                    out = wm.embed(img, self.payload(wm, s.id))
                out.save(p)
        log.info("embed stage done")

    # ---------------------------------------------------------------- stage 2
    def edit(self):
        from .editors import build_editor
        from .methods import build_method

        for ed_name, ed_cfg in self.ed_cfgs.items():
            todo = [(m, w, s) for m in self.methods for w in self.wm_cfgs for s in self.samples
                    if self._applicable(m, w) and not self.level_path(m, ed_name, w, s.id, self.rounds).exists()]
            if not todo:
                continue
            editor = build_editor(ed_name, **{k: v for k, v in ed_cfg.items() if k not in ("name", "mechanism", "split")})
            t0, n = time.time(), 0
            for m_name, wm_name, s in todo:
                method = build_method(self.methods[m_name], editor, self.watermark(wm_name))
                prev = Image.open(self.l0_path(wm_name, s.id)).convert("RGB")
                state: dict = {}
                for step in s.plan:
                    out_p = self.level_path(m_name, ed_name, wm_name, s.id, step.round)
                    if out_p.exists():
                        prev = Image.open(out_p).convert("RGB")
                        continue
                    seed = stable_seed("edit", self.seed, s.id, step.round)  # shared across wm/method: paired design
                    seed_everything(seed)
                    t1 = time.time()
                    out, info = method.run(prev, step, seed, state)
                    info["seconds"] = time.time() - t1
                    info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out.save(out_p)
                    write_json(out_p.with_suffix(".json"), info)
                    prev, n = out, n + 1
                log.info("[%s/%s/%s] %s done (%d edits, %.1fs/edit)", ed_name, m_name, wm_name, s.id, n,
                         (time.time() - t0) / max(n, 1))
            del editor
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _applicable(method: str, wm_name: str) -> bool:
        return method == "baseline" or wm_name != "none"

    # ---------------------------------------------------------------- stage 3
    def evaluate(self, image_metrics: bool = True):
        from .metrics import ImageMetrics, image_quality

        ev = self.cfg.get("evaluation", {})
        fpr = float(ev.get("target_fpr", 1e-3))
        metrics = ImageMetrics() if image_metrics and ev.get("image_metrics", True) else None
        rows_dir = self.res_dir / "rows"
        detectors = [w for w in self.wm_cfgs if w != "none"]
        for m_name in self.methods:
            for ed_name in self.ed_cfgs:
                for wm_name in self.wm_cfgs:
                    if not self._applicable(m_name, wm_name):
                        continue
                    for s in self.samples:
                        prev, prev_caption = None, s.source_caption
                        orig = self.images[s.id]
                        for r in range(0, self.rounds + 1):
                            key = f"{m_name}__{ed_name}__{wm_name}__{s.id}__L{r}"
                            row_p = rows_dir / f"{key}.json"
                            p = self.level_path(m_name, ed_name, wm_name, s.id, r)
                            if not p.exists():
                                break
                            img = Image.open(p).convert("RGB")
                            step: EditStep | None = s.plan[r - 1] if r else None
                            if not row_p.exists():
                                row = {"method": m_name, "editor": ed_name, "watermark": wm_name,
                                       "id": s.id, "category": s.category, "level": r,
                                       "round_category": step.category if step else "none",
                                       "instruction": step.instruction if step else ""}
                                # watermark detection: own payload, or (control track) every detector
                                for det in ([wm_name] if wm_name != "none" else detectors):
                                    wm = self.watermark(det)
                                    sc = wm.score(img, self.payload(wm, s.id), fpr)
                                    pre = "" if wm_name != "none" else f"{det}_"
                                    row.update({f"{pre}{k}": v for k, v in sc.items()})
                                    if det == "wam":
                                        row[f"{pre}wam_mask_frac"] = wm.mask_fraction(img)
                                if r == 0:
                                    row.update({f"wm_{k}": v for k, v in image_quality(orig, img).items()})
                                elif metrics is not None:
                                    keep = (1 - self.masks[s.id]) if r == 1 else None
                                    row.update(metrics.pair(prev, img, step.source_caption, step.target_caption, keep))
                                    # cumulative drift from the clean original
                                    row["psnr_vs_orig"] = image_quality(orig, img)["psnr"]
                                info_p = p.with_suffix(".json")
                                if info_p.exists():
                                    row.update({f"edit_{k}": v for k, v in json.loads(info_p.read_text()).items()
                                                if not isinstance(v, (list, dict))})
                                write_json(row_p, row)
                            prev = img
        self.collect()

    def collect(self) -> pd.DataFrame:
        rows = [json.loads(p.read_text()) for p in sorted((self.res_dir / "rows").glob("*.json"))]
        df = pd.DataFrame(rows)
        df.to_csv(self.res_dir / "rows.csv", index=False)
        log.info("collected %d rows -> %s", len(df), self.res_dir / "rows.csv")
        return df


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:  # simple inheritance: base config, then this file's keys
        base = load_config(Path(path).parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    for o in overrides or []:  # dotted overrides: dataset.max_images=5
        k, v = o.split("=", 1)
        d = cfg
        *head, last = k.split(".")
        for h in head:
            d = d.setdefault(h, {})
        d[last] = yaml.safe_load(v)
    return cfg
