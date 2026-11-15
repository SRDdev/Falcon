"""PIE-Bench++ loading and the deterministic sequential-editing prompt plan.

Round 1 uses the image's own PIE-Bench++ edit (instruction derived by template from `edit_action`).
Rounds 2..R use a fixed pool of image-agnostic edits, one category per round, chosen with a seed
derived from the image id. Every prompt is written to a machine-readable manifest.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

PIE_REPO = "UB-CVML-Group/PIE_Bench_pp"
PIE_SUBSETS = {  # subset dir -> category name (0_random mixes several edit types per image)
    "0_random_140": "random",
    "1_change_object_80": "change_object",
    "2_add_object_80": "add_object",
    "3_delete_object_80": "delete_object",
    "4_change_attribute_content_40": "change_content",
    "5_change_attribute_pose_40": "change_pose",
    "6_change_attribute_color_40": "change_color",
    "7_change_attribute_material_40": "change_material",
    "8_change_background_80": "change_background",
    "9_change_style_80": "change_style",
}

# Image-agnostic follow-up edits. (category, instruction for instruction-based editors,
# caption suffix appended to the running target caption for caption-based editors / CLIP metrics)
FOLLOWUP_POOL: dict[str, list[tuple[str, str]]] = {
    "global_attribute": [
        ("make it nighttime", "at night"),
        ("change the season to winter", "in winter with snow"),
        ("make it sunset lighting", "at sunset"),
        ("make it rainy", "in the rain"),
    ],
    "background": [
        ("replace the background with a beach", "on a beach"),
        ("make the background a city street", "on a city street"),
        ("make the background a forest", "in a forest"),
        ("make the background a desert", "in a desert"),
    ],
    "style": [
        ("convert it to an oil painting", "oil painting"),
        ("make it anime style", "anime style"),
        ("turn it into a watercolor painting", "watercolor painting"),
        ("make it look cinematic", "cinematic film still"),
    ],
}
FOLLOWUP_ORDER = ["global_attribute", "background", "style"]  # round 2, 3, 4 (then cycles)


@dataclass
class EditStep:
    round: int
    category: str
    instruction: str          # for instruction-based editors (InstructPix2Pix)
    source_caption: str       # caption of the input to this round
    target_caption: str       # caption of the desired output of this round


@dataclass
class Sample:
    id: str
    subset: str
    category: str
    source_caption: str
    plan: list[EditStep] = field(default_factory=list)
    edit_mask_frac: float | None = None  # PIE-Bench++ round-1 edit region size


def clean_target(target_prompt: str) -> str:
    return re.sub(r"\s+", " ", target_prompt.replace("[", "").replace("]", "")).strip()


def instruction_from_action(edit_action: str) -> str:
    """Deterministic template: '+' -> add, '-' -> remove, otherwise change X to Y."""
    def strip(p: str) -> str:  # PIE-Bench phrases carry connectives: "and scarf", "of yellow flowers"
        p = re.sub(r"^(and|with|of|on|in|wearing|holding|surrounded by)\s+", "", p.strip())
        return re.sub(r"\s+(of|of a|of an)$", "", p)

    def bg(p: str) -> str:
        return strip(p.replace("background", "")).strip() or p

    def det(p: str) -> str:
        return p if p.startswith(("a ", "an ", "the ", "two ", "one ")) else f"the {p}"

    parts, styles = [], []
    for word, spec in json.loads(edit_action).items():
        a, t = spec["action"], spec["edit_type"]
        if t == 9:  # style: collect, emit once
            styles.append(strip(word))
        elif a == "-":
            parts.append(f"remove {det(strip(word))}")
        elif t == 8:
            parts.append(f"change the background to {bg(word)}")
        elif a == "+" or word.startswith(("with ", "and ")):
            parts.append(f"add {strip(word)}" if t in (1, 2) else f"make it {strip(word)}")
        else:
            parts.append(f"change {det(a)} to {strip(word)}")
    if styles:
        parts.append("make it " + " ".join(dict.fromkeys(styles)))
    return " and ".join(parts)


def decode_rle_mask(rle: str, h: int = 512, w: int = 512) -> np.ndarray:
    """PIE-Bench mask: space-separated (start, length) pairs over the flattened image."""
    nums = [int(x) for x in rle.split()]
    flat = np.zeros(h * w, dtype=np.uint8)
    for s, l in zip(nums[0::2], nums[1::2]):
        flat[s:s + l] = 1
    return flat.reshape(h, w)


def build_plan(sample_id: str, source_caption: str, target_caption: str, instruction: str,
               category: str, rounds: int, seed: int) -> list[EditStep]:
    from .utils import stable_seed

    rng = np.random.default_rng(stable_seed("plan", seed, sample_id))
    plan = [EditStep(1, category, instruction, source_caption, target_caption)]
    caption = target_caption
    for r in range(2, rounds + 1):
        cat = FOLLOWUP_ORDER[(r - 2) % len(FOLLOWUP_ORDER)]
        instr, suffix = FOLLOWUP_POOL[cat][rng.integers(len(FOLLOWUP_POOL[cat]))]
        new_caption = f"{caption}, {suffix}"
        plan.append(EditStep(r, cat, instr, caption, new_caption))
        caption = new_caption
    return plan


def load_pie_bench(cache_dir: Path, subsets: list[str] | None, per_subset: int | None,
                   max_images: int | None, rounds: int, seed: int, skip_per_subset: int = 0) -> tuple[list[Sample], dict[str, Image.Image], dict[str, np.ndarray]]:
    """Returns samples, id->image, id->edit mask. Selection is deterministic (first N per subset by id)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    subsets = subsets or [s for s in PIE_SUBSETS if s != "0_random_140"]
    per_sub: list[list[Sample]] = []
    images, masks = {}, {}
    for sub in subsets:
        samples = []
        path = hf_hub_download(PIE_REPO, f"{sub}/V1-00000-of-00001.parquet", repo_type="dataset",
                               cache_dir=str(cache_dir))
        df = pd.read_parquet(path).sort_values("id").iloc[skip_per_subset:]  # skip = held-out dev images
        if per_subset:
            df = df.head(per_subset)
        for row in df.itertuples():
            sid = str(row.id)
            img = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
            m = decode_rle_mask(row.mask, img.height, img.width)
            tgt = clean_target(row.target_prompt)
            cat = PIE_SUBSETS[sub]
            s = Sample(sid, sub, cat, row.source_prompt,
                       build_plan(sid, row.source_prompt, tgt, instruction_from_action(row.edit_action),
                                  cat, rounds, seed), float(m.mean()))
            samples.append(s)
            images[sid] = img
            masks[sid] = m
        per_sub.append(samples)
    # round-robin across categories so any prefix (max_images) stays category-balanced
    samples = [s for group in _zip_longest(per_sub) for s in group]
    if max_images:
        samples = samples[:max_images]
    keep = {s.id for s in samples}
    return samples, {k: v for k, v in images.items() if k in keep}, {k: v for k, v in masks.items() if k in keep}


def _zip_longest(groups: list[list]) -> list[list]:
    n = max((len(g) for g in groups), default=0)
    return [[g[i] for g in groups if i < len(g)] for i in range(n)]


def write_prompt_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
