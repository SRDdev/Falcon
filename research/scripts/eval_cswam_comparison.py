#!/usr/bin/env python
"""Offline Evaluation & Comparison: Consensus-Saliency WAM (CS-WAM) vs Vanilla WAM.

Evaluates CS-WAM's Two-Stage Spatial Consensus Erasure Decoder directly against
existing edited images without requiring re-running diffusion editing steps.

Usage:
    python research/scripts/eval_cswam_comparison.py \
        --exp experiments/saliency_hybrid \
        --editor ip2p \
        --source-wm wam
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
sys.path.insert(0, str(RESEARCH_DIR / "src"))

from wmedit.utils import stable_seed
from wmedit.watermarks import WAM, ConsensusSaliencyWAM, detection_threshold, p_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_cswam")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CS-WAM vs Vanilla WAM on existing experiment images.")
    parser.add_argument("--exp", type=str, default="experiments/saliency_hybrid", help="Experiment root directory")
    parser.add_argument("--editor", type=str, default="ip2p", choices=["ip2p", "sdedit", "all"], help="Editor to evaluate")
    parser.add_argument("--source-wm", type=str, default="wam", help="Watermark run to evaluate (default: wam)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for ground-truth payload reconstruction")
    parser.add_argument("--target-fpr", type=float, default=1e-3, help="FPR threshold for detection")
    parser.add_argument("--logit-temp", type=float, default=2.0, help="Logit temperature for CS-WAM")
    parser.add_argument("--top-k-tiles", type=int, default=16, help="Top-K consensus tiles for CS-WAM")
    parser.add_argument("--trim-frac", type=float, default=0.125, help="Trimmed vote fraction for CS-WAM")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save comparison CSV")
    return parser.parse_args()


def get_ground_truth_payload(seed: int, sid: str, wm_name: str, nbits: int = 32) -> np.ndarray:
    """Reconstruct exact deterministic ground-truth payload for a sample ID."""
    rng = np.random.default_rng(stable_seed("payload", seed, wm_name, sid))
    return rng.integers(0, 2, nbits).astype(np.uint8)


def evaluate_decoders(
    exp_dir: Path,
    editors: list[str],
    source_wm: str,
    seed: int = 42,
    target_fpr: float = 1e-3,
    logit_temp: float = 2.0,
    top_k_tiles: int = 16,
    trim_frac: float = 0.125,
    device: str | None = None,
) -> pd.DataFrame:
    log.info("Initializing Vanilla WAM detector...")
    wam_vanilla = WAM(device=device)

    log.info("Initializing Consensus-Saliency WAM (CS-WAM) detector...")
    wam_cs = ConsensusSaliencyWAM(
        logit_temp=logit_temp,
        top_k_tiles=top_k_tiles,
        trim_frac=trim_frac,
        device=device,
    )

    k_det = detection_threshold(32, target_fpr)
    log.info("Evaluation threshold for 32 bits at FPR=%g is %d matching bits (%.1f%%)",
             target_fpr, k_det, (k_det / 32) * 100)

    rows = []
    images_root = exp_dir / "images"
    l0_dir = images_root / "_wm" / source_wm

    if not l0_dir.exists():
        raise FileNotFoundError(f"L0 watermarked images not found at {l0_dir}")

    sample_ids = sorted([p.stem for p in l0_dir.glob("*.png")])
    log.info("Found %d samples in %s", len(sample_ids), l0_dir)

    for ed in editors:
        log.info("=== Evaluating editor: %s ===", ed)
        for sid in sample_ids:
            gt_bits = get_ground_truth_payload(seed, sid, source_wm, nbits=32)

            for lvl in range(0, 5):
                if lvl == 0:
                    img_path = l0_dir / f"{sid}.png"
                else:
                    img_path = images_root / "baseline" / ed / source_wm / sid / f"L{lvl}.png"

                if not img_path.exists():
                    log.warning("Image missing: %s", img_path)
                    continue

                img = Image.open(img_path).convert("RGB")

                # Vanilla WAM decode
                bits_vanilla = wam_vanilla.decode(img)
                matches_vanilla = int((bits_vanilla == gt_bits).sum())
                acc_vanilla = matches_vanilla / 32.0
                det_vanilla = int(matches_vanilla >= k_det)

                # CS-WAM decode
                bits_cs = wam_cs.decode(img)
                matches_cs = int((bits_cs == gt_bits).sum())
                acc_cs = matches_cs / 32.0
                det_cs = int(matches_cs >= k_det)

                rows.append({
                    "editor": ed,
                    "source_wm": source_wm,
                    "sample_id": sid,
                    "level": lvl,
                    "bit_acc_vanilla": acc_vanilla,
                    "det_vanilla": det_vanilla,
                    "bit_acc_cswam": acc_cs,
                    "det_cswam": det_cs,
                    "matches_vanilla": matches_vanilla,
                    "matches_cswam": matches_cs,
                    "gain_bit_acc": acc_cs - acc_vanilla,
                    "gain_det": det_cs - det_vanilla,
                })

    df = pd.DataFrame(rows)
    return df


def display_comparison_summary(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  CONSENSUS-SALIENCY WAM (CS-WAM) VS VANILLA WAM DECODER COMPARISON")
    print("=" * 80)

    for ed, g_ed in df.groupby("editor"):
        print(f"\n--- Editor: {ed.upper()} ---")
        summary = g_ed.groupby("level").agg({
            "bit_acc_vanilla": "mean",
            "bit_acc_cswam": "mean",
            "gain_bit_acc": "mean",
            "det_vanilla": "mean",
            "det_cswam": "mean",
            "gain_det": "mean",
        }).reset_index()

        summary.columns = [
            "Level",
            "WAM BitAcc",
            "CS-WAM BitAcc",
            "Δ BitAcc",
            "WAM DetRate",
            "CS-WAM DetRate",
            "Δ DetRate",
        ]

        # Format percentages
        summary["WAM BitAcc"] = (summary["WAM BitAcc"] * 100).map("{:.2f}%".format)
        summary["CS-WAM BitAcc"] = (summary["CS-WAM BitAcc"] * 100).map("{:.2f}%".format)
        summary["Δ BitAcc"] = summary["Δ BitAcc"].map(lambda x: f"{x * 100:+.2f}%")
        summary["WAM DetRate"] = (summary["WAM DetRate"] * 100).map("{:.1f}%".format)
        summary["CS-WAM DetRate"] = (summary["CS-WAM DetRate"] * 100).map("{:.1f}%".format)
        summary["Δ DetRate"] = summary["Δ DetRate"].map(lambda x: f"{x * 100:+.1f}%")

        print(summary.to_string(index=False))

    print("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    exp_dir = Path(args.exp)
    if not exp_dir.is_absolute():
        exp_dir = RESEARCH_DIR / exp_dir

    editors = ["ip2p", "sdedit"] if args.editor == "all" else [args.editor]

    df = evaluate_decoders(
        exp_dir=exp_dir,
        editors=editors,
        source_wm=args.source_wm,
        seed=args.seed,
        target_fpr=args.target_fpr,
        logit_temp=args.logit_temp,
        top_k_tiles=args.top_k_tiles,
        trim_frac=args.trim_frac,
        device=args.device,
    )

    display_comparison_summary(df)

    out_csv = Path(args.output_csv) if args.output_csv else exp_dir / "results" / "cswam_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Saved detailed comparison rows to %s", out_csv)


if __name__ == "__main__":
    main()
