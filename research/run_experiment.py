#!/usr/bin/env python
"""Single entry point.

  python run_experiment.py --config configs/pilot.yaml                 # all stages
  python run_experiment.py --config configs/pilot.yaml --stages embed edit
  python run_experiment.py --config configs/pilot.yaml --set dataset.max_images=5
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wmedit.pipeline import Experiment, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", default=["embed", "edit", "evaluate"],
                    choices=["embed", "edit", "evaluate", "collect"])
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. dataset.max_images=5")
    ap.add_argument("--no-image-metrics", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    exp = Experiment(cfg)
    exp.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(exp.root / "run.log")])
    if a.stages != ["collect"]:
        exp.prepare()
    for st in a.stages:
        if st == "evaluate":
            exp.evaluate(image_metrics=not a.no_image_metrics)
        else:
            getattr(exp, st)()


if __name__ == "__main__":
    main()
