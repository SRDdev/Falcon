# Reproduce

Every number in `tables/` and every plot in `figures/` is generated from `experiments/<name>/results/*.csv`.
Every stage is resumable: re-running a command skips work whose outputs exist.

```bash
pytest -m "not gpu"                 # fast unit tests (CPU)
pytest -m gpu                       # model-backed smoke tests: watermark round-trips, clean-image negatives,
                                    # decoder gradients, custom IP2P loop == official diffusers pipeline

python run_experiment.py --config configs/pilot.yaml        # Stage 1: 10 images, pipeline validation
python run_probes.py      --config configs/probes.yaml      # failure-mode probes, 27 images
bash scripts/run_ours.sh                                     # guided-editing falsification test, 18 held-out images
bash scripts/run_baseline.sh                                 # Stage 2: 90 images
bash scripts/generate_tables.sh && bash scripts/generate_figures.sh
```

Overrides without editing files: `python run_experiment.py --config configs/stage2.yaml --set dataset.max_images=5`.

## Experimental design
- **Paired**: the same images, prompts and editor seed (`stable_seed("edit", seed, image_id, round)`) are used
  for every watermark and method, so differences are attributable to the watermark/method.
- **Prompts**: round 1 = the image's own PIE-Bench++ edit (instruction templated from `edit_action`, caption =
  PIE-Bench++ target prompt); rounds 2–4 = global attribute → background → style, drawn per image from a fixed pool
  with a seeded RNG. All prompts: `experiments/<name>/prompts.jsonl`.
- **Payloads**: random per (watermark, image), seeded; ground truth never leaves the evaluator. The re-embed and
  guided methods *decode* the payload from their input (they never see the ground truth).
- **Detection**: one-sided binomial test at FPR 1e-3 (k=32 → ≥26 matching bits; k=100 → ≥66). Empirical FPR is
  measured on the unwatermarked track for every detector, editor and round.
- Pre-registered definitions: `analysis/PREREGISTRATION.md`.

## Moving results
The GPU host has no persistent volume. `scripts/sync.sh pull` copies experiments (images, JSON, CSV, logs) back.
