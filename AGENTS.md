# Repository Guidelines

## Project Structure & Module Organization

Falcon is a Python research pipeline for editing and watermark-robustness experiments. Dataset loading lives in `data/`; reusable configuration and logging code lives in `configs/`; orchestration and helpers live in `pipeline/`; model wrappers live in `models/editing/` and `models/watermarking/`. Shell setup scripts are in `scripts/`, external model repositories are expected under `external/`, and runtime artifacts are written to `experiments/` and `logs/`. Keep large datasets, checkpoints, and generated images out of source directories and commits.

## Build, Test, and Development Commands

Use the Conda environment defined by `environment.yml` (Python 3.10):

```bash
bash setup.sh                         # Create/configure the Falcon environment
conda activate Falcon
python pipeline/run_single.py         # Run the configured sample pipeline
python -m compileall data configs models pipeline  # Syntax smoke check
```

`setup.sh` also prepares InstructPix2Pix; run `bash scripts/setup_watermarkanything.sh Falcon` when WatermarkAnything dependencies and checkpoints are required. The pipeline needs the external submodule, model weights, network access for the Hugging Face dataset, and a CUDA-capable PyTorch installation for practical execution.

## Coding Style & Naming Conventions

Use 4-space indentation, readable `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Prefer small wrappers and configuration-driven behavior over hard-coded paths or parameters. Match the existing logging pattern (`setup_logger`) and use type hints for new utility interfaces where practical. No formatter or linter is configured; keep imports organized and preserve the surrounding style.

## Testing Guidelines

There is currently no automated test suite or coverage threshold. Before submitting changes, run the compile smoke check above and, when model dependencies are available, execute the affected pipeline path with a small configured dataset limit. Inspect generated JSON, images, and log output under `experiments/` and `logs/` rather than committing them.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects consistent with project history, for example `Add watermark detection metric` or `Refactor pipeline configuration`. Pull requests should explain the experiment or behavior change, identify configuration and dependency changes, include reproducible commands, and link any relevant issue. For output or visualization changes, attach representative before/after images or metric summaries; do not include model weights or large generated artifacts.

## Security & Configuration Tips

Keep credentials, private datasets, and checkpoint files local. Review `configs/config.yaml` before running experiments, especially dataset limits and checkpoint paths, and avoid silently changing reproducibility-critical seeds or model parameters.
