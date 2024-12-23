from ml_collections import config_dict
import yaml
from diffusers.schedulers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDPMScheduler,
)
from utils import (
    deterministic_ddim_step,
    deterministic_ddpm_step,
    deterministic_euler_step,
    deterministic_non_ancestral_euler_step,
)

BREAKDOWNS = ["x_t_c_hat", "x_t_hat_c", "no_breakdown", "x_t_hat_c_with_zeros"]
SCHEDULERS = ["ddpm", "ddim", "euler", "euler_non_ancestral"]
MODELS = [
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "CompVis/stable-diffusion-v1-4",
]

def get_num_steps_actual(cfg):
    return (
        cfg.num_steps_inversion
        - cfg.step_start
        + (1 if cfg.clean_step_timestep > 0 else 0)
        if cfg.timesteps is None
        else len(cfg.timesteps) + (1 if cfg.clean_step_timestep > 0 else 0)
    )


def get_config(args):
    if args.config_from_file and args.config_from_file != "":
        with open(args.config_from_file, "r") as f:
            cfg = config_dict.ConfigDict(yaml.safe_load(f))

        num_steps_actual = get_num_steps_actual(cfg)

    else:
        cfg = config_dict.ConfigDict()

        cfg.seed = 2
        cfg.self_r = 0.5
        cfg.cross_r = 0.9
        cfg.eta = 1
        cfg.scheduler_type = SCHEDULERS[0]

        cfg.num_steps_inversion = 50  # timesteps: 999, 799, 599, 399, 199
        cfg.step_start = 20
        cfg.timesteps = None
        cfg.noise_timesteps = None
        num_steps_actual = get_num_steps_actual(cfg)
        cfg.ws1 = [2] * num_steps_actual
        cfg.ws2 = [1] * num_steps_actual
        cfg.real_cfg_scale = 0
        cfg.real_cfg_scale_save = 0
        cfg.breakdown = BREAKDOWNS[1]
        cfg.noise_shift_delta = 1
        cfg.max_norm_zs = [-1] * (num_steps_actual - 1) + [15.5]

        cfg.clean_step_timestep = 0

        cfg.model = MODELS[1]

    if cfg.scheduler_type == "ddim":
        cfg.scheduler_class = DDIMScheduler
        cfg.step_function = deterministic_ddim_step
    elif cfg.scheduler_type == "ddpm":
        cfg.scheduler_class = DDPMScheduler
        cfg.step_function = deterministic_ddpm_step
    elif cfg.scheduler_type == "euler":
        cfg.scheduler_class = EulerAncestralDiscreteScheduler
        cfg.step_function = deterministic_euler_step
    elif cfg.scheduler_type == "euler_non_ancestral":
        cfg.scheduler_class = EulerDiscreteScheduler
        cfg.step_function = deterministic_non_ancestral_euler_step
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler_type}")

    with cfg.ignore_type():
        if isinstance(cfg.max_norm_zs, (int, float)):
            cfg.max_norm_zs = [cfg.max_norm_zs] * num_steps_actual

        if isinstance(cfg.ws1, (int, float)):
            cfg.ws1 = [cfg.ws1] * num_steps_actual

        if isinstance(cfg.ws2, (int, float)):
            cfg.ws2 = [cfg.ws2] * num_steps_actual

    if not hasattr(cfg, "update_eta"):
        cfg.update_eta = False

    if not hasattr(cfg, "save_timesteps"):
        cfg.save_timesteps = None

    if not hasattr(cfg, "scheduler_timesteps"):
        cfg.scheduler_timesteps = None

    assert (
        cfg.scheduler_type == "ddpm" or cfg.timesteps is None
    ), "timesteps must be None for ddim/euler"

    assert (
        len(cfg.max_norm_zs) == num_steps_actual
    ), f"len(cfg.max_norm_zs) ({len(cfg.max_norm_zs)}) != num_steps_actual ({num_steps_actual})"

    assert (
        len(cfg.ws1) == num_steps_actual
    ), f"len(cfg.ws1) ({len(cfg.ws1)}) != num_steps_actual ({num_steps_actual})"

    assert (
        len(cfg.ws2) == num_steps_actual
    ), f"len(cfg.ws2) ({len(cfg.ws2)}) != num_steps_actual ({num_steps_actual})"

    assert cfg.noise_timesteps is None or len(cfg.noise_timesteps) == (
        num_steps_actual - (1 if cfg.clean_step_timestep > 0 else 0)
    ), f"len(cfg.noise_timesteps) ({len(cfg.noise_timesteps)}) != num_steps_actual ({num_steps_actual})"

    assert cfg.save_timesteps is None or len(cfg.save_timesteps) == (
        num_steps_actual - (1 if cfg.clean_step_timestep > 0 else 0)
    ), f"len(cfg.save_timesteps) ({len(cfg.save_timesteps)}) != num_steps_actual ({num_steps_actual})"

    return cfg


def get_config_name(config, args):
    if args.folder_name is not None and args.folder_name != "":
        return args.folder_name
    timesteps_str = (
        f"step_start {config.step_start}"
        if config.timesteps is None
        else f"timesteps {config.timesteps}"
    )
    return f"""\
ws1 {config.ws1[0]} ws2 {config.ws2[0]} real_cfg_scale {config.real_cfg_scale} {timesteps_str} \
real_cfg_scale_save {config.real_cfg_scale_save} seed {config.seed} max_norm_zs {config.max_norm_zs[-1]} noise_shift_delta {config.noise_shift_delta} \
scheduler_type {config.scheduler_type} fp16 {args.fp16}\
"""