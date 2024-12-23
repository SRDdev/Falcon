import itertools
from typing import List, Optional, Union
import PIL
import PIL.Image
import torch
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw, ImageFont
import os
from diffusers.utils import (
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
)
from diffusers.image_processor import VaeImageProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers import DiffusionPipeline


VECTOR_DATA_FOLDER = "vector_data"
VECTOR_DATA_DICT = "vector_data"


def encode_image(image: PIL.Image, pipe: DiffusionPipeline):
    pipe.image_processor: VaeImageProcessor = pipe.image_processor  # type: ignore
    image = pipe.image_processor.pil_to_numpy(image)
    image = pipe.image_processor.numpy_to_pt(image)
    image = image.to(pipe.device)
    return (
        pipe.vae.encode(
            pipe.image_processor.preprocess(image),
        ).latent_dist.mode()
        * pipe.vae.config.scaling_factor
    )


def decode_latents(latent, pipe):
    latent_img = pipe.vae.decode(
        latent / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    return pipe.image_processor.postprocess(latent_img, output_type="pil")


def get_device(argv, args=None):
    import sys

    def debugger_is_active():
        return hasattr(sys, "gettrace") and sys.gettrace() is not None

    if args:
        return (
            torch.device("cuda")
            if (torch.cuda.is_available() and not debugger_is_active())
            and not args.force_use_cpu
            else torch.device("cpu")
        )

    return (
        torch.device("cuda")
        if (torch.cuda.is_available() and not debugger_is_active())
        and not "cpu" in set(argv[1:])
        else torch.device("cpu")
    )


def deterministic_ddim_step(
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
    scheduler=None,
):

    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )
    return prev_sample


def deterministic_euler_step(
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    eta,
    use_clipped_model_output,
    generator,
    variance_noise,
    return_dict,
    scheduler,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

    Returns:
        [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`,
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
            otherwise a tuple is returned where the first element is the sample tensor.

    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma * model_output
    elif scheduler.config.prediction_type == "v_prediction":
        # * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
            sample / (sigma**2 + 1)
        )
    elif scheduler.config.prediction_type == "sample":
        raise NotImplementedError("prediction_type not implemented yet: sample")
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )

    sigma_from = scheduler.sigmas[scheduler.step_index]
    sigma_to = scheduler.sigmas[scheduler.step_index + 1]
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma

    dt = sigma_down - sigma

    prev_sample = sample + derivative * dt

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    scheduler._step_index += 1

    return prev_sample


def deterministic_non_ancestral_euler_step(
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
    scheduler=None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
            tuple.

    Returns:
        [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
            returned, otherwise a tuple is returned where the first element is the sample tensor.
    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if not scheduler.is_scale_input_called:
        logger.warning(
            "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
            "See `StableDiffusionPipeline` for a usage example."
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    sigma = scheduler.sigmas[scheduler.step_index]

    gamma = (
        min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1)
        if s_tmin <= sigma <= s_tmax
        else 0.0
    )

    sigma_hat = sigma * (gamma + 1)

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    # NOTE: "original_sample" should not be an expected prediction_type but is left in for
    # backwards compatibility
    if (
        scheduler.config.prediction_type == "original_sample"
        or scheduler.config.prediction_type == "sample"
    ):
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma_hat * model_output
    elif scheduler.config.prediction_type == "v_prediction":
        # denoised = model_output * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
            sample / (sigma**2 + 1)
        )
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma_hat

    dt = scheduler.sigmas[scheduler.step_index + 1] - sigma_hat

    prev_sample = sample + derivative * dt

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    scheduler._step_index += 1

    return prev_sample


def deterministic_ddpm_step(
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    eta,
    use_clipped_model_output,
    generator,
    variance_noise,
    return_dict,
    scheduler,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    return pred_prev_sample


def normalize(
    z_t,
    i,
    max_norm_zs,
):
    max_norm = max_norm_zs[i]
    if max_norm < 0:
        return z_t, 1

    norm = torch.norm(z_t)
    if norm < max_norm:
        return z_t, 1

    coeff = max_norm / norm
    z_t = z_t * coeff
    return z_t, coeff


def find_index(timesteps, timestep):
    for i, t in enumerate(timesteps):
        if t == timestep:
            return i
    return -1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
map_timpstep_to_index = {
    torch.tensor(799): 0,
    torch.tensor(599): 1,
    torch.tensor(399): 2,
    torch.tensor(199): 3,
    torch.tensor(799, device=device): 0,
    torch.tensor(599, device=device): 1,
    torch.tensor(399, device=device): 2,
    torch.tensor(199, device=device): 3,
}

def step_save_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
):
    # print(self._save_timesteps)
    # timestep_index = map_timpstep_to_index[timestep]
    # timestep_index = ((self._save_timesteps == timestep).nonzero(as_tuple=True)[0]).item()
    timestep_index = self._save_timesteps.index(timestep) if not self.clean_step_run else -1
    next_timestep_index = timestep_index + 1 if not self.clean_step_run else -1
    u_hat_t = self.step_function(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        use_clipped_model_output=use_clipped_model_output,
        generator=generator,
        variance_noise=variance_noise,
        return_dict=False,
        scheduler=self,
    )

    x_t_minus_1 = self.x_ts[next_timestep_index]
    self.x_ts_c_hat.append(u_hat_t)

    z_t = x_t_minus_1 - u_hat_t
    self.latents.append(z_t)

    z_t, _ = normalize(z_t, timestep_index, self._config.max_norm_zs)

    x_t_minus_1_predicted = u_hat_t + z_t

    if not return_dict:
        return (x_t_minus_1_predicted,)

    return DDIMSchedulerOutput(prev_sample=x_t_minus_1, pred_original_sample=None)


def step_use_latents(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
):
    # timestep_index = ((self._save_timesteps == timestep).nonzero(as_tuple=True)[0]).item()
    timestep_index = self._timesteps.index(timestep) if not self.clean_step_run else -1
    next_timestep_index = (
        timestep_index + 1 if not self.clean_step_run else -1
    )
    z_t = self.latents[next_timestep_index]  # + 1 because latents[0] is X_T

    _, normalize_coefficient = normalize(
        z_t[0] if self._config.breakdown == "x_t_hat_c_with_zeros" else z_t,
        timestep_index,
        self._config.max_norm_zs,
    )

    if normalize_coefficient == 0:
        eta = 0

    # eta = normalize_coefficient

    x_t_hat_c_hat = self.step_function(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        use_clipped_model_output=use_clipped_model_output,
        generator=generator,
        variance_noise=variance_noise,
        return_dict=False,
        scheduler=self,
    )

    w1 = self._config.ws1[timestep_index]
    w2 = self._config.ws2[timestep_index]

    x_t_minus_1_exact = self.x_ts[next_timestep_index]
    x_t_minus_1_exact = x_t_minus_1_exact.expand_as(x_t_hat_c_hat)

    x_t_c_hat: torch.Tensor = self.x_ts_c_hat[next_timestep_index]
    if self._config.breakdown == "x_t_c_hat":
        raise NotImplementedError("breakdown x_t_c_hat not implemented yet")

    # x_t_c_hat = x_t_c_hat.expand_as(x_t_hat_c_hat)
    x_t_c = x_t_c_hat[0].expand_as(x_t_hat_c_hat)

    # if self._config.breakdown == "x_t_c_hat":
    #     v1 = x_t_hat_c_hat - x_t_c_hat
    #     v2 = x_t_c_hat - x_t_c
    if (
        self._config.breakdown == "x_t_hat_c"
        or self._config.breakdown == "x_t_hat_c_with_zeros"
    ):
        zero_index_reconstruction = 1 if not self.time_measure_n else 0
        edit_prompts_num = (
            (model_output.size(0) - zero_index_reconstruction) // 3
            if self._config.breakdown == "x_t_hat_c_with_zeros" and not self.p_to_p
            else (model_output.size(0) - zero_index_reconstruction) // 2
        )
        x_t_hat_c_indices = (zero_index_reconstruction, edit_prompts_num + zero_index_reconstruction)
        edit_images_indices = (
            edit_prompts_num + zero_index_reconstruction,
            (
                model_output.size(0)
                if self._config.breakdown == "x_t_hat_c"
                else zero_index_reconstruction + 2 * edit_prompts_num
            ),
        )
        x_t_hat_c = torch.zeros_like(x_t_hat_c_hat)
        x_t_hat_c[edit_images_indices[0] : edit_images_indices[1]] = x_t_hat_c_hat[
            x_t_hat_c_indices[0] : x_t_hat_c_indices[1]
        ]
        v1 = x_t_hat_c_hat - x_t_hat_c
        v2 = x_t_hat_c - normalize_coefficient * x_t_c
        if self._config.breakdown == "x_t_hat_c_with_zeros" and not self.p_to_p:
            path = os.path.join(
                self.folder_name,
                VECTOR_DATA_FOLDER,
                self.image_name,
            )
            if not hasattr(self, VECTOR_DATA_DICT):
                os.makedirs(path, exist_ok=True)
                self.vector_data = dict()

            x_t_0 = x_t_c_hat[1]
            empty_prompt_indices = (1 + 2 * edit_prompts_num, 1 + 3 * edit_prompts_num)
            x_t_hat_0 = x_t_hat_c_hat[empty_prompt_indices[0] : empty_prompt_indices[1]]

            self.vector_data[timestep.item()] = dict()
            self.vector_data[timestep.item()]["x_t_hat_c"] = x_t_hat_c[
                edit_images_indices[0] : edit_images_indices[1]
            ]
            self.vector_data[timestep.item()]["x_t_hat_0"] = x_t_hat_0
            self.vector_data[timestep.item()]["x_t_c"] = x_t_c[0].expand_as(x_t_hat_0)
            self.vector_data[timestep.item()]["x_t_0"] = x_t_0.expand_as(x_t_hat_0)
            self.vector_data[timestep.item()]["x_t_hat_c_hat"] = x_t_hat_c_hat[
                edit_images_indices[0] : edit_images_indices[1]
            ]
            self.vector_data[timestep.item()]["x_t_minus_1_noisy"] = x_t_minus_1_exact[
                0
            ].expand_as(x_t_hat_0)
            self.vector_data[timestep.item()]["x_t_minus_1_clean"] = self.x_0s[
                next_timestep_index
            ].expand_as(x_t_hat_0)

    else:  # no breakdown
        v1 = x_t_hat_c_hat - normalize_coefficient * x_t_c
        v2 = 0

    if self.save_intermediate_results and not self.p_to_p:
        delta = v1 + v2
        v1_plus_x0 = self.x_0s[next_timestep_index] + v1
        v2_plus_x0 = self.x_0s[next_timestep_index] + v2
        delta_plus_x0 = self.x_0s[next_timestep_index] + delta

        v1_images = decode_latents(v1, self.pipe)
        self.v1s_images.append(v1_images)
        v2_images = (
            decode_latents(v2, self.pipe)
            if self._config.breakdown != "no_breakdown"
            else [PIL.Image.new("RGB", (1, 1))]
        )
        self.v2s_images.append(v2_images)
        delta_images = decode_latents(delta, self.pipe)
        self.deltas_images.append(delta_images)
        v1_plus_x0_images = decode_latents(v1_plus_x0, self.pipe)
        self.v1_x0s.append(v1_plus_x0_images)
        v2_plus_x0_images = (
            decode_latents(v2_plus_x0, self.pipe)
            if self._config.breakdown != "no_breakdown"
            else [PIL.Image.new("RGB", (1, 1))]
        )
        self.v2_x0s.append(v2_plus_x0_images)
        delta_plus_x0_images = decode_latents(delta_plus_x0, self.pipe)
        self.deltas_x0s.append(delta_plus_x0_images)

    # print(f"v1 norm: {torch.norm(v1, dim=0).mean()}")
    # if self._config.breakdown != "no_breakdown":
    #     print(f"v2 norm: {torch.norm(v2, dim=0).mean()}")
    #     print(f"v sum norm: {torch.norm(v1 + v2, dim=0).mean()}")

    x_t_minus_1 = normalize_coefficient * x_t_minus_1_exact + w1 * v1 + w2 * v2

    if (
        self._config.breakdown == "x_t_hat_c"
        or self._config.breakdown == "x_t_hat_c_with_zeros"
    ):
        x_t_minus_1[x_t_hat_c_indices[0] : x_t_hat_c_indices[1]] = x_t_minus_1[
            edit_images_indices[0] : edit_images_indices[1]
        ] # update x_t_hat_c to be x_t_hat_c_hat
        if self._config.breakdown == "x_t_hat_c_with_zeros" and not self.p_to_p:
            x_t_minus_1[empty_prompt_indices[0] : empty_prompt_indices[1]] = (
                x_t_minus_1[edit_images_indices[0] : edit_images_indices[1]]
            )
            self.vector_data[timestep.item()]["x_t_minus_1_edited"] = x_t_minus_1[
                edit_images_indices[0] : edit_images_indices[1]
            ]
            if timestep == self._timesteps[-1]:
                torch.save(
                    self.vector_data,
                    os.path.join(
                        path,
                        f"{VECTOR_DATA_DICT}.pt",
                    ),
                )
    # p_to_p_force_perfect_reconstruction
    if not self.time_measure_n:
        x_t_minus_1[0] = x_t_minus_1_exact[0]

    if not return_dict:
        return (x_t_minus_1,)

    return DDIMSchedulerOutput(
        prev_sample=x_t_minus_1,
        pred_original_sample=None,
    )



def get_ddpm_inversion_scheduler(
    scheduler,
    step_function,
    config,
    timesteps,
    save_timesteps,
    latents,
    x_ts,
    x_ts_c_hat,
    save_intermediate_results,
    pipe,
    x_0,
    v1s_images,
    v2s_images,
    deltas_images,
    v1_x0s,
    v2_x0s,
    deltas_x0s,
    folder_name,
    image_name,
    time_measure_n,
):
    def step(
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        # if scheduler.is_save:
        # start = timer()
        res_inv =  step_save_latents(
            scheduler,
            model_output[:1, :, :, :],
            timestep,
            sample[:1, :, :, :],
            eta,
            use_clipped_model_output,
            generator,
            variance_noise,
            return_dict,
        )
        # end = timer()
        # print(f"Run Time Inv: {end - start}")

        res_inf = step_use_latents(
            scheduler,
            model_output[1:, :, :, :],
            timestep,
            sample[1:, :, :, :],
            eta,
            use_clipped_model_output,
            generator,
            variance_noise,
            return_dict,
        )
        # res = res_inv
        res = (torch.cat((res_inv[0], res_inf[0]), dim=0),)
        return res
        # return res

    scheduler.step_function = step_function
    scheduler.is_save = True
    scheduler._timesteps = timesteps
    scheduler._save_timesteps = save_timesteps if save_timesteps else timesteps
    scheduler._config = config
    scheduler.latents = latents
    scheduler.x_ts = x_ts
    scheduler.x_ts_c_hat = x_ts_c_hat
    scheduler.step = step
    scheduler.save_intermediate_results = save_intermediate_results
    scheduler.pipe = pipe
    scheduler.v1s_images = v1s_images
    scheduler.v2s_images = v2s_images
    scheduler.deltas_images = deltas_images
    scheduler.v1_x0s = v1_x0s
    scheduler.v2_x0s = v2_x0s
    scheduler.deltas_x0s = deltas_x0s
    scheduler.clean_step_run = False
    scheduler.x_0s = create_xts(
        config.noise_shift_delta,
        config.noise_timesteps,
        config.clean_step_timestep,
        None,
        pipe.scheduler,
        timesteps,
        x_0,
        no_add_noise=True,
    )
    scheduler.folder_name = folder_name
    scheduler.image_name = image_name
    scheduler.p_to_p = False
    scheduler.p_to_p_replace = False
    scheduler.time_measure_n = time_measure_n
    return scheduler


def create_grid(
    images,
    p_to_p_images,
    prompts,
    original_image_path,
):
    images_len = len(images) if len(images) > 0 else len(p_to_p_images)
    images_size = images[0].size if len(images) > 0 else p_to_p_images[0].size
    x_0 = Image.open(original_image_path).resize(images_size)

    images_ = [x_0] + images + ([x_0] + p_to_p_images if p_to_p_images else [])

    l1 = 1 if len(images) > 0 else 0
    l2 = 1 if len(p_to_p_images) else 0
    grid = make_image_grid(images_, rows=l1 + l2, cols=images_len + 1, resize=None)

    width = images_size[0]
    height = width // 5
    font = ImageFont.truetype("font.ttf", width // 14)

    grid1 = Image.new("RGB", size=(grid.size[0], grid.size[1] + height))
    grid1.paste(grid, (0, 0))

    draw = ImageDraw.Draw(grid1)

    c_width = 0
    for prompt in prompts:
        if len(prompt) > 30:
            prompt = prompt[:30] + "\n" + prompt[30:]
        draw.text((c_width, width * 2), prompt, font=font, fill=(255, 255, 255))
        c_width += width

    return grid1


def save_intermediate_results(
    v1s_images,
    v2s_images,
    deltas_images,
    v1_x0s,
    v2_x0s,
    deltas_x0s,
    folder_name,
    original_prompt,
):
    from diffusers.utils import make_image_grid

    path = f"{folder_name}/{original_prompt}_intermediate_results/"
    os.makedirs(path, exist_ok=True)
    make_image_grid(
        list(itertools.chain(*v1s_images)),
        rows=len(v1s_images),
        cols=len(v1s_images[0]),
    ).save(f"{path}v1s_images.png")
    make_image_grid(
        list(itertools.chain(*v2s_images)),
        rows=len(v2s_images),
        cols=len(v2s_images[0]),
    ).save(f"{path}v2s_images.png")
    make_image_grid(
        list(itertools.chain(*deltas_images)),
        rows=len(deltas_images),
        cols=len(deltas_images[0]),
    ).save(f"{path}deltas_images.png")
    make_image_grid(
        list(itertools.chain(*v1_x0s)),
        rows=len(v1_x0s),
        cols=len(v1_x0s[0]),
    ).save(f"{path}v1_x0s.png")
    make_image_grid(
        list(itertools.chain(*v2_x0s)),
        rows=len(v2_x0s),
        cols=len(v2_x0s[0]),
    ).save(f"{path}v2_x0s.png")
    make_image_grid(
        list(itertools.chain(*deltas_x0s)),
        rows=len(deltas_x0s[0]),
        cols=len(deltas_x0s),
    ).save(f"{path}deltas_x0s.png")
    for i, image in enumerate(list(itertools.chain(*deltas_x0s))):
        image.save(f"{path}deltas_x0s_{i}.png")


# copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.py and removed the add_noise line
def prepare_latents_no_add_noise(
    self,
    image,
    timestep,
    batch_size,
    num_images_per_prompt,
    dtype,
    device,
    generator=None,
):
    from diffusers.utils import deprecate

    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            init_latents = [
                self.retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        init_latents = self.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate(
            "len(prompt) != len(image)",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    # get latents
    latents = init_latents

    return latents


# Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_empty_prompt_zeros_sdxl(
    self,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    device = device or self._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = (
        [self.tokenizer, self.tokenizer_2]
        if self.tokenizer is not None
        else [self.tokenizer_2]
    )
    text_encoders = (
        [self.text_encoder, self.text_encoder_2]
        if self.text_encoder is not None
        else [self.text_encoder_2]
    )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(device), output_hidden_states=True
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            if self.config.force_zeros_for_empty_prompt:
                prompt_embeds[[i for i in range(len(prompt)) if prompt[i] == ""]] = 0
                pooled_prompt_embeds[
                    [i for i in range(len(prompt)) if prompt[i] == ""]
                ] = 0

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = (
        negative_prompt is None and self.config.force_zeros_for_empty_prompt
    )
    if (
        do_classifier_free_guidance
        and negative_prompt_embeds is None
        and zero_out_negative_prompt
    ):
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )
        negative_prompt_2 = (
            batch_size * [negative_prompt_2]
            if isinstance(negative_prompt_2, str)
            else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(
            uncond_tokens, tokenizers, text_encoders
        ):

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if self.text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        if self.text_encoder_2 is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder_2.dtype, device=device
            )
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.unet.dtype, device=device
            )

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)

    if self.text_encoder is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def create_xts(
    noise_shift_delta,
    noise_timesteps,
    clean_step_timestep,
    generator,
    scheduler,
    timesteps,
    x_0,
    no_add_noise=False,
):
    if noise_timesteps is None:
        noising_delta = noise_shift_delta * (timesteps[0] - timesteps[1])
        noise_timesteps = [timestep - int(noising_delta) for timestep in timesteps]

    first_x_0_idx = len(noise_timesteps)
    for i in range(len(noise_timesteps)):
        if noise_timesteps[i] <= 0:
            first_x_0_idx = i
            break

    noise_timesteps = noise_timesteps[:first_x_0_idx]

    x_0_expanded = x_0.expand(len(noise_timesteps), -1, -1, -1)
    noise = (
        torch.randn(x_0_expanded.size(), generator=generator, device="cpu").to(
            x_0.device
        )
        if not no_add_noise
        else torch.zeros_like(x_0_expanded)
    )
    x_ts = scheduler.add_noise(
        x_0_expanded,
        noise,
        torch.IntTensor(noise_timesteps),
    )
    x_ts = [t.unsqueeze(dim=0) for t in list(x_ts)]
    x_ts += [x_0] * (len(timesteps) - first_x_0_idx)
    x_ts += [x_0]
    if clean_step_timestep > 0:
        x_ts += [x_0]
    return x_ts


# Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
def add_noise(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    image_timesteps: torch.IntTensor,
    noise_timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[image_timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[noise_timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = (
        sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    )
    return noisy_samples


def make_image_grid(
    images: List[PIL.Image.Image], rows: int, cols: int, resize: int = None, size=None
) -> PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid