"""Unified interfaces for text-guided image editing models."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Iterable, Optional

import torch
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


logger = logging.getLogger(__name__)


class BaseEditor(ABC):
    """Common lifecycle and configuration handling for image editors."""

    def __init__(self, config: dict):
        self.config: dict = dict(config)
        self.device: str = str(self.config.get("device", "cuda"))
        self.dtype: torch.dtype = self._resolve_dtype(
            self.config.get("dtype", "float16")
        )
        self.model_path: Optional[str] = self.config.get("model_path")
        if not self.model_path:
            raise ValueError("Editor configuration must include a non-empty 'model_path'.")

        self.model: Any = None
        self._load_model()

    @staticmethod
    def _resolve_dtype(dtype: Any) -> torch.dtype:
        """Convert a config dtype value into a torch dtype."""
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "float16":
            return torch.float16
        if dtype == "float32":
            return torch.float32
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        raise ValueError(
            "Unsupported dtype. Use 'float16', 'float32', or 'bfloat16'."
        )

    def _pretrained_kwargs(self) -> Dict[str, Any]:
        """Return optional keyword arguments for a pretrained model loader."""
        values = self.config.get("pretrained_kwargs", {})
        if not isinstance(values, dict):
            raise TypeError("'pretrained_kwargs' must be a dictionary when provided.")
        return dict(values)

    def _generation_value(self, name: str, default: Any, kwargs: dict) -> Any:
        """Resolve an edit-time override, then a config value, then a fallback."""
        if name in kwargs:
            return kwargs[name]
        return self.config.get(name, default)

    def _generator(self, kwargs: dict) -> Optional[torch.Generator]:
        generator = kwargs.get("generator")
        if generator is not None:
            return generator

        seed = kwargs.get("seed", self.config.get("seed"))
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _diffusion_kwargs(
        self,
        kwargs: dict,
        default_steps: int,
        default_guidance: float,
    ) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "num_inference_steps": int(
                self._generation_value("num_inference_steps", default_steps, kwargs)
            ),
            "guidance_scale": float(
                self._generation_value("guidance_scale", default_guidance, kwargs)
            ),
        }
        negative_prompt = self._generation_value("negative_prompt", None, kwargs)
        if negative_prompt is not None:
            values["negative_prompt"] = negative_prompt

        generator = self._generator(kwargs)
        if generator is not None:
            values["generator"] = generator
        return values

    @staticmethod
    def _first_image(result: Any) -> Image.Image:
        """Extract the first PIL image from a diffusers pipeline result."""
        if isinstance(result, Image.Image):
            return result

        images = getattr(result, "images", None)
        if images is None and isinstance(result, dict):
            images = result.get("images")
        if isinstance(images, Iterable) and not isinstance(images, (str, bytes)):
            first = next(iter(images), None)
            if isinstance(first, Image.Image):
                return first
        raise RuntimeError("The editor pipeline did not return a PIL image.")

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights and preprocessing components."""

    @abstractmethod
    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        """Edit an image according to a text prompt."""


class MagicBrushEditor(BaseEditor):
    """Editor backed by Stable Diffusion InstructPix2Pix."""

    def _load_model(self) -> None:
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            **self._pretrained_kwargs(),
        )
        self.model.to(self.device)
        logger.info("Loaded MagicBrush editor from %s on %s", self.model_path, self.device)

    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        generation_kwargs = self._diffusion_kwargs(kwargs, default_steps=20, default_guidance=7.5)
        generation_kwargs["image_guidance_scale"] = float(
            self._generation_value("image_guidance_scale", 1.5, kwargs)
        )
        result = self.model(
            prompt=prompt,
            image=image.convert("RGB"),
            **generation_kwargs,
        )
        return self._first_image(result)


class InstructDiffusionEditor(BaseEditor):
    """Editor backed by a generic Diffusers InstructDiffusion pipeline."""

    def _load_model(self) -> None:
        self.model = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            **self._pretrained_kwargs(),
        )
        self.model.to(self.device)
        logger.info(
            "Loaded InstructDiffusion editor from %s on %s",
            self.model_path,
            self.device,
        )

    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        generation_kwargs = self._diffusion_kwargs(kwargs, default_steps=50, default_guidance=7.5)
        result = self.model(
            prompt=prompt,
            image=image.convert("RGB"),
            **generation_kwargs,
        )
        return self._first_image(result)


class UltraEditEditor(BaseEditor):
    """Editor backed by a generic Diffusers UltraEdit pipeline."""

    def _load_model(self) -> None:
        self.model = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            **self._pretrained_kwargs(),
        )
        self.model.to(self.device)
        logger.info("Loaded UltraEdit editor from %s on %s", self.model_path, self.device)

    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        generation_kwargs = self._diffusion_kwargs(kwargs, default_steps=50, default_guidance=7.5)
        result = self.model(
            prompt=prompt,
            image=image.convert("RGB"),
            **generation_kwargs,
        )
        return self._first_image(result)


class FluxKontextEditor(BaseEditor):
    """Editor backed by the Diffusers Flux Kontext pipeline."""

    def _load_model(self) -> None:
        self.model = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            **self._pretrained_kwargs(),
        )
        self.model.to(self.device)
        logger.info(
            "Loaded Flux Kontext editor from %s on %s",
            self.model_path,
            self.device,
        )

    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        generation_kwargs = self._diffusion_kwargs(kwargs, default_steps=28, default_guidance=3.5)
        result = self.model(
            prompt=prompt,
            image=image.convert("RGB"),
            **generation_kwargs,
        )
        return self._first_image(result)


class QwenImageEditEditor(BaseEditor):
    """Vision-language image editor using a multimodal chat template."""

    def _load_model(self) -> None:
        load_kwargs = self._pretrained_kwargs()
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            **dict(self.config.get("processor_kwargs", {})),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded Qwen image editor from %s on %s", self.model_path, self.device)

    @staticmethod
    def _image_from_output(output: Any) -> Optional[Image.Image]:
        if isinstance(output, Image.Image):
            return output

        images = getattr(output, "images", None)
        if images is None and isinstance(output, dict):
            images = output.get("images")
        if isinstance(images, Iterable) and not isinstance(images, (str, bytes)):
            first = next(iter(images), None)
            if isinstance(first, Image.Image):
                return first
        return None

    def edit(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(
                self._generation_value("max_new_tokens", 512, kwargs)
            ),
            "do_sample": bool(self._generation_value("do_sample", False, kwargs)),
        }
        temperature = self._generation_value("temperature", None, kwargs)
        if temperature is not None:
            generation_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        output_image = self._image_from_output(generated)
        if output_image is None:
            decode_image = getattr(self.model, "decode_image", None)
            if callable(decode_image):
                output_image = self._image_from_output(decode_image(generated))
        if output_image is None:
            decode_image = getattr(self.processor, "decode_image", None)
            if callable(decode_image):
                output_image = self._image_from_output(decode_image(generated))
        if output_image is None:
            raise RuntimeError(
                "QwenImageEditEditor expected the configured VLM to return an image "
                "or expose decode_image()."
            )
        return output_image


def get_editor(editor_name: str, config: dict) -> BaseEditor:
    """Instantiate an editor by its configuration-facing name."""
    editors = {
        "magicbrush": MagicBrushEditor,
        "instruct_diffusion": InstructDiffusionEditor,
        "ultraedit": UltraEditEditor,
        "flux_kontext": FluxKontextEditor,
        "qwen_image_edit": QwenImageEditEditor,
    }
    normalized_name = editor_name.strip().lower().replace("-", "_")
    editor_class = editors.get(normalized_name)
    if editor_class is None:
        supported = ", ".join(sorted(editors))
        raise ValueError(
            f"Unknown editor '{editor_name}'. Supported editors: {supported}."
        )
    return editor_class(config)
