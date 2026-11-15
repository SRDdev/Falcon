"""Adapters for external image watermarking baselines."""

import importlib
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms


logger = logging.getLogger(__name__)


def pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert a PIL image to a batched float tensor in the [0, 1] range."""
    tensor = transforms.ToTensor()(img.convert("RGB"))
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, C, H, W] or [C, H, W] tensor to a PIL image."""
    image_tensor = tensor.detach().float().cpu()
    if image_tensor.ndim == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("tensor_to_pil expects a single image batch.")
        image_tensor = image_tensor[0]
    if image_tensor.ndim != 3:
        raise ValueError("tensor_to_pil expects a [C, H, W] or [1, C, H, W] tensor.")
    return transforms.ToPILImage()(image_tensor.clamp(0.0, 1.0))


def _resolve_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(
            "Unsupported dtype. Use float16, float32, or bfloat16."
        )
    return dtype_map[dtype]


def _as_tensor(value: Any, keys: Tuple[str, ...]) -> torch.Tensor:
    """Extract a tensor from common external-model output structures."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        if all(isinstance(item, (bool, int, float)) for item in value):
            return torch.tensor(value)
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return _as_tensor(value[key], keys)
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, (torch.Tensor, dict, tuple, list)):
                try:
                    return _as_tensor(item, keys)
                except (KeyError, TypeError, ValueError):
                    continue
    raise TypeError("External watermark model output does not contain a tensor.")


class BaseWatermark(ABC):
    """Base adapter that standardizes model setup, device, and detection output."""

    def __init__(self, config: dict):
        self.config: dict = dict(config)
        self.device: str = str(self.config.get("device", "cuda"))
        self.dtype: torch.dtype = _resolve_dtype(
            self.config.get("dtype", "float32")
        )
        configured_repo_dir = str(self.config.get("repo_dir", "external/VINE"))
        self.repo_dir: str = os.path.abspath(configured_repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.append(self.repo_dir)

        self._setup()

    def _component_kwargs(self, role: str) -> Dict[str, Any]:
        values = self.config.get(f"{role}_kwargs", {})
        if not isinstance(values, dict):
            raise TypeError(f"'{role}_kwargs' must be a dictionary when provided.")
        return dict(values)

    def _load_component(
        self,
        role: str,
        default_module: str,
        default_class: str,
    ) -> Any:
        module_name = str(self.config.get(f"{role}_module", default_module))
        class_name = str(self.config.get(f"{role}_class", default_class))
        try:
            # Configure this module/class pair to match the external repository's exact import path.
            module = importlib.import_module(module_name)
            component_class = getattr(module, class_name)
            component = component_class(**self._component_kwargs(role))
        except (ImportError, AttributeError, TypeError) as exc:
            raise ImportError(
                f"Unable to load {role} '{class_name}' from '{module_name}'. "
                "Set the corresponding module, class, and kwargs in config."
            ) from exc

        checkpoint_path = self.config.get(f"{role}_checkpoint")
        if checkpoint_path:
            checkpoint = torch.load(
                os.path.abspath(str(checkpoint_path)),
                map_location="cpu",
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            component.load_state_dict(state_dict, strict=False)

        if hasattr(component, "to"):
            component.to(device=self.device, dtype=self.dtype)
        if hasattr(component, "eval"):
            component.eval()
        return component

    def _payload_tensor(self, payload: List[int]) -> torch.Tensor:
        if not payload:
            raise ValueError("payload must contain at least one bit.")
        if any(bit not in (0, 1) for bit in payload):
            raise ValueError("payload must contain only binary values: 0 or 1.")
        return torch.tensor(
            payload,
            device=self.device,
            dtype=self.dtype,
        ).view(1, -1)

    def _detection_result(
        self,
        output: Any,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Normalize logits, bit arrays, strings, or dict outputs from detectors."""
        raw_output = output
        if isinstance(output, dict):
            confidence = output.get("confidence", confidence)
            for key in ("payload", "bits", "bit_array", "logits", "output"):
                if key in output:
                    raw_output = output[key]
                    break

        if isinstance(raw_output, str):
            payload = [int(bit) for bit in raw_output if bit in "01"]
            score = float(confidence if confidence is not None else 1.0)
        else:
            values = _as_tensor(raw_output, ("payload", "bits", "bit_array", "logits"))
            values = values.detach().float().flatten()
            probabilities = (
                values.sigmoid()
                if torch.any((values < 0.0) | (values > 1.0))
                else values
            )
            payload = (probabilities >= 0.5).to(torch.int64).cpu().tolist()
            margins = (probabilities - 0.5).abs() * 2.0
            score = float(
                confidence if confidence is not None else margins.mean().item()
            )

        threshold = float(self.config.get("detection_threshold", 0.5))
        return {
            "payload": [int(bit) for bit in payload],
            "confidence": max(0.0, min(1.0, score)),
            "is_watermarked": bool(score >= threshold),
        }

    @abstractmethod
    def _setup(self) -> None:
        """Load external model components and move them to self.device."""

    @abstractmethod
    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        """Embed payload bits into an image."""

    @abstractmethod
    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        """Detect a payload and return standardized confidence metadata."""


class VineWatermark(BaseWatermark):
    """Adapter for a VINE encoder/decoder pair."""

    def _setup(self) -> None:
        self.encoder = self._load_component("encoder", "vine", "Encoder")
        self.decoder = self._load_component("decoder", "vine", "Decoder")

    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        payload_tensor = self._payload_tensor(payload)
        with torch.inference_mode():
            output = self.encoder(image_tensor, payload_tensor, **kwargs)
        watermarked = _as_tensor(output, ("watermarked", "image", "output"))
        return tensor_to_pil(watermarked)

    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        with torch.inference_mode():
            output = self.decoder(image_tensor, **kwargs)
        return self._detection_result(output)


class RobustWideWatermark(BaseWatermark):
    """Adapter for a frequency-domain or robust deep watermark network."""

    def _setup(self) -> None:
        self.encoder = self._load_component("encoder", "robust_wide", "Encoder")
        self.decoder = self._load_component("decoder", "robust_wide", "Decoder")

    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        payload_tensor = self._payload_tensor(payload)
        with torch.inference_mode():
            output = self.encoder(image_tensor, payload_tensor, **kwargs)
        watermarked = _as_tensor(output, ("watermarked", "image", "output"))
        return tensor_to_pil(watermarked)

    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        with torch.inference_mode():
            output = self.decoder(image_tensor, **kwargs)
        return self._detection_result(output)


class PeccaviWatermark(BaseWatermark):
    """Adapter for the CVPR'26 PeCCaVi multi-carrier Fourier watermark."""

    def _setup(self) -> None:
        self.embedder = self._load_component("embedder", "peccavi", "Encoder")
        self.detector = self._load_component("detector", "peccavi", "Decoder")

    def _fourier_template(
        self,
        payload: List[int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        carrier_count = int(self.config.get("num_carriers", len(payload)))
        if carrier_count < len(payload):
            raise ValueError("num_carriers must be at least the payload length.")
        template = torch.zeros(
            (1, carrier_count, height, width),
            device=self.device,
            dtype=self.dtype,
        )
        rows = self.config.get("carrier_rows")
        cols = self.config.get("carrier_cols")
        for index, bit in enumerate(payload):
            row = int(rows[index]) if rows else (index + 1) % height
            col = int(cols[index]) if cols else (index + 1) % width
            template[0, index, row % height, col % width] = float(bit)
        return template

    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        _, _, height, width = image_tensor.shape
        template = self._fourier_template(payload, height, width)
        with torch.inference_mode():
            output = self.embedder(image_tensor, template, **kwargs)
        watermarked = _as_tensor(output, ("watermarked", "image", "output"))
        return tensor_to_pil(watermarked)

    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        with torch.inference_mode():
            output = self.detector(image_tensor, **kwargs)
        raw_result = self._detection_result(output)
        raw_payload = raw_result["payload"]
        # Replace this pass-through with the repository's RS erasure decoder.
        decoded_payload = raw_payload
        return {
            "payload": decoded_payload,
            "confidence": raw_result["confidence"],
            "is_watermarked": raw_result["is_watermarked"],
        }


class StableSignatureWatermark(BaseWatermark):
    """Adapter for Stable Signature's VAE extractor."""

    def _setup(self) -> None:
        self.extractor = self._load_component(
            "extractor",
            "stable_signature",
            "Extractor",
        )

    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        if not payload:
            raise ValueError("payload must contain at least one bit.")
        logger.debug(
            "Stable Signature embedding is generation-time; returning the input image."
        )
        return image.convert("RGB").copy()

    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        with torch.inference_mode():
            output = self.extractor(image_tensor, **kwargs)
        return self._detection_result(output)


class ZodiacWatermark(BaseWatermark):
    """Adapter for a zero-shot diffusion watermark using DDIM inversion."""

    def _setup(self) -> None:
        self.model = self._load_component("model", "zodiac", "ZodiacModel")

    def embed(
        self,
        image: Image.Image,
        payload: List[int],
        **kwargs: Any,
    ) -> Image.Image:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        target_latent = self._payload_tensor(payload)
        inversion_steps = int(
            kwargs.get(
                "ddim_steps",
                self.config.get("ddim_steps", 50),
            )
        )
        with torch.inference_mode():
            inverted = self.model.ddim_invert(
                image_tensor,
                num_inference_steps=inversion_steps,
            )
            output = self.model.embed(inverted, target_latent, **kwargs)
        watermarked = _as_tensor(output, ("watermarked", "image", "output"))
        return tensor_to_pil(watermarked)

    def detect(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        image_tensor = pil_to_tensor(image, self.device).to(dtype=self.dtype)
        inversion_steps = int(
            kwargs.get(
                "ddim_steps",
                self.config.get("ddim_steps", 50),
            )
        )
        with torch.inference_mode():
            inverted = self.model.ddim_invert(
                image_tensor,
                num_inference_steps=inversion_steps,
            )
            output = self.model.detect(inverted, **kwargs)
        return self._detection_result(output)


def get_watermark_baseline(baseline_name: str, config: dict) -> BaseWatermark:
    """Instantiate a configured external watermarking baseline."""
    baselines = {
        "vine": VineWatermark,
        "robust_wide": RobustWideWatermark,
        "peccavi": PeccaviWatermark,
        "stable_signature": StableSignatureWatermark,
        "zodiac": ZodiacWatermark,
    }
    normalized_name = baseline_name.strip().lower().replace("-", "_")
    baseline_class = baselines.get(normalized_name)
    if baseline_class is None:
        supported = ", ".join(sorted(baselines))
        raise ValueError(
            f"Unknown watermark baseline '{baseline_name}'. "
            f"Supported baselines: {supported}."
        )
    return baseline_class(config)
