import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import random

class TextBasedImageEditor:
    def __init__(self, model_id: str, device: str = "cuda"):
        """Initialize the model and the device (GPU/CPU)."""
        self.model_id = model_id
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to(self.device)

    def generate_image(self, input_image: Image.Image, instruction: str, steps: int, randomize_seed: bool,
                       seed: int, randomize_cfg: bool, text_cfg_scale: float, image_cfg_scale: float) -> Image.Image:
        """Generate the edited image based on the provided instruction and configuration."""
        seed = random.randint(0, 100000) if randomize_seed else seed
        text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
        image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

        generator = torch.manual_seed(seed)
        edited_image = self.pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]
        return edited_image
