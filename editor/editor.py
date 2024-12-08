import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
import random
import yaml
import os
#----------------------------------------------------------------------------------------------------------------#
class TextBasedImageEditor:
    """
    Text-based image editor class to generate an edited image based on the provided input image and instruction.
    
    Args:
        model_name (str): Name of the model to be used for editing the image.
        config_path (str): Path to the configuration file containing the model configurations
    """
    def __init__(self, model_name: str, config_path: str = "config/config.yaml"):
        """
        Initialize the text-based image editor with the provided model name and configuration path.
        
        Args:
            model_name (str): Name of the model to be used for editing the image.
            config_path (str): Path to the configuration file containing the model configurations
        """
        self.model_name = model_name
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.device = config.get('device', 'cuda')
        if self.model_name not in config:
            raise ValueError(f"Model '{self.model_name}' not found in the configuration. Available models: {list(config.keys())}")

        self.model_config = config[self.model_name]


        self.supported_models = {
            'pix_to_pix': StableDiffusionInstructPix2PixPipeline,
            # Add more models and their pipeline classes here
        }

        if self.model_name not in self.supported_models:
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models are: {list(self.supported_models.keys())}")

        PipelineClass = self.supported_models[self.model_name]

        model_id = self.model_config.get('model_id')
        if not model_id:
            raise ValueError(f"'model_id' must be specified for model '{self.model_name}'.")

        torch_dtype = getattr(torch, self.model_config.get('torch_dtype', 'float16'))
        safety_checker = self.model_config.get('safety_checker', None)

        self.pipe = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=safety_checker
        ).to(self.device)

#----------------------------------------------------------------------------------------------------------------#
    def generate_image(self, input_image: Image.Image = None, instruction: str = "") -> Image.Image:
        """
        Generate an edited image based on the provided input image and instruction.

        Args:
            input_image (Image.Image): Input image to be edited.
            instruction (str): Instruction to edit the image.
        """
        steps = self.model_config.get('steps', 50)
        randomize_seed = self.model_config.get('randomize_seed', False)
        seed = self.model_config.get('seed', 42)
        randomize_cfg = self.model_config.get('randomize_cfg', False)
        text_cfg_scale = self.model_config.get('text_cfg_scale', 7.5)
        image_cfg_scale = self.model_config.get('image_cfg_scale', 1.5)

        if randomize_seed:
            seed = random.randint(0, 100000)
        generator = torch.manual_seed(seed)

        if randomize_cfg:
            text_cfg_scale = round(random.uniform(6.0, 9.0), 2)
            image_cfg_scale = round(random.uniform(1.2, 1.8), 2)

        if self.model_name == 'pix_to_pix':
            if input_image is None:
                raise ValueError("Input image must be provided for 'pix_to_pix' model.")
            if not instruction:
                raise ValueError("Instruction must be provided for 'pix_to_pix' model.")

            edited_image = self.pipe(
                instruction=instruction,
                image=input_image,
                guidance_scale=text_cfg_scale,
                image_guidance_scale=image_cfg_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        else:
            edited_image = self.pipe(
                prompt=instruction,
                num_inference_steps=steps,
                guidance_scale=text_cfg_scale,
                generator=generator,
            ).images[0]

        return edited_image
