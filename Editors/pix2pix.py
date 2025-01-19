"""
This file contains the code to use pix to pix text based image editing tool
"""
import os
import numpy as np
from pathlib import Path
import sys
import torch
from PIL import Image
import cv2
import random
import yaml
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import load_config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
config = load_config(config_path)['Editor']['pix2pix']

class Pix2Pix:
    """
    Text-based image editor class to generate an edited image based on the provided input image and instruction.
    """
    def __init__(self, config):
        """
        Initialize the text-based image editor with the provided model name and configuration path.
        """
        self.config = config
        self.device = self.config['device']
        self.steps = self.config['steps']
        self.random_seed = self.config['seed']
        self.randomize_cfg = self.config['randomize_cfg']
        self.text_cfg_scale = self.config['text_cfg_scale']
        self.image_cfg_scale = self.config['image_cfg_scale']
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.config['model_id'],torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.generator = torch.manual_seed(self.random_seed)
    
    def _load_image(self, image_path:Path) -> Image:
        """
        Load the image from the provided image path.
        """
        return Image.open(image_path).convert("RGB")

    def edit_image(self, input_image_path: Path, instruction: str, output_path:Path) -> Image:
        """
        Edit the input image based on the provided instruction.

        Args:
            input_image_path (str): Path to the input image.
            instruction (str): Instruction for editing the image.

        Returns:
            Image: Edited image.
        """
        print("Device :", self.device)
        image = self._load_image(input_image_path)
        edited_image = self.model(
            prompt=instruction,
            image=image,
            num_inference_steps=self.steps,
            generator=self.generator,
            negative_prompt=None,  # Provide a negative prompt if desired
            guidance_scale=self.text_cfg_scale,  # Text CFG scale
            image_guidance_scale=self.image_cfg_scale  # Image CFG scale
        ).images[0]

        cv2.imwrite(output_path, np.array(edited_image))

        return edited_image
    
if __name__ == "__main__":
    model = Pix2Pix(config)
    model.edit_image("../data/image.jpg", "A yellow bird sitting on a branch.", "../data/edited_image.jpg")