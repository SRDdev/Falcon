# import torch
# from diffusers import (
#     StableDiffusionInstructPix2PixPipeline,
#     StableDiffusionPipeline,
# )
# from PIL import Image
# import random
# import yaml
# import os
# #----------------------------------------------------------------------------------------------------------------#
# class TextBasedImageEditor:
#     """
#     Text-based image editor class to generate an edited image based on the provided input image and instruction.
    
#     Args:
#         model_name (str): Name of the model to be used for editing the image.
#         config_path (str): Path to the configuration file containing the model configurations
#     """
#     def __init__(self, model_name: str, config_path: str = "config/config.yaml"):
#         """
#         Initialize the text-based image editor with the provided model name and configuration path.
        
#         Args:
#             model_name (str): Name of the model to be used for editing the image.
#             config_path (str): Path to the configuration file containing the model configurations
#         """
#         self.model_name = model_name
#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)

#         self.device = config.get('device', 'cuda')
#         if self.model_name not in config:
#             raise ValueError(f"Model '{self.model_name}' not found in the configuration. Available models: {list(config.keys())}")

#         self.model_config = config[self.model_name]


#         self.supported_models = {
#             'pix_to_pix': StableDiffusionInstructPix2PixPipeline,
#             # Add more models and their pipeline classes here
#         }

#         if self.model_name not in self.supported_models:
#             raise ValueError(f"Model '{self.model_name}' is not supported. Supported models are: {list(self.supported_models.keys())}")

#         PipelineClass = self.supported_models[self.model_name]

#         model_id = self.model_config.get('model_id')
#         if not model_id:
#             raise ValueError(f"'model_id' must be specified for model '{self.model_name}'.")

#         torch_dtype = getattr(torch, self.model_config.get('torch_dtype', 'float16'))
#         safety_checker = self.model_config.get('safety_checker', None)

#         self.pipe = PipelineClass.from_pretrained(
#             model_id,
#             torch_dtype=torch_dtype,
#             safety_checker=safety_checker
#         ).to(self.device)

# #----------------------------------------------------------------------------------------------------------------#
#     def generate_image(self, input_image: Image.Image = None, instruction: str = "") -> Image.Image:
#         """
#         Generate an edited image based on the provided input image and instruction.

#         Args:
#             input_image (Image.Image): Input image to be edited.
#             instruction (str): Instruction to edit the image.
#         """
#         steps = self.model_config.get('steps', 50)
#         randomize_seed = self.model_config.get('randomize_seed', False)
#         seed = self.model_config.get('seed', 42)
#         randomize_cfg = self.model_config.get('randomize_cfg', False)
#         text_cfg_scale = self.model_config.get('text_cfg_scale', 7.5)
#         image_cfg_scale = self.model_config.get('image_cfg_scale', 1.5)

#         if randomize_seed:
#             seed = random.randint(0, 100000)
#         generator = torch.manual_seed(seed)

#         if randomize_cfg:
#             text_cfg_scale = round(random.uniform(6.0, 9.0), 2)
#             image_cfg_scale = round(random.uniform(1.2, 1.8), 2)

#         if self.model_name == 'pix_to_pix':
#             if input_image is None:
#                 raise ValueError("Input image must be provided for 'pix_to_pix' model.")
#             if not instruction:
#                 raise ValueError("Instruction must be provided for 'pix_to_pix' model.")

#             edited_image = self.pipe(
#                 instruction=instruction,
#                 image=input_image,
#                 guidance_scale=text_cfg_scale,
#                 image_guidance_scale=image_cfg_scale,
#                 num_inference_steps=steps,
#                 generator=generator,
#             ).images[0]
#         else:
#             edited_image = self.pipe(
#                 prompt=instruction,
#                 num_inference_steps=steps,
#                 guidance_scale=text_cfg_scale,
#                 generator=generator,
#             ).images[0]

#         return edited_image


import torch
import os
import yaml
import random
import uuid
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from .Turbo_edit import TurboEdit

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

        # Ensure the model name is found in config
        if self.model_name not in config:
            raise ValueError(
                f"Model '{self.model_name}' not found in the configuration. "
                f"Available models: {list(config.keys())}"
            )

        self.model_config = config[self.model_name]

        # We add a new entry 'turbo_edit' to the supported models
        self.supported_models = {
            'pix_to_pix': StableDiffusionInstructPix2PixPipeline,
            'turbo_edit': TurboEdit,
        }

        # If the model is a standard diffusers pipeline, we store the pipeline class
        # but if it's 'turbo_edit', we will instantiate TurboEdit directly.
        if self.model_name not in self.supported_models:
            raise ValueError(
                f"Model '{self.model_name}' is not supported. "
                f"Supported models are: {list(self.supported_models.keys())}"
            )

        # If the user picks a diffusers model, we build it from .from_pretrained
        # If they pick 'turbo_edit', we’ll instantiate our custom TurboEdit class
        if self.model_name == 'turbo_edit':
            # Initialize TurboEdit directly
            self.pipe = TurboEdit(
                model_name=self.model_name,
                config_path=config_path
            )
        else:
            PipelineClass = self.supported_models[self.model_name]

            model_id = self.model_config.get('model_id')
            if not model_id:
                raise ValueError(
                    f"'model_id' must be specified for model '{self.model_name}'."
                )

            torch_dtype = getattr(
                torch,
                self.model_config.get('torch_dtype', 'float16')
            )
            safety_checker = self.model_config.get('safety_checker', None)

            self.pipe = PipelineClass.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=safety_checker,
            ).to(self.device)

    #----------------------------------------------------------------------------------------------------------------#
    def generate_image(self, input_image: Image.Image = None, instruction: str = "") -> Image.Image:
        """
        Generate an edited image based on the provided input image and instruction.

        Args:
            input_image (Image.Image): Input image to be edited.
            instruction (str): Instruction to edit the image.
        """
        # Common parameters from config
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

        #--------------------------------------------------------------------#
        # Standard usage for pix_to_pix or any stable diffusion pipeline
        #--------------------------------------------------------------------#
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
            return edited_image

        #--------------------------------------------------------------------#
        # TurboEdit model usage
        #--------------------------------------------------------------------#
        elif self.model_name == 'turbo_edit':
            """
            The TurboEdit class expects the following inputs:
                run(image_path, src_prompt, tgt_prompt, seed, w1, w2)
            
            We only have `instruction` as input. We can decide how to parse
            `instruction` into src_prompt and tgt_prompt. For example, we will assume
            instruction = "Turn the cat into a dog" => (src_prompt="A cat", tgt_prompt="A dog")
            but you can adapt the logic as you wish. For simplicity, let's treat
            the `instruction` as the target prompt, and just keep the same text as source + target.
            """

            if input_image is None:
                raise ValueError("Input image must be provided for 'turbo_edit' model.")
            if not instruction:
                raise ValueError("Instruction must be provided for 'turbo_edit' model.")

            # In your TurboEdit code, you expect an image path, so let's save the input PIL image temporarily
            temp_filename = f"temp_{uuid.uuid4().hex}.png"
            input_image.save(temp_filename)

            # Example usage: 
            #   src_prompt = instruction
            #   tgt_prompt = instruction
            # Or define them differently if you have separate strings
            src_prompt = instruction
            tgt_prompt = instruction

            # Pull w1, w2 from config or default
            w1 = self.model_config.get('w1', 0.8)
            w2 = self.model_config.get('w2', 0.5)

            # Now run the pipeline
            result_image = self.pipe.run(
                image_path=temp_filename,
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
                seed=seed,
                w1=w1,
                w2=w2
            )

            # Remove the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            return result_image
        #--------------------------------------------------------------------#
        # Basic usage for any stable diffusion pipeline
        #--------------------------------------------------------------------#
        else:
            edited_image = self.pipe(
                prompt=instruction,
                num_inference_steps=steps,
                guidance_scale=text_cfg_scale,
                generator=generator,
            ).images[0]
            return edited_image
