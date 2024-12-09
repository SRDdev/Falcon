import os
import torch
import torch.optim as optim
from diffusers import DDIMScheduler
import yaml
import logging
from ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline
from ZoDiac.main.wmpatch import GTWatermark, GTWatermarkMulti
from ZoDiac.main.utils import *
from ZoDiac.loss.loss import LossProvider
from ZoDiac.loss.pytorch_ssim import ssim

class ZodiacWatermarking:
    """
    A class to handle the watermarking process using Zodiac.
    The class processes an image by applying watermarking, training the model, and calculating detection probabilities.

    Args:
        config_path (str): Path to the YAML configuration file for settings.
    """

    def __init__(self, config_path: str = 'ZoDiac/example/config/config.yaml'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.config_path = config_path

        # Load configuration from YAML
        self.cfgs = self.load_config(config_path)
        
        # Set up watermarking pipeline
        self.wm_pipe = self.setup_watermarking_pipeline()
        self.scheduler = DDIMScheduler.from_pretrained(self.cfgs['model_id'], subfolder="scheduler")
        self.pipe = WMDetectStableDiffusionPipeline.from_pretrained(self.cfgs['model_id'], scheduler=self.scheduler).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # Prepare training setup
        self.optimizer, self.loss_provider = self.setup_training()

        # Save path for watermarked images
        self.wm_path = self.cfgs['save_img']

    def load_config(self, config_path):
        """Load the configuration from the given YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def setup_watermarking_pipeline(self):
        """Initialize the watermarking pipeline based on the configuration."""
        if self.cfgs['w_type'] == 'single':
            return GTWatermark(self.device, w_channel=self.cfgs['w_channel'], w_radius=self.cfgs['w_radius'], generator=torch.Generator(self.device).manual_seed(self.cfgs['w_seed']))
        elif self.cfgs['w_type'] == 'multi':
            return GTWatermarkMulti(self.device, w_settings=self.cfgs['w_settings'], generator=torch.Generator(self.device).manual_seed(self.cfgs['w_seed']))
        else:
            raise ValueError("Unknown watermark type. Please check the config.")

    def setup_training(self):
        """Initialize the optimizer and loss provider for training the watermarking model."""
        # Placeholder initialization of latents (for the sake of simplicity here)
        init_latents_approx = torch.randn((1, 3, 256, 256), device=self.device)  # Example tensor
        init_latents = init_latents_approx.detach().clone()
        init_latents.requires_grad = True
        optimizer = optim.Adam([init_latents], lr=0.01)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3) 
        loss_provider = LossProvider(self.cfgs['loss_weights'], self.device)
        return optimizer, loss_provider

    def get_init_latent(self, img_tensor, text_embeddings, guidance_scale=1.0):
        """Generate the initial latent for the image."""
        img_latents = self.pipe.get_image_latents(img_tensor, sample=False)
        reversed_latents = self.pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
        )
        return reversed_latents

    def process_image(self, image_path: str):
        """
        Process the image by applying the watermark and training the model.

        Args:
            image_path (str): Path to the image to be watermarked.

        Returns:
            tuple: Path to the watermarked image and detection probability.
        """
        # Load the image tensor
        self.gt_img_tensor = get_img_tensor(image_path, self.device)

        # Step 1: Get initial latent
        init_latents_approx = self.get_init_latent(self.gt_img_tensor, self.pipe.get_text_embedding(''))
        
        # Step 2: Train the initial latents
        loss_lst = []
        for i in range(self.cfgs['iters']):
            logging.info(f'Iter {i}:')
            init_latents_wm = self.wm_pipe.inject_watermark(init_latents_approx)
            pred_img_tensor = self.pipe(
                '', guidance_scale=1.0, num_inference_steps=50, 
                output_type='tensor', use_trainable_latents=True, 
                init_latents=init_latents_wm
            ).images
            
            loss = self.loss_provider(pred_img_tensor, self.gt_img_tensor, init_latents_wm, self.wm_pipe)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_lst.append(loss.item())
            if (i + 1) in self.cfgs['save_iters']:
                save_img(os.path.join(self.wm_path, f"{os.path.basename(image_path).split('.')[0]}_{i+1}.png"), pred_img_tensor, self.pipe)

        # Step 3: Postprocessing
        wm_img_path = os.path.join(self.wm_path, f"{os.path.basename(image_path).split('.')[0]}_{self.cfgs['save_iters'][-1]}.png")
        self.wm_img_tensor = get_img_tensor(wm_img_path, self.device)
        ssim_value = ssim(self.wm_img_tensor, self.gt_img_tensor).item()
        logging.info(f'Original SSIM: {ssim_value}')
        
        # Find optimal theta for postprocessing
        optimal_theta = self.binary_search_theta(ssim_value)
        img_tensor = (self.gt_img_tensor - self.wm_img_tensor) * optimal_theta + self.wm_img_tensor

        ssim_value = ssim(img_tensor, self.gt_img_tensor).item()
        psnr_value = compute_psnr(img_tensor, self.gt_img_tensor)

        tester_prompt = ''  # Placeholder for any test prompt if needed
        text_embeddings = self.pipe.get_text_embedding(tester_prompt)
        det_prob = 1 - watermark_prob(img_tensor, self.pipe, self.wm_pipe, text_embeddings)

        # Save final image after postprocessing
        path = os.path.join(self.wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{self.cfgs['ssim_threshold']}.png")
        save_img(path, img_tensor, self.pipe)

        # Return the path to the watermarked image and detection probability
        return path, det_prob

    def calculate_detection_score(self, image_path: str):
        """
        Calculate the watermark detection probability for the given image.

        Args:
            image_path (str): Path to the image to calculate the detection score.

        Returns:
            float: Watermark detection probability (between 0 and 1).
        """
        img_tensor = get_img_tensor(image_path, self.device)
        text_embeddings = self.pipe.get_text_embedding('')
        det_prob = 1 - watermark_prob(img_tensor, self.pipe, self.wm_pipe, text_embeddings)
        return det_prob

    def binary_search_theta(self, ssim_threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
        """Binary search to find the optimal theta value."""
        for i in range(max_iter):
            mid_theta = (lower + upper) / 2
            img_tensor = (self.gt_img_tensor - self.wm_img_tensor) * mid_theta + self.wm_img_tensor
            ssim_value = ssim(img_tensor, self.gt_img_tensor).item()

            if ssim_value <= ssim_threshold:
                lower = mid_theta
            else:
                upper = mid_theta
            if upper - lower < precision:
                break
        return lower

# Example usage:
# Initialize the class with the config path
processor = ZodiacWatermarking(config_path='ZoDiac/example/config/config.yaml')

# Process the image and get the watermarked image path and detection probability
image_path = 'path/to/your/image.png'
watermarked_image_path, det_prob = processor.process_image(image_path)
print(f"Watermarked image saved at: {watermarked_image_path}")
print(f"Watermark Detection Probability: {det_prob}")

# Calculate the detection score for any image
detection_score = processor.calculate_detection_score(watermarked_image_path)
print(f"Watermark Detection Probability for the final image: {detection_score}")
