"""
This file contains the code to watermark a given image using Watermark Anything Model.
"""

import os
from pathlib import Path
import sys
import string
import random
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'watermarkanything')))
from watermark_anything.data.metrics import msg_predict_inference, bit_accuracy, bit_accuracy_inference
from notebooks.inference_utils import load_model_from_checkpoint, default_transform, create_random_mask, torch_to_np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import load_config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
config = load_config(config_path)['Watermarker']['WAM']

class WAM:
    def __init__(self, config):
        """Initialize the WAM model with configuration."""
        self.config = config
        self.device = config['device']
        self.seed = torch.manual_seed(config.get('seed', 42))
        self.epsilon = config['epsilon']
        self.min_samples = config['min_samples']
        self.proportion_masked = config['proportion_masked']
        
        # Load model
        self.exp_dir = config['exp_dir']
        json_path = f"../watermarkers/watermarkanything/{self.exp_dir}/params.json"
        ckpt_path = os.environ.get("CHECKPOINT_MODEL_PATH", self.exp_dir)
        ckpt_file = f"../watermarkers/watermarkanything/{ckpt_path}/checkpoint.pth"
        self.wam = load_model_from_checkpoint(json_path, ckpt_file).to(self.device).eval()
        
        # Define color map for visualization
        self.color_map = {
            -1: [0, 0, 0],      # Black
            0: [255, 0, 255],   # Magenta
            1: [255, 0, 0],     # Red
            2: [0, 255, 0],     # Green
            3: [0, 0, 255],     # Blue
            4: [255, 255, 0],   # Yellow
            5: [0, 255, 255],   # Cyan
        }

    def generate_random_watermark(self, length=8):
        """Generate a random hexadecimal watermark message."""
        return ''.join(random.choice(string.hexdigits[:16]) for _ in range(length))

    def create_random_locations(self, image_size, num_locations=2):
        """Generate random locations for watermark placement.
        
        Args:
            image_size (tuple): (height, width) of the image
            num_locations (int): Number of watermark locations to generate
            
        Returns:
            list: List of tuples (x1, y1, x2, y2) defining watermark regions
        """
        height, width = image_size
        min_size = min(height, width) // 8
        max_size = min(height, width) // 4
        
        locations = []
        for _ in range(num_locations):
            size = random.randint(min_size, max_size)
            x = random.randint(0, width - size)
            y = random.randint(0, height - size)
            locations.append((x, y, x + size, y + size))
        return locations

    def create_location_masks(self, image_size, locations):
        """Create binary masks for specified locations.
        
        Args:
            image_size (tuple): (height, width) of the image
            locations (list): List of (x1, y1, x2, y2) tuples defining regions
            
        Returns:
            torch.Tensor: Binary masks of shape [num_locations, 1, height, width]
        """
        height, width = image_size
        masks = torch.zeros((len(locations), 1, height, width), dtype=torch.float32).to(self.device)
        for idx, (x1, y1, x2, y2) in enumerate(locations):
            masks[idx, 0, y1:y2, x1:x2] = 1
        return masks

    def image_detect(self, img_pil, scan_mult=False, timeout_seconds=5):
        """Detect watermarks in an image.
        
        Args:
            img_pil (PIL.Image): Input image
            scan_mult (bool): Whether to detect multiple watermarks
            timeout_seconds (int): Timeout for multiple watermark detection
            
        Returns:
            tuple: (predicted_mask, message_prediction, bit_accuracy)
        """
        img_pt = default_transform(img_pil).unsqueeze(0).to(self.device)

        # Detect watermark
        preds = self.wam.detect(img_pt)["preds"]  # [1, 33, H, W]
        mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, H, W]
        bit_preds = preds[:, 1:, :, :]  # [1, 32, H, W]

        # Resize mask predictions to match input image size
        mask_preds_res = F.interpolate(
            mask_preds.unsqueeze(1), 
            size=(img_pt.shape[-2], img_pt.shape[-1]), 
            mode="bilinear", 
            align_corners=False
        )  # [1, 1, H, W]

        # Predict message
        message_pred = msg_predict_inference(bit_preds, mask_preds)  # [1, 32]
        print("Message Sum:",message_pred.unsqueeze(0).sum())
        # Calculate bit accuracy if message is found
        bit_acc = None
        if message_pred.sum() > 0:
            # Convert predicted bits to hex string
            binary_str = ''.join(str(int(b.item())) for b in message_pred[0])
            hex_str = format(int(binary_str, 2), '08x')
            print(f"Detected watermark message: {hex_str}")
            
            # Calculate bit accuracy
            if hasattr(img_pil, 'original_message'):
                original_msg = torch.tensor(
                    [int(b) for b in bin(int(img_pil.original_message, 16))[2:].zfill(32)]
                ).to(self.device).unsqueeze(0)
                bit_acc = bit_accuracy(
                    bit_preds, 
                    original_msg, 
                    mask_preds.unsqueeze(1)
                ).item()
                print(f"Bit accuracy: {bit_acc:.4f}")

        return (mask_preds_res > 0.5).float(), message_pred, bit_acc

    def image_embed(self, image, wm_msgs, wm_masks):
        """Embed watermarks in an image.
        
        Args:
            image (PIL.Image): Input image
            wm_msgs (torch.Tensor): Watermark messages to embed
            wm_masks (torch.Tensor): Binary masks defining watermark locations
            
        Returns:
            tuple: (original_tensor, watermarked_tensor, combined_mask)
        """
        img_pt = default_transform(image).unsqueeze(0).to(self.device)
        multi_wm_img = img_pt.clone()
        
        for ii in range(len(wm_msgs)):
            wm_msg, mask = wm_msgs[ii].unsqueeze(0), wm_masks[ii]
            outputs = self.wam.embed(img_pt, wm_msg)
            multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)
            
        return img_pt, multi_wm_img, wm_masks.sum(0)

    def add_random_watermarks(self, img_path, output_dir, num_watermarks=2):
        """Add random watermarks to an image at random locations.
        
        Args:
            img_path (str): Path to input image
            output_dir (str): Directory to save output images
            num_watermarks (int): Number of watermarks to add
        
        Returns:
            tuple: (watermarked_image_path, watermark_messages)
        """
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare image
        try:
            img_pil = Image.open(img_path)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image from {img_path}: {str(e)}")
        
        # Generate random watermark messages
        wm_messages = []
        for _ in range(num_watermarks):
            wm_hex = self.generate_random_watermark()
            binary = bin(int(wm_hex, 16))[2:].zfill(32)
            wm_messages.append([int(b) for b in binary])
        wm_msgs = torch.tensor(wm_messages, dtype=torch.float32).to(self.device)
        
        # Generate random locations and create masks
        locations = self.create_random_locations(img_pil.size[::-1], num_watermarks)
        wm_masks = self.create_location_masks(img_pil.size[::-1], locations)
        
        # Embed watermarks
        _, embed_img_pt, _ = self.image_embed(img_pil, wm_msgs, wm_masks)
        
        # Process and save output image
        output_array = embed_img_pt.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        output_array = ((output_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        output_image = Image.fromarray(output_array)
        output_path = os.path.join(output_dir, f"{Path(img_path).stem}_watermarked.jpg")
        output_image.save(output_path)
        
        # Convert watermark messages to hex strings
        wm_hex_messages = []
        for msg in wm_messages:
            binary_str = ''.join(str(int(b)) for b in msg)
            hex_str = format(int(binary_str, 2), '08x')
            wm_hex_messages.append(hex_str)
        
        return output_path, wm_hex_messages


# Example usage:
if __name__ == "__main__":
    wam = WAM(config)
    
    input_image = "../data/image.jpg"
    output_dir = "../data/"
    
    output_path, watermarks = wam.add_random_watermarks(input_image, output_dir, num_watermarks=2)
    print(f"\nEmbedding Results:")
    print(f"Watermarked image saved to: {output_path}")
    print(f"Watermark messages used: {watermarks}")
    
    # Then detect watermarks
    print(f"\nDetection Results:")
    watermarked_img = Image.open(output_path).convert("RGB")
    # Store original messages for bit accuracy calculation
    watermarked_img.original_message = watermarks[0]  # Using first watermark for simplicity
    
    mask_pred, message_pred, bit_acc = wam.image_detect(watermarked_img)
    
    if message_pred.sum() > 0:
        binary_str = ''.join(str(int(b.item())) for b in message_pred[0])
        detected_hex = format(int(binary_str, 2), '08x')
        print(f"Detected watermark: {detected_hex}")
        if bit_acc is not None:
            print(f"Bit accuracy: {bit_acc:.4f}")
    else:
        print("No watermark detected")
