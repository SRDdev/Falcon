# import os
# import json
# from PIL import Image
# from datasets import load_dataset
# from tqdm import tqdm
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from watermarkers.WAM import WAM
# from editors.pix2pix import Pix2Pix
# from config.config import load_config

# config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
# config_pix2pix = load_config(config_path)['Editor']['pix2pix']
# config_wam = load_config(config_path)['Watermarker']['WAM']    

# def watermark_images(wam, ds, output_base_dir):
#     """Apply watermarks to all images in the dataset"""
#     watermarked_dir = os.path.join(output_base_dir, "watermarked")
#     os.makedirs(watermarked_dir, exist_ok=True)
    
#     watermark_results = []
    
#     for idx, item in enumerate(tqdm((ds["V1"]), desc="Watermarking")):
#         if idx == 1:
#             image = item["image"]
#             image_id = item["id"]
            
#             # Save temporary image
#             temp_image_path = os.path.join(output_base_dir, f"temp_{image_id}.jpg")
#             image.save(temp_image_path)
            
#             # Add watermark
#             watermarked_path = os.path.join(watermarked_dir, f"{image_id}.jpg")
#             watermarked_path, watermarks = wam.add_random_watermarks(
#                 temp_image_path,
#                 watermarked_dir,
#                 num_watermarks=2
#             )
            
#             # Store results
#             watermark_results.append({
#                 "image_id": image_id,
#                 "original_prompt": item["source_prompt"],
#                 "target_prompt": item["target_prompt"],
#                 "watermarked_path": watermarked_path,
#                 "watermark_messages": watermarks
#             })
            
#             # Clean up
#             os.remove(temp_image_path)
            
#     return watermark_results

# def edit_images(model, watermark_results, output_base_dir):
#     """Edit all watermarked images"""
#     edited_dir = os.path.join(output_base_dir, "edited")
#     os.makedirs(edited_dir, exist_ok=True)
    
#     edit_results = []
    
#     for item in tqdm(watermark_results, desc="Editing"):
#         image_id = item["image_id"]
#         watermarked_path = item["watermarked_path"]
#         target_prompt = item["target_prompt"]
        
#         # Edit image
#         edited_path = os.path.join(edited_dir, f"{image_id}_edited.jpg")
#         model.edit_image(watermarked_path, target_prompt, edited_path)
        
#         # Store results
#         edit_results.append({
#             **item,  # Include all previous information
#             "edited_path": edited_path
#         })
    
#     return edit_results

# def detect_watermarks(wam, edit_results):
#     """Detect watermarks in all edited images"""
#     detection_results = []
    
#     for item in tqdm(edit_results, desc="Detecting"):
#         edited_path = item["edited_path"]
#         watermarks = item["watermark_messages"]
        
#         # Detect watermark
#         edited_img = Image.open(edited_path).convert("RGB")
#         edited_img.original_message = watermarks[0]  # Using first watermark
        
#         mask_pred, message_pred, bit_acc = wam.image_detect(edited_img)
        
#         # Process detection results
#         detection_result = "not_detected"
#         detected_hex = None
#         if message_pred.sum() > 0:
#             binary_str = ''.join(str(int(b.item())) for b in message_pred[0])
#             detected_hex = format(int(binary_str, 2), '08x')
#             detection_result = "detected"
        
#         # Store results
#         detection_results.append({
#             **item,  # Include all previous information
#             "detection_result": detection_result,
#             "detected_watermark": detected_hex,
#             "bit_accuracy": float(bit_acc) if bit_acc is not None else None
#         })
    
#     return detection_results

# def save_results(results, output_base_dir, filename="results.json"):
#     """Save results to JSON file"""
#     output_path = os.path.join(output_base_dir, filename)
#     with open(output_path, "w") as f:
#         json.dump({"processed_images": results}, f, indent=2)

# if __name__ == "__main__":
#     # Setup
#     output_base_dir = "../data/output"
#     os.makedirs(output_base_dir, exist_ok=True)
    
#     # Initialize models
#     wam = WAM(config_wam)
#     model = Pix2Pix(config_pix2pix)
    
#     # Load dataset
#     ds = load_dataset("UB-CVML-Group/PIE_Bench_pp", "1_change_object_80")
    
#     # Process pipeline
#     print("-"*100)
#     print("Stage 1: Watermarking images...")
#     watermark_results = watermark_images(wam, ds, output_base_dir)
#     save_results(watermark_results, output_base_dir, "watermark_results.json")
#     print("-"*100)
#     print("Stage 2: Editing images...")
#     edit_results = edit_images(model, watermark_results, output_base_dir)
#     save_results(edit_results, output_base_dir, "edit_results.json")
#     print("-"*100)
#     print("Stage 3: Detecting watermarks...")
#     final_results = detect_watermarks(wam, edit_results)
#     save_results(final_results, output_base_dir, "final_results.json")

import os
import json
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import wandb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from watermarkers.WAM import WAM
from editors.pix2pix import Pix2Pix
from config.config import load_config
from wandb_API import API
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
config = load_config(config_path)
config_pix2pix = config['Editor']['pix2pix']
config_wam = config['Watermarker']['WAM']
config_wandb = config['WandB']

# Initialize wandb
wandb.login(key=API)
wandb.init(project=config_wandb['project_name'], name=config_wandb['run_name'], config={
    "watermarker_config": config_wam,
    "editor_config": config_pix2pix,
})

def watermark_images(wam, ds, output_base_dir):
    """Apply watermarks to all images in the dataset"""
    watermarked_dir = os.path.join(output_base_dir, "watermarked")
    os.makedirs(watermarked_dir, exist_ok=True)
    
    watermark_results = []
    
    for idx, item in enumerate(tqdm((ds["V1"]), desc="Watermarking")):
        image = item["image"]
        image_id = item["id"]
        
        # Save temporary image
        temp_image_path = os.path.join(output_base_dir, f"temp_{image_id}.jpg")
        image.save(temp_image_path)
        
        # Add watermark
        watermarked_path = os.path.join(watermarked_dir, f"{image_id}.jpg")
        watermarked_path, watermarks = wam.add_random_watermarks(
            temp_image_path,
            watermarked_dir,
            num_watermarks=2
        )
        
        # Log initial image and watermarked image to wandb
        wandb.log({
            "initial_image": wandb.Image(image, caption=f"Initial Image (ID: {image_id})"),
            "watermarked_image": wandb.Image(watermarked_path, caption=f"Watermarked Image (ID: {image_id})"),
            "initial_prompt": item["source_prompt"],
            "watermark_messages": watermarks,
        })
        
        # Store results
        watermark_results.append({
            "image_id": image_id,
            "original_prompt": item["source_prompt"],
            "target_prompt": item["target_prompt"],
            "watermarked_path": watermarked_path,
            "watermark_messages": watermarks
        })
        
        # Clean up
        os.remove(temp_image_path)
            
    return watermark_results

def edit_images(model, watermark_results, output_base_dir):
    """Edit all watermarked images"""
    edited_dir = os.path.join(output_base_dir, "edited")
    os.makedirs(edited_dir, exist_ok=True)
    
    edit_results = []
    
    for item in tqdm(watermark_results, desc="Editing"):
        image_id = item["image_id"]
        watermarked_path = item["watermarked_path"]
        target_prompt = item["target_prompt"]
        
        # Edit image
        edited_path = os.path.join(edited_dir, f"{image_id}_edited.jpg")
        model.edit_image(watermarked_path, target_prompt, edited_path)
        
        # Log edited image to wandb
        wandb.log({
            "edited_image": wandb.Image(edited_path, caption=f"Edited Image (ID: {image_id})"),
            "edited_prompt": target_prompt,
        })
        
        # Store results
        edit_results.append({
            **item,  # Include all previous information
            "edited_path": edited_path
        })
    
    return edit_results

def detect_watermarks(wam, edit_results):
    """Detect watermarks in all edited images"""
    detection_results = []
    
    for item in tqdm(edit_results, desc="Detecting"):
        edited_path = item["edited_path"]
        watermarks = item["watermark_messages"]
        
        # Detect watermark
        edited_img = Image.open(edited_path).convert("RGB")
        edited_img.original_message = watermarks[0]  # Using first watermark
        
        mask_pred, message_pred, bit_acc = wam.image_detect(edited_img)
        
        # Process detection results
        detection_result = "not_detected"
        detected_hex = None
        if message_pred.sum() > 0:
            binary_str = ''.join(str(int(b.item())) for b in message_pred[0])
            detected_hex = format(int(binary_str, 2), '08x')
            print(f"Detected watermark: {detected_hex}")
            if bit_acc is not None:
                print(f"Bit accuracy: {bit_acc:.4f}")
        else:
            print("No watermark detected")
        
        # Log detection results to wandb
        wandb.log({
            "edited_image_with_detection": wandb.Image(edited_path, caption=f"Detection on Image (ID: {item['image_id']})"),
            "detection_result": detection_result,
            "detected_watermark": detected_hex,
            "bit_accuracy": float(bit_acc) if bit_acc is not None else None,
        })
        
        # Store results
        detection_results.append({
            **item,  # Include all previous information
            "detection_result": detection_result,
            "detected_watermark": detected_hex,
            "bit_accuracy": float(bit_acc) if bit_acc is not None else None
        })
    
    return detection_results

def save_results(results, output_base_dir, filename="results.json"):
    """Save results to JSON file"""
    output_path = os.path.join(output_base_dir, filename)
    with open(output_path, "w") as f:
        json.dump({"processed_images": results}, f, indent=2)

if __name__ == "__main__":
    # Setup
    output_base_dir = "../data/output"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Initialize models
    wam = WAM(config_wam)
    model = Pix2Pix(config_pix2pix)
    
    # Load dataset
    ds = load_dataset("UB-CVML-Group/PIE_Bench_pp", "0_random_140")
    
    # Process pipeline
    print("-"*100)
    print("Stage 1: Watermarking images...")
    watermark_results = watermark_images(wam, ds, output_base_dir)
    save_results(watermark_results, output_base_dir, "watermark_results.json")
    print("-"*100)
    print("Stage 2: Editing images...")
    edit_results = edit_images(model, watermark_results, output_base_dir)
    save_results(edit_results, output_base_dir, "edit_results.json")
    print("-"*100)
    print("Stage 3: Detecting watermarks...")
    final_results = detect_watermarks(wam, edit_results)
    save_results(final_results, output_base_dir, "final_results.json")

    # Finish wandb logging
    wandb.finish()
