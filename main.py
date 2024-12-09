import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watermarking/ZoDiac'))
import torch
import pandas as pd
from config.config import Config
from editor.editor import TextBasedImageEditor
from watermarking.watermark import Watermarker  # Import the Watermarker class
from PIL import Image

#-----------------------------------Main-----------------------------------#
def main(config, watermarking_technique, editing_technique, num_iterations=3):
    """
    Main function for Falcon. This function processes images in several stages: 
    1. Watermarks the image, saves it.
    2. Edits the watermarked image using instructions from a CSV, saves it.
    3. Calculates watermark detection probability, logs it.
    4. Repeats the process for `num_iterations` times.

    Args:
        config (Config): Configuration object for the Falcon project.
        num_iterations (int, optional): Number of iterations to perform on each image. Defaults to 3.
    """
    #-----------------------------------Directories-----------------------------------#
    image_dir = config.get('dirs')['original_image']
    watermarked_dir = config.get('dirs')['watermarked_image']
    edited_dir = config.get('dirs')['generated_image_1']
    instruction_csv = config.get('dirs')['instruction_csv']
    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)

    #-----------------------------------Load the captions csv-----------------------------------#
    instructions_df = pd.read_csv(instruction_csv)


    detection_probabilities = []
    model_name = config.get(f'{editing_technique}')['model_id']
    watermarker = Watermarker(watermarking_technique=watermarking_technique)
    editor = TextBasedImageEditor(model_name)

    for i in range(1, num_iterations + 1):
        image_filename = f"images_{i}.png"
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: {image_filename} not found in {image_dir}")
            continue

        #------------------------------------------Step 1: Apply watermark to the image------------------------------------------#
        print(f"Processing {image_filename}...")
        watermarked_image_path, det_prob = watermarker.apply_watermark(image_path)
        watermarked_image = Image.open(watermarked_image_path)

        #-------------------------------------------------Step 2: Edit the watermarked image using the TextBasedImageEditor-------------------------------------------------#
        image_instructions = instructions_df[instructions_df['image_filename'] == image_filename].iloc[0, 1:].values
        if len(image_instructions) < num_iterations:
            print(f"Warning: Not enough instructions for {image_filename}. Using available instructions.")
        
        # Process the iterations for each image
        for j in range(min(num_iterations, len(image_instructions))):
            # Step 3: Edit the watermarked image using the TextBasedImageEditor
            instruction = image_instructions[j]  # Get the j-th instruction for this iteration
            edited_image = editor.generate_image(input_image=watermarked_image, instruction=instruction)
            edited_image_path = os.path.join(edited_dir, f"generated_image_{i}_{j + 1}.png")
            edited_image.save(edited_image_path)

            # Step 4: Calculate watermark detection probability for the edited image
            detection_prob = watermarker.calculate_detection_probability(edited_image_path)
            detection_probabilities.append({
                'image': image_filename,
                'iteration': j + 1,
                'detection_probability': detection_prob
            })

    # Step 5: Save the detection probabilities to a CSV file
    probabilities_df = pd.DataFrame(detection_probabilities)
    probabilities_df.to_csv(os.path.join(image_dir, 'watermark_detection_probabilities.csv'), index=False)

    print("Processing complete. Results saved.")

if __name__ == "__main__":
    
    config_path = "config/config.yaml"
    config = Config(config_path)
    watermarking_technique = "ZoDiac"
    editing_technique = "pix2pix"
    num_iterations = 5
    
    main(config,watermarking_technique,editing_technique,num_iterations)
