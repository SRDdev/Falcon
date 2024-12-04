import os
from PIL import Image
from editor.config import Config
from editor.preprocessor import Preprocessor
from editor.image_editor import TextBasedImageEditor

def main(input_image_path, output_image_path):
    """
    Main function to generate an edited image based on the provided input image.
    """
    config = Config(config_path="../config/config.yaml")
    model_id = config.get("Image-Editor")['model_id']
    
    example_image = Image.open(input_image_path).convert("RGB")

    preprocessor = Preprocessor()
    editor = TextBasedImageEditor(model_id=model_id)

    processed_image = preprocessor.resize_image(example_image)

    edited_image = editor.generate_image(
        input_image=processed_image,
        instruction="Turn it into an anime.",
        steps=50,
        randomize_seed=True,
        seed=1371,
        randomize_cfg=True,
        text_cfg_scale=7.5,
        image_cfg_scale=1.5
    )

    edited_image.save(output_image_path)
    edited_image.show()

if __name__ == "__main__":
    main("/home/shreyas/Desktop/Shreyas/Projects/Text_Based_Image_Editing/data/Input_Image/image_1.jpg", 
         "/home/shreyas/Desktop/Shreyas/Projects/Text_Based_Image_Editing/data/Output_Image/image_1.jpg")
