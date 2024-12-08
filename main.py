import os
from PIL import Image
from editor.config import Config
from editor.preprocessor import Preprocessor
from editor.editor import TextBasedImageEditor

def main(model_name, input_image_path, output_image_path):
    """
    Main function to generate an edited image based on the provided input image.

    Args:
        config (Config): Configuration object containing the model configurations
        model_name (str): Name of the model to be used for editing the image
        input_image_path (str): Path to the input image
        output_image_path (str): Path to save the output edited image
    """
    example_image = Image.open(input_image_path).convert("RGB")
    preprocessor = Preprocessor()
    editor = TextBasedImageEditor(model_name=model_name)
    processed_image = preprocessor.resize_image(example_image)
    edited_image = editor.generate_image(input_image=processed_image, instruction="Turn it into an anime.")
    edited_image.save(output_image_path)
    edited_image.show()

if __name__ == "__main__":
    model_name = "pix_to_pix"
    image_path = "/home/shreyas/Desktop/Shreyas/Projects/Text_Based_Image_Editing/data/Input_Image/image_1.jpg"
    output_image_path = "/home/shreyas/Desktop/Shreyas/Projects/Text_Based_Image_Editing/data/Output_Image/image_1.jpg"
    main(model_name, image_path, output_image_path)
