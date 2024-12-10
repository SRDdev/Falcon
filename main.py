import os
from PIL import Image
from editor.config import Config
from editor.preprocessor import Preprocessor
from editor.editor import TextBasedImageEditor

def main(model_name, input_image_path, output_image_path, instruction):
    """
    Main function to generate an edited image based on the provided input image and instruction.

    Args:
        model_name (str): Name of the model to be used for editing the image.
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the output edited image.
        instruction (str): Text instruction for editing the image.
    """
    # Load and preprocess the image
    example_image = Image.open(input_image_path).convert("RGB")
    preprocessor = Preprocessor()
    processed_image = preprocessor.resize_image(example_image)
    
    # Initialize the editor with the specified model
    editor = TextBasedImageEditor(model_name=model_name)
    
    # Generate the edited image based on the instruction
    edited_image = editor.generate_image(input_image=processed_image, instruction=instruction)
    
    # Save and display the edited image
    edited_image.save(output_image_path)
    print(f"Edited image saved to: {output_image_path}")
    edited_image.show()

if __name__ == "__main__":
    # Configuration
    model_name = "pix_to_pix"
    image_path = "image_1.jpg"
    output_image_path = "edit_image_1.jpg"
    instruction = "Turn it into an anime."

    # Run the main function
    main(model_name, image_path, output_image_path, instruction)
