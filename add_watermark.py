from PIL import Image, ImageDraw, ImageFont
from editor.editor import TextBasedImageEditor
from editor.preprocessor import Preprocessor

def add_watermark(input_image_path, watermark_text, output_image_path):
    """
    Adds a watermark to an image.

    Args:
        input_image_path (str): Path to the input image.
        watermark_text (str): Text to be used as the watermark.
        output_image_path (str): Path to save the output image.
    """
    # Open the input image
    original_image = Image.open(input_image_path).convert("RGBA")
    
    # Make a copy of the image to add watermark on
    image_with_watermark = original_image.copy()

    # Get image dimensions
    width, height = original_image.size

    # Prepare the watermark text
    draw = ImageDraw.Draw(image_with_watermark)
    font = ImageFont.load_default()  # You can change this to a custom font if needed

    # Calculate the width and height of the watermark using textbbox
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position the watermark at the bottom-right corner
    x_position = width - text_width - 10
    y_position = height - text_height - 10

    # Add watermark text to the image
    draw.text((x_position, y_position), watermark_text, font=font, fill=(255, 255, 255, 128))  # RGBA for transparency

    # Convert the image to RGB before saving as JPEG (JPEG doesn't support RGBA)
    image_with_watermark_rgb = image_with_watermark.convert("RGB")

    # Save the watermarked image
    image_with_watermark_rgb.save(output_image_path)
    print(f"Watermarked image saved to: {output_image_path}")

def main(model_name, input_image_path, output_image_path, watermark_text, final_output_path):
    """
    Main function to generate an edited image and add a watermark.

    Args:
        model_name (str): Name of the model for editing the image.
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the edited image.
        watermark_text (str): Text for the watermark.
        final_output_path (str): Path to save the final watermarked image.
    """
    # Preprocess the image
    example_image = Image.open(input_image_path).convert("RGB")
    preprocessor = Preprocessor()
    processed_image = preprocessor.resize_image(example_image)

    # Use Pix2Pix model for image editing (with a default instruction)
    editor = TextBasedImageEditor(model_name=model_name)
    instruction = "Transform the image"  # Default instruction if none is provided
    edited_image = editor.generate_image(input_image=processed_image, instruction=instruction)

    # Save the edited image
    edited_image.save(output_image_path)

    # Add watermark to the edited image
    add_watermark(output_image_path, watermark_text, final_output_path)

    print(f"Watermarked image saved to: {final_output_path}")

if __name__ == "__main__":
    # Example configuration
    model_name = "pix_to_pix"
    image_path = "image_1.jpg"
    output_image_path = "edited_image.jpg"
    watermark_text = "Sample Watermark"
    final_output_path = "watermarked_image.jpg"

    # Run the main function
    main(model_name, image_path, output_image_path, watermark_text, final_output_path)
