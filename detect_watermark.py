import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from editor.editor import TextBasedImageEditor
from editor.preprocessor import Preprocessor

def detect_watermark(edited_image_path):
    """
    Detects watermark in the already edited image by analyzing pixel differences.

    Args:
        edited_image_path (str): Path to the edited image.

    Returns:
        bool: True if a watermark is detected, otherwise False.
    """
    # Load the edited image
    edited_image = Image.open(edited_image_path).convert("RGB")
    
    # Convert the image to a numpy array for analysis
    edited_array = np.array(edited_image)
    
    # Convert the image to grayscale to focus on intensity changes
    gray_image = cv2.cvtColor(edited_array, cv2.COLOR_RGB2GRAY)
    
    # Apply a Gaussian blur to reduce noise and highlight significant changes
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Threshold the blurred image to detect significant changes
    _, thresh = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY)
    
    # Count the number of white pixels (areas with significant changes)
    non_zero_pixels = np.count_nonzero(thresh)
    
    # If the number of significant pixels exceeds a threshold, assume watermark presence
    if non_zero_pixels > 1000:  # You can adjust this threshold based on your needs
        print("Watermark detected!")
        return True
    else:
        print("No watermark detected.")
        return False

def main(image_path):
    """
    Main function to detect watermark in the edited image.

    Args:
        image_path (str): Path to the edited image.
    """
    # Call watermark detection on the provided image path
    detect_watermark(image_path)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Detect watermark in the edited image.")
    #parser.add_argument('--image', required=True, help="Path to the edited image.")

    #args = parser.parse_args()

    # Run the main function
    main("image_1.jpg")
