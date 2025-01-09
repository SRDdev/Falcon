from PIL import Image, ImageOps
import math

class Preprocessor:
    @staticmethod
    def resize_image(input_image: Image.Image) -> Image.Image:
        """
        Resize the input image to be compatible with the model. Ensures the dimensions are multiples of 64.
        
        Args:
            input_image (Image.Image): The input image to resize.
            
        Returns:
            Image.Image: The resized image.
        """
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        return ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    @staticmethod
    def preprocess_control_image(control_image: Image.Image) -> Image.Image:
        """
        Preprocess the control image for ControlNet by converting it to grayscale and resizing it.

        Args:
            control_image (Image.Image): The input control image.
        
        Returns:
            Image.Image: The preprocessed control image.
        """
        # Convert to grayscale if necessary
        if control_image.mode != "L":
            control_image = control_image.convert("L")
        
        # Resize the image to be compatible with the model
        control_image = Preprocessor.resize_image(control_image)
        
        return control_image
