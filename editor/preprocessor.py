from PIL import Image, ImageOps
import math

class Preprocessor:
    @staticmethod
    def resize_image(input_image: Image.Image) -> Image.Image:
        """Resize the input image to be compatible with the model."""
        width, height = input_image.size
        factor = 512 / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        return ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
