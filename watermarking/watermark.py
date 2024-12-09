"""Acts as a common class to load any watermarking technique and use it."""

from .zodiac import ZodiacWatermarking

class Watermarker:
    """
    Class to apply watermarking using a specified technique and calculate detection probabilities.
    
    Args:
        watermarking_technique (str): Watermarking technique to be used. E.g., "ZoDiac".
    """

    def __init__(self, watermarking_technique):
        if watermarking_technique not in ["ZoDiac"]:
            raise ValueError(f"Unsupported watermarking technique: {watermarking_technique}")

        self.watermarking_technique = watermarking_technique
        self.detection_probabilities = [] 
        self.watermarked_image_paths = []

        if self.watermarking_technique == "ZoDiac":
            self.watermarker = ZodiacWatermarking()
            
    def apply_watermark(self, image_path):
        """Applies watermarking to the image and returns the watermarked image path and detection probability."""
        if self.watermarking_technique == "ZoDiac":
            watermarked_image_path, det_prob = self.watermarker.process_image(image_path)
            self.watermarked_image_paths.append(watermarked_image_path)
            self.detection_probabilities.append(det_prob)

            return watermarked_image_path, det_prob
        else:
            raise NotImplementedError(f"Watermarking technique '{self.watermarking_technique}' is not implemented.")

    def calculate_detection_probability(self, image_path):
        """Calculates and returns the watermark detection probability for the given image."""
        if self.watermarking_technique == "ZoDiac":
            detection_score = self.watermarker.calculate_detection_score(image_path=image_path)
            print(f"Watermark Detection Probability: {detection_score}")
            return detection_score
        else:
            raise NotImplementedError(f"Watermarking technique '{self.watermarking_technique}' is not implemented.")
