"""
Wrapper for DWT-DCT watermarking.
Supports watermarking of images using an watermark image.
"""

import os
import json
import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct



def convert_image(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img = img.convert('L')
    image_array = np.array(img.getdata(), dtype=float).reshape((size, size))
    return image_array

def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    return list(coeffs)

def apply_dct(image_array):
    size = image_array.shape[0]
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct

def inverse_dct(all_subdct):
    size = all_subdct.shape[0]
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct
    return all_subidct

def embed_watermark(watermark_array, orig_image):
    watermark_flat = watermark_array.ravel()
    ind = 0
    for x in range(0, orig_image.shape[0], 8):
        for y in range(0, orig_image.shape[1], 8):
            if ind < len(watermark_flat):
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1
    return orig_image

def get_watermark(dct_watermarked_coeff, watermark_size):
    subwatermarks = []
    for x in range(0, dct_watermarked_coeff.shape[0], 8):
        for y in range(0, dct_watermarked_coeff.shape[1], 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])
    return np.array(subwatermarks).reshape(watermark_size, watermark_size)

def recover_watermark(image_array, model='haar', level=1, watermark_size=128, save_path=None):
    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    watermark_array = get_watermark(dct_watermarked_coeff, watermark_size)
    watermark_array = np.uint8(watermark_array)
    img = Image.fromarray(watermark_array)
    if save_path:
        img.save(save_path)
    return img

def print_image_from_array(image_array, name):
    image_array_copy = np.clip(image_array, 0, 255).astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save(name)
    return img





class DwtDctWrapper:
    def __init__(self, model="haar", level=1, image_size=2048, watermark_size=128):
        self.model = model
        self.level = level
        self.image_size = image_size
        self.watermark_size = watermark_size

    def watermark_image(self, image_path: str, watermark_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess
        image_array = convert_image(image_path, self.image_size)
        watermark_array = convert_image(watermark_path, self.watermark_size)

        # Wavelet decomposition
        coeffs_image = process_coefficients(image_array, self.model, level=self.level)

        # Apply DCT + embed watermark
        dct_array = apply_dct(coeffs_image[0])
        dct_array = embed_watermark(watermark_array, dct_array)

        # Inverse DCT
        coeffs_image[0] = inverse_dct(dct_array)

        # Reconstruction
        image_array_H = pywt.waverec2(coeffs_image, self.model)
        wm_img_path = os.path.join(output_dir, "image_with_watermark.jpg")
        print_image_from_array(image_array_H, wm_img_path)

        return wm_img_path, image_array_H

    def detect_watermark(self, watermarked_image_array, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        wm_recovered_path = os.path.join(output_dir, "recovered_watermark.jpg")
        recovered_img = recover_watermark(
            image_array=watermarked_image_array,
            model=self.model,
            level=self.level,
            watermark_size=self.watermark_size,
            save_path=wm_recovered_path
        )
        return wm_recovered_path, recovered_img






def process_sample(sample_dir: str, image_file: str, watermark_file: str):
    dwt_dct = DwtDctWrapper()

    image_path = os.path.join(sample_dir, "original", image_file)
    watermark_path = os.path.join(sample_dir, "original", watermark_file)
    out_dir = os.path.join(sample_dir, "watermarking", "DWTDCT")
    os.makedirs(out_dir, exist_ok=True)

    # Embed
    wm_img_path, wm_image_array = dwt_dct.watermark_image(
        image_path=image_path,
        watermark_path=watermark_path,
        output_dir=out_dir
    )

    # Detect
    wm_recovered_path, _ = dwt_dct.detect_watermark(
        watermarked_image_array=wm_image_array,
        output_dir=out_dir
    )

    # Save report
    report = {
        "original_image": image_path,
        "watermark_image": watermark_path,
        "watermarked_output": wm_img_path,
        "recovered_watermark": wm_recovered_path,
        "parameters": {
            "model": dwt_dct.model,
            "level": dwt_dct.level,
            "image_size": dwt_dct.image_size,
            "watermark_size": dwt_dct.watermark_size
        }
    }
    with open(os.path.join(out_dir, "detection_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return {
        "output_dir": out_dir,
        "generation": wm_img_path,
        "detection": wm_recovered_path,
        "report": report
    }

if __name__ == "__main__":
    sample_dir = "experiments/image_000000000003"
    results = process_sample(
        sample_dir=sample_dir,
        image_file="image.png",
        watermark_file="watermark.png"
    )

    print("Output dir:", results["output_dir"])
    print("Watermarked image saved at:", results["generation"])
    print("Recovered watermark saved at:", results["detection"])