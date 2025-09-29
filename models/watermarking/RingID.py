"""
Wrapper for RingID watermarking.
Supports watermark embedding/detection for enhanced multi-key identification.
"""


import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import itertools


from diffusers import DPMSolverMultistepScheduler
import open_clip


from configs.logger import setup_logger
from configs.config import Config


from external.RingID.inverse_stable_diffusion import InversableStableDiffusionPipeline
from external.RingID.utils import (
    generate_Fourier_watermark_latents, make_Fourier_ringid_pattern,
    ring_mask, fft, ifft, get_distance, transform_img, image_distortion,
    set_random_seed, measure_similarity
)


logger = setup_logger("RingID", "logs", "debug")



class RingIDWrapper:
    def __init__(self, model_name="ringid_model1",
                 model_id="stabilityai/stable-diffusion-2-1-base",
                 reference_model="ViT-g-14",
                 reference_model_pretrain="laion2b_s12b_b42k",
                 device=None, online=True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.online = online
        logger.info(f"Using device: {self.device}")


        self.RADIUS = 14
        self.RADIUS_CUTOFF = 3
        self.HETER_WATERMARK_CHANNEL = [0]  # Gaussian noise watermark
        self.RING_WATERMARK_CHANNEL = [3]   # Ring watermark
        self.WATERMARK_CHANNEL = [0, 3]     # Both channels


        self.fix_gt = True          # Lossless imprinting (real-part only)
        self.time_shift = True      # Spatial shift for rotation robustness  
        self.channel_min = True     # Multi-channel heterogeneous detection
        self.USE_ROUNDER_RING = True # Better rotation-invariant ring masks


        logger.info("Loading RingID diffusion model...")
        self._load_model(model_id, reference_model, reference_model_pretrain)
        logger.info("RingID model loaded successfully")


    def _load_model(self, model_id: str, reference_model: str, reference_model_pretrain: str):
        model_dtype = torch.float16
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder='scheduler', local_files_only=(not self.online)
        )


        if self.online:
            pipeline_pretrain = model_id
            reference_model_pretrain_path = reference_model_pretrain
        else:
            home = os.path.expanduser("~")
            pipeline_pretrain = f"{home}/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base"
            reference_model_pretrain_path = f"{home}/.cache/huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/open_clip_pytorch_model.bin"


        self.pipeline = InversableStableDiffusionPipeline.from_pretrained(
            pipeline_pretrain, scheduler=scheduler, torch_dtype=model_dtype, revision="fp16"
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)


        if reference_model:
            self.ref_model, _, self.ref_clip_preprocess = open_clip.create_model_and_transforms(
                reference_model, pretrained=reference_model_pretrain_path, device=self.device
            )
            self.ref_tokenizer = open_clip.get_tokenizer(reference_model)
        else:
            self.ref_model = None


    def _create_watermark_patterns(self,
                              num_keys: int = 2,
                              ring_value_range: int = 64,
                              quantization_levels: int = 2):
        """
        RingID watermark pattern generation.

        Steps:
        1) Build per-slot options: for each ring slot, enumerate all per-channel quantized values
        (uses itertools.product over channel-values).
        2) Take Cartesian product across slots to get ALL possible key combinations.
        3) Sample 'num_keys' combinations (without replacement).
        4) Convert each chosen combo to a Fourier watermark pattern using make_Fourier_ringid_pattern(...) and apply robustness fixes.
        """

        base_latents = self.pipeline.get_random_latents().to(self.device).to(torch.float64)
        size = base_latents.shape[-1]


        # build single-channel masks and move to device with explicit dtype
        single_ring_mask = torch.tensor(
            ring_mask(size=size, r_out=self.RADIUS, r_in=self.RADIUS_CUTOFF),
            dtype=torch.float32, device=self.device
        )
        single_heter_mask = torch.tensor(
            ring_mask(size=size, r_out=self.RADIUS, r_in=self.RADIUS_CUTOFF),
            dtype=torch.float32, device=self.device
        )


        # heter mask for heterogeneous watermark channels
        heter_watermark_region_mask = None
        if len(self.HETER_WATERMARK_CHANNEL) > 0:
            heter_watermark_region_mask = single_heter_mask.unsqueeze(0).repeat(
                len(self.HETER_WATERMARK_CHANNEL), 1, 1
            ).to(self.device)


        # full watermark_region_mask across channels
        watermark_region_mask = torch.stack([
            single_ring_mask if ch in self.RING_WATERMARK_CHANNEL else single_heter_mask
            for ch in self.WATERMARK_CHANNEL
        ], dim=0).to(self.device).bool()


        # prepare quantized per-channel values and slot count
        values = np.linspace(-ring_value_range, ring_value_range, quantization_levels).tolist()
        single_channel_num_slots = self.RADIUS - self.RADIUS_CUTOFF  # number of ring slots
        num_ring_channels = len(self.RING_WATERMARK_CHANNEL)


        # per-slot options: each slot option enumerates tuples of length num_ring_channels
        per_slot_channel_options = [
            [list(c) for c in itertools.product(values, repeat=num_ring_channels)]
            for _ in range(single_channel_num_slots)
        ]


        # Cartesian product across slots -> all combos
        # each combo is a tuple of length single_channel_num_slots,
        # where each element is a list/tuple of length num_ring_channels
        all_combos = list(itertools.product(*per_slot_channel_options))
        total_combinations = len(all_combos)


        if total_combinations == 0 or num_keys <= 0:
            logger.error("No valid key combinations (total_combinations=0) or num_keys <= 0")
            return [], watermark_region_mask


        # Safety: avoid high memory usage in experiments
        MAX_ENUMERATE = 200000 
        if total_combinations > MAX_ENUMERATE:
            logger.warning(
                f"Total combinations ({total_combinations}) > {MAX_ENUMERATE}. "
                "This may exhaust memory. Consider lowering quantization_levels or using "
                "a streaming sampler."
            )

        if num_keys > total_combinations:
            logger.warning(f"Requested num_keys ({num_keys}) > total combos ({total_combinations}). Using all combos.")
            num_keys = total_combinations


        seed_env = os.environ.get("RINGID_SEED")
        if seed_env is not None:
            np.random.seed(int(seed_env))


        sampled_indices = np.random.choice(total_combinations, size=num_keys, replace=False)
        selected_combos = [all_combos[int(i)] for i in sampled_indices]


        # build Fourier patterns from selected combinations
        fourier_patterns = []
        for combo in selected_combos:
            # combo is a tuple length slot_count; each element is a list of per-channel values
            pat = make_Fourier_ringid_pattern(
                self.device, list(combo), base_latents,
                radius=self.RADIUS, radius_cutoff=self.RADIUS_CUTOFF,
                ring_watermark_channel=self.RING_WATERMARK_CHANNEL,
                heter_watermark_channel=self.HETER_WATERMARK_CHANNEL,
                heter_watermark_region_mask=heter_watermark_region_mask
            )
            if isinstance(pat, torch.Tensor):
                pat = pat.to(self.device)
            fourier_patterns.append(pat)


        # robustness transforms (IFFT -> real -> FFT) and shift for ring channels
        if self.fix_gt:
            # Lossless imprinting: only keep real part
            fourier_patterns = [fft(ifft(pattern).real) for pattern in fourier_patterns]

        if self.time_shift:
            # Spatial shift for rotation robustness
            for pattern in fourier_patterns:
                pattern[:, self.RING_WATERMARK_CHANNEL, ...] = fft(
                    torch.fft.fftshift(ifft(pattern[:, self.RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2))
                )

        return fourier_patterns, watermark_region_mask


    def watermark_image(self, text_prompt: str, num_keys: int = 2, ring_value_range: int = 64,
                       guidance_scale: float = 7.5, num_inference_steps: int = 50, 
                       output_dir: str = None, seed: int = 42):
        """
        Watermark image generation using RingID methodology
        
        Args:
            text_prompt: Text description for image generation
            num_keys: Number of watermark keys to generate
            ring_value_range: Range for discretization (±value)
            guidance_scale: Diffusion guidance scale
            num_inference_steps: Number of diffusion steps
            output_dir: Directory to save results
            seed: Random seed for reproducibility
            
        Returns:
            dict: Results containing images, patterns, and metadata
        """
        logger.info(f"Starting RingID watermarking with prompt: '{text_prompt}'")
        
        set_random_seed(seed)
        
        # Generate random initial latents (NOT from input image)
        original_latents = self.pipeline.get_random_latents()
        
        patterns, mask = self._create_watermark_patterns(
            num_keys=num_keys, 
            ring_value_range=ring_value_range,
            quantization_levels=2
        )
        
        if not patterns:
            logger.error("Failed to generate watermark patterns")
            return None
        
        wm_pattern = patterns[0]
        logger.info(f"Generated {len(patterns)} watermark patterns, using pattern 0")
        
        # Generate watermarked and non-watermarked latents
        with torch.no_grad():
            wm_latents = generate_Fourier_watermark_latents(
                device=self.device,
                radius=self.RADIUS,
                radius_cutoff=self.RADIUS_CUTOFF,
                original_latents=original_latents,
                watermark_pattern=wm_pattern,
                watermark_channel=self.WATERMARK_CHANNEL,
                watermark_region_mask=mask,
            )
            
            batched_latents = torch.cat([
                original_latents.to(torch.float16), 
                wm_latents.to(torch.float16)
            ], dim=0)
            
            images = self.pipeline(
                [text_prompt] * 2,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=512,
                width=512,
                latents=batched_latents,
            ).images
        
        no_wm_img, wm_img = images
        
        # Calculate CLIP similarity if reference model available
        no_wm_clip = wm_clip = torch.tensor(0.0)
        if self.ref_model:
            no_wm_clip, wm_clip = measure_similarity(
                [no_wm_img, wm_img], text_prompt,
                self.ref_model, self.ref_clip_preprocess, self.ref_tokenizer, self.device
            )
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            no_wm_img.save(os.path.join(output_dir, "no_watermark.png"))
            wm_img.save(os.path.join(output_dir, "watermarked.png"))
            logger.info(f"Images saved to {output_dir}")
        
        results = {
            "no_watermark_image": no_wm_img,
            "watermarked_image": wm_img,
            "watermark_patterns": patterns,
            "watermark_mask": mask,
            "used_pattern_index": 0,
            "metadata": {
                "text_prompt": text_prompt,
                "num_keys_generated": len(patterns),
                "ring_value_range": ring_value_range,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "clip_similarity_no_watermark": float(no_wm_clip.item()),
                "clip_similarity_watermark": float(wm_clip.item()),
                "parameters": {
                    "radius": self.RADIUS,
                    "radius_cutoff": self.RADIUS_CUTOFF,
                    "channels": self.WATERMARK_CHANNEL,
                    "fix_gt": self.fix_gt,
                    "time_shift": self.time_shift,
                    "channel_min": self.channel_min
                }
            }
        }
        
        logger.info(f"Watermarking complete | CLIP scores - No-WM: {no_wm_clip.item():.3f}, WM: {wm_clip.item():.3f}")
        return results


    def detect_watermark(self, image_path_or_pil, watermark_patterns: List[torch.Tensor], 
                        watermark_mask: torch.Tensor, num_inference_steps: int = 50,
                        return_all_scores: bool = False):
        """
        Detect watermark in an image using official RingID detection method
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object
            watermark_patterns: List of watermark patterns to test against
            watermark_mask: Watermark region mask
            num_inference_steps: Steps for DDIM inversion
            return_all_scores: Whether to return scores for all patterns
            
        Returns:
            dict: Detection results with scores and identified key
        """
        if isinstance(image_path_or_pil, str):
            logger.info(f"Detecting watermark in image: {image_path_or_pil}")
            img = Image.open(image_path_or_pil).convert("RGB")
            image_source = image_path_or_pil
        else:
            logger.info("Detecting watermark in provided PIL image")
            img = image_path_or_pil
            image_source = "PIL_Image"
        
        img_tensor = transform_img(img).unsqueeze(0).to(self.device)
        img_tensor = img_tensor.to(self.pipeline.vae.dtype)
        
        # Use empty prompt for detection (official RingID method)
        tester_prompt = ""
        detection_text_embeddings = self.pipeline.get_text_embedding(tester_prompt)
        
        with torch.no_grad():
            recovered_latents = self.pipeline.get_image_latents(img_tensor, sample=False)
            
            reconstructed_latents = self.pipeline.forward_diffusion(
                latents=recovered_latents,
                text_embeddings=detection_text_embeddings,  
                guidance_scale=1, 
                num_inference_steps=num_inference_steps,
            )
            
            reconstructed_fft = fft(reconstructed_latents)
            
            distances = []
            detection_scores = []
            
            for i, pattern in enumerate(watermark_patterns):
                distance = get_distance(
                    pattern, reconstructed_fft, watermark_mask,
                    channel=self.WATERMARK_CHANNEL, 
                    p=1, 
                    mode="complex", 
                    channel_min=self.channel_min  # Multi-channel heterogeneous detection
                )
                
                try:
                    dist_value = float(distance.item()) if hasattr(distance, 'item') else float(distance)
                except:
                    dist_value = float(distance)
                
                distances.append(dist_value)
                detection_scores.append(-dist_value)  # Negative distance for score
                
                logger.debug(f"Pattern {i}: distance = {dist_value:.6f}, score = {-dist_value:.6f}")
        
        # Find best matching pattern (minimum distance = maximum score)
        best_match_idx = int(np.argmin(distances))
        best_distance = distances[best_match_idx]
        best_score = detection_scores[best_match_idx]
        
        # Calculate confidence and watermark detection
        sorted_distances = np.sort(distances)
        if len(sorted_distances) > 1:
            separation = sorted_distances[1] - sorted_distances[0]  # Gap between best and second-best
            # Normalize separation to 0-1 confidence
            confidence = min(1.0, separation / 10.0)
        else:
            confidence = 1.0
        
        # Calculate confidence (normalize score to 0-1 range)
        #confidence = min(0.0, min(1.0, (best_score + 20.0) / 20.0))

        detection_threshold = -40.0
        is_watermarked = best_score > detection_threshold
        
        results = {
            "image_source": image_source,
            "is_watermarked": is_watermarked,
            "confidence": confidence,
            "detected_key_index": best_match_idx,
            "best_detection_score": best_score,
            "best_distance": best_distance,
            "detection_threshold": detection_threshold,
            "method": "RingID_official",
            "detection_parameters": {
                "empty_prompt_used": True,
                "guidance_scale": 1,
                "num_inference_steps": num_inference_steps,
                "channel_min": self.channel_min
            }
        }
        
        if return_all_scores:
            results["all_distances"] = distances
            results["all_detection_scores"] = detection_scores
            results["pattern_rankings"] = np.argsort(distances).tolist()  # Best to worst
        
        logger.info(f"Detection complete | Watermarked: {is_watermarked} | Key: {best_match_idx} | Score: {best_score:.3f} | Confidence: {confidence:.3f}")
        
        return results



def process_sample(sample_dir: str, num_keys: int = 2, ring_value_range: int = 64,
                   guidance_scale: float = 7.5, num_inference_steps: int = 50, seed: int = 42):
    ringid = RingIDWrapper(model_name="ringid_model1", online=True)

    captions_path = os.path.join(sample_dir, "original", "captions.json")
    with open(captions_path, "r") as f:
        captions = json.load(f) 
    text_prompt = captions.get("original", "") 

    out_dir = os.path.join(sample_dir, "watermarking", ringid.model_name)
    os.makedirs(out_dir, exist_ok=True)
    wm_results = ringid.watermark_image(
        text_prompt=text_prompt,
        num_keys=num_keys,
        ring_value_range=ring_value_range,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_dir=out_dir,
        seed=seed
    )

    if wm_results is None:
        raise RuntimeError("Watermarking failed: no patterns produced")

    watermarked_img_path = os.path.join(out_dir, "watermarked.png")
    no_wm_img_path = os.path.join(out_dir, "no_watermark.png")

    det_wm = ringid.detect_watermark(
        watermarked_img_path,
        watermark_patterns=wm_results["watermark_patterns"],
        watermark_mask=wm_results["watermark_mask"],
        num_inference_steps=num_inference_steps,
        return_all_scores=False
    )

    det_no = ringid.detect_watermark(
        no_wm_img_path,
        watermark_patterns=wm_results["watermark_patterns"],
        watermark_mask=wm_results["watermark_mask"],
        num_inference_steps=num_inference_steps,
        return_all_scores=False
    )

    report = {
        "text_prompt": text_prompt,
        "num_keys": num_keys,
        "ring_value_range": ring_value_range,
        "guidance_scale_gen": guidance_scale,
        "guidance_scale_detect": 1,
        "detection": {
            "watermarked": {
                "score": det_wm["best_detection_score"],
                "distance": det_wm["best_distance"],
                "key_index": det_wm["detected_key_index"],
                "is_watermarked": det_wm["is_watermarked"],
                "confidence": det_wm["confidence"]
            },
            "no_watermark": {
                "score": det_no["best_detection_score"],
                "distance": det_no["best_distance"],
                "key_index": det_no["detected_key_index"],
                "is_watermarked": det_no["is_watermarked"],
                "confidence": det_no["confidence"]
            },
            "score_difference": det_wm["best_detection_score"] - det_no["best_detection_score"]
        }
    }
    with open(os.path.join(out_dir, "detection_report.json"), "w") as f:
        json.dump(report, f, indent=2) 

    return {
        "output_dir": out_dir,
        "generation": wm_results,
        "detection": {
            "watermarked": det_wm,
            "no_watermark": det_no,
            "score_difference": report["detection"]["score_difference"]
        }
    }

    

if __name__ == "__main__":
    cfg = Config()

    sample_dir = "./experiments/image_000000000003"

    results = process_sample(
        sample_dir=sample_dir,
        num_keys=4,                 # number of keys to enumerate for pattern generation
        ring_value_range=64,        
        guidance_scale=7.5,         
        num_inference_steps=50,    
        seed=42
    )

    print("Output dir:", results["output_dir"])
    print("Watermarked image saved at:",
          os.path.join(results["output_dir"], "watermarked.png"))
    print("No-watermark image saved at:",
          os.path.join(results["output_dir"], "no_watermark.png"))

    print("Watermarked score:",
          f"{results['detection']['watermarked']['best_detection_score']:.6f}")
    print("No-watermark score:",
          f"{results['detection']['no_watermark']['best_detection_score']:.6f}")
    print("Score difference:",
          f"{results['detection']['score_difference']:.6f}")