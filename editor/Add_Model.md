# How to Add New Models to `editor.py`

This README provides a detailed guide on how to extend the `editor.py` by adding new models for text-based image editing. Follow the steps below to add a new model for processing images based on specific instructions.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Steps to Add a New Model](#steps-to-add-a-new-model)
   1. [Step 1: Add Model Configuration in the YAML File](#step-1-add-model-configuration-in-the-yaml-file)
   2. [Step 2: Update the `TextBasedImageEditor` Class](#step-2-update-the-textbasedimageeditor-class)
   3. [Step 3: Add New Model to the `main` Function](#step-3-add-new-model-to-the-main-function)
4. [Testing the New Model](#testing-the-new-model)
5. [Conclusion](#conclusion)

---

## Overview

The purpose of the `editor.py` is to allow users to generate edited images based on text instructions. The program uses different models for image editing, such as `pix_to_pix`. To add new models, you must modify both the model configuration in the YAML file and the code within the `TextBasedImageEditor` class. 

This README will walk you through how to add new models for text-based image editing.

---

## Prerequisites

Before adding new models to the `editor.py`:

1. **Basic Knowledge of Python**: Ensure you understand basic Python programming, especially how to use classes and functions.
2. **Install Required Libraries**:
   - `torch`
   - `diffusers`
   - `PIL` (Pillow)
   
   Install them via pip if you haven't already:

   ```bash
   pip install torch diffusers pillow
   ```

3. **Existing Configuration File**: Make sure you have a configuration YAML file that stores model-specific configurations. This YAML file is crucial for defining model parameters, such as `model_id`, `steps`, `seed`, etc.

---

## Steps to Add a New Model

### Step 1: Add Model Configuration in the YAML File

1. **Open your `config.yaml` file**.
2. **Add the model configuration**: Under the root section of the YAML file, add a new model configuration, similar to the existing one for `pix_to_pix`. Here’s an example for adding a new model, say `new_model`:

   ```yaml
   device: "cuda:0"

   pix_to_pix:
     model_id: "timbrooks/instruct-pix2pix"
     steps: 50
     randomize_seed: False
     seed: 42
     randomize_cfg: False
     text_cfg_scale: 7.5
     image_cfg_scale: 1.5
     torch_dtype: "torch.float16"
     safety_checker: None

   new_model:
     model_id: "your/model-id-here"
     steps: 50
     randomize_seed: True
     seed: 12345
     randomize_cfg: True
     text_cfg_scale: 8.0
     image_cfg_scale: 2.0
     torch_dtype: "torch.float32"
     safety_checker: "stability_checker_function"
   ```

   - Replace `"your/model-id-here"` with the actual model ID from the model repository or location.
   - Adjust parameters like `steps`, `seed`, and other relevant configurations to match the desired behavior of the new model.

3. **Save the YAML file**.

### Step 2: Update the `TextBasedImageEditor` Class

Now, we need to ensure that the `TextBasedImageEditor` class can handle the new model. Here's how to do it:

1. **Open the `editor/image_editor.py` file**.
2. **Add the new model to the `supported_models` dictionary**: The `TextBasedImageEditor` class contains a dictionary that maps model names to their corresponding pipeline classes. You will need to add your new model here.

   Example update for the `supported_models` dictionary:
   
   ```python
   self.supported_models = {
       'pix_to_pix': StableDiffusionInstructPix2PixPipeline,
       'new_model': YourNewModelPipeline,  # Add your new model pipeline here
       # Add more models as needed
   }
   ```

   Replace `YourNewModelPipeline` with the appropriate class or pipeline for your model. For example, if your new model uses `StableDiffusionPipeline`, you would write:

   ```python
   'new_model': StableDiffusionPipeline,
   ```

3. **Ensure the model-specific configuration is handled correctly**: When initializing the `TextBasedImageEditor`, it uses the model configuration loaded from the YAML file. The class should already be generic enough to use any model configuration defined in the YAML file. No additional changes are required if you follow the previous step.

4. **Save the `TextBasedImageEditor` class**.

### Step 3: Add New Model to the `main` Function

Next, we need to ensure that the `main` function can handle the new model. This function reads the configuration and processes the input image accordingly.

1. **Open the `main` function in your script**.
2. **Update the `model_name` variable** to use the new model (e.g., `new_model`) when calling the `TextBasedImageEditor`.

   Example update for `main`:
   
   ```python
   def main(input_image_path, output_image_path):
       # Load configuration
       config = Config(config_path="../config/config.yaml")
       
       model_name = "new_model"  # Set to your new model's name

       # Retrieve model-specific configuration
       model_config = config.get(model_name)
       model_id = model_config['model_id']
       steps = model_config['steps']
       randomize_seed = model_config['randomize_seed']
       seed = model_config['seed']
       randomize_cfg = model_config['randomize_cfg']
       text_cfg_scale = model_config['text_cfg_scale']
       image_cfg_scale = model_config['image_cfg_scale']

       example_image = Image.open(input_image_path).convert("RGB")
       preprocessor = Preprocessor()
       editor = TextBasedImageEditor(model_name=model_name)
       processed_image = preprocessor.resize_image(example_image)
       edited_image = editor.generate_image(input_image=processed_image, instruction="Turn it into an anime.")
       edited_image.save(output_image_path)
       edited_image.show()
   ```

3. **Save your script**.

---

## Testing the New Model

After updating the configuration and script, it's time to test your new model.

1. **Run your script**:

   ```bash
   python your_script_name.py
   ```

2. **Verify**: Check the generated output image to ensure the new model is working as expected. You may need to adjust the configuration parameters (e.g., `steps`, `text_cfg_scale`, etc.) for better results.

---

## Conclusion

By following the steps outlined above, you can easily add new models to the `TextBasedImageEditor`. The key steps involve updating the model configuration in the YAML file, modifying the `TextBasedImageEditor` class to support new models, and updating the `main` function to select and use the new model.

Once you've added the model and verified its output, you can continue extending the editor by adding more models or fine-tuning existing ones.