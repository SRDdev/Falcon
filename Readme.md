# Text-Based Image Editing with Watermark Detection

## Overview

This project focuses on editing images using text-based commands and investigating how watermarked images sustain various image editing attacks. It allows users to perform a variety of image manipulations, such as resizing, cropping, and applying filters, while also supporting watermark detection post-editing. The project explores the impact of text-based image editing on the integrity of watermarks in AI-generated images.

## Features
- Resize images
- Crop images
- Apply filters
- Adjust brightness and contrast
- Add text annotations
- Add watermarks to images
- Edit AI-generated watermarked images
- Calculate watermark detection after the image has been edited

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Editing an Image

To start editing an image, use the following command with your desired text-based command for editing:

```bash
python edit_image.py --image <path_to_image> --command "<edit_command>"
```

Example:

```bash
python edit_image.py --image sample.jpg --command "resize 800x600"
```

This will apply the specified edit to the image.

### Adding a Watermark to an Image

To add a watermark to an image, use the following command:

```bash
python add_watermark.py --image <path_to_image> --watermark "<watermark_text>" --output <output_image_path>
```

Example:

```bash
python add_watermark.py --image sample.jpg --watermark "Confidential" --output watermarked_sample.jpg
```

This command will add the watermark text `"Confidential"` to the image and save the output as `watermarked_sample.jpg`.

### Editing a Watermarked Image

To edit a watermarked image, simply provide the image path and the desired text-based editing command. The system will automatically detect and handle watermark-specific changes after editing.

Example:

```bash
python edit_image.py --image watermarked_sample.jpg --command "turn it into an anime"
```

### Watermark Detection After Editing

After editing an image, you can calculate the watermark detection using the following command:

```bash
python detect_watermark.py --image <path_to_edited_image>
```

Example:

```bash
python detect_watermark.py --image edited_sample.jpg
```

This will analyze the edited image to check for the presence of a watermark and determine how well it has sustained the editing process.

## Contributing

Contributions are welcome! If you'd like to enhance the watermark detection, improve the watermark adding functionality, or contribute to any other part of the project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, please contact [Shreyas](mailto:shreyasrd31@gmail.com).
