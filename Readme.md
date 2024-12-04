# Text Based Image Editing

## Overview
This project focuses on editing images using text-based commands. It allows users to perform various image manipulations through a simple and intuitive text interface. Additionally, it supports editing AI-generated watermarked images and calculating watermark detection after the image has been edited.

## Features
- Resize images
- Crop images
- Apply filters
- Adjust brightness and contrast
- Add text annotations
- Edit AI-generated watermarked images
- Calculate watermark detection post-editing

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To start editing an image, use the following command:
```bash
python edit_image.py --image <path_to_image> --command "<edit_command>"
```
Example:
```bash
python edit_image.py --image sample.jpg --command "resize 800x600"
```

To calculate watermark detection after editing, use the following command:
```bash
python detect_watermark.py --image <path_to_edited_image>
```
Example:
```bash
python detect_watermark.py --image edited_sample.jpg
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any questions or feedback, please contact [Shreyas](mailto:shreyasrd31@gmail.com).
