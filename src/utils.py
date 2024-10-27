"""
utils.py

This module provides utility functions for loading configuration files 
and images while ensuring proper handling of errors and image formats.
"""

import os
import configparser
from PIL import Image


def load_config(path):
    """Load configuration from a specified path."""
    config = configparser.ConfigParser(allow_no_value=True, delimiters=('=', ':'), strict=False)
    config.optionxform = str  # Maintain exact casing of options
    config.read(path, encoding='utf-8')
    return config


def load_image(path):
    """Load an image from a specified path, converting it to RGB if necessary."""
    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The image is not found at the path: {path}")

    try:
        # Load the image
        img = Image.open(path)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (IOError, Image.UnidentifiedImageError) as e:
        raise IOError(
            f"The file at {path} cannot be opened or is not a valid image. "
            f"Detail: {e}"
        ) from e  # Re-raise the error for clarity
