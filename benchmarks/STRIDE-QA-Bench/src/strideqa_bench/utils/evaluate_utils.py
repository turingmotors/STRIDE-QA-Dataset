import base64
import json
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image


def encode_image_from_path(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_from_pil(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL image to a Base64 string.

    Args:
        image (Image.Image): PIL image object.
        format (str): Image format (e.g., "PNG", "JPEG").

    Returns:
        str: Base64-encoded image string.
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image(image: Image.Image | Path | str) -> str:
    if isinstance(image, Image.Image):
        return encode_image_from_pil(image)
    elif isinstance(image, Path | str):
        return encode_image_from_path(image)
    else:
        raise ValueError(f"Invalid image type: {type(image)}")


def is_url(path_str: str) -> bool:
    """
    Checks if the given string is a URL.

    Args:
        path_str (str): The input string to check.

    Returns:
        bool: True if the string is determined to be a URL, False otherwise.
    """
    parsed = urlparse(path_str)
    # If both scheme and netloc exist, treat it as a URL
    return bool(parsed.scheme and parsed.netloc)


def save_as_jsonl(data: list[dict], path: str, lines: bool = True, **kwargs) -> None:
    """
    Save a list of dictionaries to a JSONL file. By default, ensure_ascii=False is used
    to preserve non-ASCII characters such as Japanese.

    Args:
        data (list[dict]): The list of dictionaries to save.
        path (str): The path to the output JSONL file.
        **kwargs: Additional keyword arguments for json.dumps().
                  'ensure_ascii' is set to False by default.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Set ensure_ascii=False by default if not provided
    kwargs.setdefault("ensure_ascii", False)

    if lines:
        # Write each dict as a separate line
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, **kwargs) + "\n")
    else:
        kwargs.setdefault("indent", 4)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, **kwargs))


def load_jsonl(data_path: str) -> list[dict]:
    """
    Loads data from a file where each line is a JSON object.

    Returns:
        list[dict]: A list of parsed JSON objects.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"File not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def resize_images(
    images: Image.Image | list[Image.Image], max_size: int
) -> Image.Image | list[Image.Image]:
    """Resize one or multiple images while maintaining their aspect ratio,
    ensuring the longest side is max_size.

    Args:
        images (Union[Image.Image, list[Image.Image]]): Single image or list of images.
        max_size (int): The maximum size of the longest dimension.

    Returns:
        Union[Image.Image, list[Image.Image]]: Resized image(s).
    """

    def resize(image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = max_size / max(width, height)  # Compute scaling factor

        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    if isinstance(images, list):
        return [resize(image) for image in images]

    return resize(images)


def resize_masks(
    masks: np.ndarray | list[np.ndarray], max_size: int
) -> np.ndarray | list[np.ndarray]:
    """Resize one or multiple binary masks so that the longer side is set to max_size,
    preserving aspect ratio. Uses nearest-neighbor interpolation.

    Args:
        masks (Union[np.ndarray, list[np.ndarray]]): A single mask (H, W) or
            a list of masks (each with shape (H, W)).
        max_size (int): The target size for the longer side.

    Returns:
        Union[np.ndarray, list[np.ndarray]]:
            - If input was a single mask, returns the resized mask (np.ndarray).
            - If input was a list of masks, returns a list of resized masks (list[np.ndarray]).
    """

    def _resize_single_mask(mask: np.ndarray, max_side: int) -> np.ndarray:
        # Determine original dimensions
        h, w = mask.shape[:2]
        # Determine scale factor
        long_side = max(h, w)
        scale = max_side / float(long_side)
        # Compute new dimensions
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        # Resize using nearest neighbor
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized

    if isinstance(masks, np.ndarray):
        # Single mask case
        return _resize_single_mask(masks, max_size)
    else:
        # Multiple masks case
        resized_list = []
        for m in masks:
            resized_list.append(_resize_single_mask(m, max_size))
        return resized_list
