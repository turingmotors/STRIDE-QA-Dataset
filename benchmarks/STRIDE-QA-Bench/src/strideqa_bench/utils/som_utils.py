import numpy as np
import pycocotools.mask as cocomask
from detectron2.data import MetadataCatalog

from strideqa_bench.utils.som import Visualizer

metadata = MetadataCatalog.get("coco_2017_train_panoptic")


def restore_mask_from_rle(rle: str) -> np.ndarray:
    mask = cocomask.decode(rle)
    return mask.astype(np.uint8)


def draw_som_on_image(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    labels: list[int],
    label_mode: str = "1",
    alpha: float = 0.1,
    anno_mode: list[str] = ["Mask"],
) -> np.ndarray:
    """
    Draw SoM on image.

    Args:
        image_rgb (np.ndarray): image to draw on
        masks (list[np.ndarray]): masks to draw
        labels (list[int]): labels to draw
        label_mode (str)
        alpha (float)
        anno_mode (list[str]): Annotation types to draw

    Returns:
        np.ndarray, image with SoM drawn on it
    """

    if len(masks) == 0:
        raise ValueError("No masks provided to draw SoM on image")

    if len(masks) != len(labels):
        raise ValueError(
            f"Number of masks and labels must be the same: masks={len(masks)}, labels={len(labels)}"
        )

    visual = Visualizer(image_rgb, metadata=metadata)
    for i in range(len(masks)):
        label = labels[i]
        mask = masks[i]
        demo = visual.draw_binary_mask_with_number(
            mask,
            text=str(label),
            label_mode=label_mode,
            alpha=alpha,
            anno_mode=anno_mode,
        )
    return demo.get_image()
