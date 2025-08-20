from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def save_video(
    frames: list[np.ndarray | Image.Image],
    filename: str | Path,
    fps: int = 1,
    align_macro_block_size: bool = True,
) -> None:
    """Save video as gif or mp4 as filename suggests

    Args:
        frames (array-like): frames to save
        filename (str): filename of a video clip. Allowed extentions are: `mp4` or `gif`
        fps (int, optional): frame rate. Defaults to 60.
        align_macro_block_size (bool, optional): Whether to align the macro block size of the frames. Defaults to True.
    Note:
        fps can be 60 at the most when saving as .gif otherwise weirdly slow gif.
        However, mp4 accepts wider range. Also, gif is typically 100x larger in
        size than mp4.

    Example:
    >>> noisy_frames = np.random.randint(0, 255, (32, 400, 608, 3), dtype=np.uint8)
    >>> save_video(noisy_frames, './logs/demo.gif')

    """

    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)

    if align_macro_block_size:
        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame) for frame in frames]
        h, w = frames[0].size[::-1]
        nh, nw = (h + 15) // 16 * 16, (w + 15) // 16 * 16
        frames = [frame.resize((nw, nh)) for frame in frames]

    frames = np.uint8(frames)

    if filename.suffix == ".gif":
        kwargs = {"duration": 1 / fps}
        imageio.mimsave(filename, frames, **kwargs)
    elif filename.suffix == ".mp4":
        kwargs = {"fps": fps}
        imageio.mimsave(filename, frames, **kwargs)
    else:
        raise ValueError(f"Not supported file type: {filename}")
