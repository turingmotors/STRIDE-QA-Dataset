from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, HfArgumentParser

from strideqa_bench.inference.inference import run_inference
from strideqa_bench.inference.inference_args import InferenceArguments
from strideqa_bench.inference.vlm_interface import VlmInterface

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class InferenceArgumentsForInternVL2_5(InferenceArguments):
    model_base: Path = field(
        default="/data/models/InternVL/InternVL2_5-8B",
        metadata={"help": "Base model name to use"},
    )
    input_size: int = field(
        default=448, metadata={"help": "Input size of the image to be processed"}
    )
    max_num: int = field(
        default=12, metadata={"help": "Max number of tiles for dynamic_preprocess"}
    )


def build_transform(input_size: int) -> Callable[[Image.Image], torch.Tensor]:
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_model(
    path: str,
    device: torch.device,
    use_flash_attn: bool = False,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load VLM and tokenizer."""
    model = (
        AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


class InternVL(VlmInterface):
    def __init__(
        self,
        model_path: Path | str,
        device: torch.device,
        *,
        input_size: int,
        max_num: int,
        use_flash_attn: bool,
    ) -> None:
        self.device = device
        self.load(model_path, use_flash_attn=use_flash_attn)
        self.transform = build_transform(input_size=input_size)
        self.input_size = input_size
        self.max_num = max_num

    def load(self, model_path: Path | str, use_flash_attn: bool = False) -> None:
        self.model, self.tokenizer = load_model(
            model_path, device=self.device, use_flash_attn=use_flash_attn
        )

    def preprocess(self, inputs: Any) -> dict[str, Any]:
        system_prompt = inputs["system_prompt"]
        images = inputs["images"]
        question = inputs["sample"]["question"]

        pixel_batches = []
        num_patches_list = []
        for image in images:
            tiles = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=True,
                max_num=self.max_num,
            )
            pixel_values = (
                torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).to(self.device)
            )
            pixel_batches.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])

        pixel_values = torch.cat(pixel_batches, dim=0)

        question = (
            f"{system_prompt}\n\n"
            + "\n".join(["<image>"] * len(num_patches_list))
            + f"\n{question}"
        )
        inputs.update(
            {
                "pixel_values": pixel_values,
                "num_patches_list": num_patches_list,
                "question": question,
            }
        )
        return inputs

    def generate(self, inputs: Any, generation_config: dict[str, Any]) -> str:
        pixel_values = inputs["pixel_values"]
        question = inputs["question"]
        num_patches_list = inputs["num_patches_list"]
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
        )
        return response


def main():
    parser = HfArgumentParser(InferenceArgumentsForInternVL2_5)
    args = parser.parse_args_into_dataclasses()[0]

    init_kwargs = {
        "model_path": args.model_base,
        "input_size": args.input_size,
        "max_num": args.max_num,
        "use_flash_attn": args.use_flash_attn,
    }
    generation_config = InternVL.load_generation_config(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
    )
    run_inference(
        args=args,
        interface=InternVL,
        init_kwargs=init_kwargs,
        generation_config=generation_config,
    )


if __name__ == "__main__":
    main()
