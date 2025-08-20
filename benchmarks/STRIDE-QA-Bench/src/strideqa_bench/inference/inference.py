import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import (
    set_seed,
)

from strideqa_bench.inference.inference_args import InferenceArguments
from strideqa_bench.inference.utils.distributed import init_distributed, split_by_rank
from strideqa_bench.inference.utils.som import (
    draw_som_on_image,
    resize_mask,
    restore_mask_from_rle,
)
from strideqa_bench.inference.utils.video import save_video
from strideqa_bench.inference.vlm_interface import VlmInterface

load_dotenv()


def validate_question(q: dict) -> None:
    question_id = q["question_id"]
    if "rle" not in q:
        raise ValueError(f"No RLE data found for question_id {question_id}")
    rle_data = q["rle"]
    if "current" not in rle_data:
        raise ValueError(f"No 'current' RLE data found for question_id {question_id}")


def process_single_annotation_file(
    ann_path: Path,
    *,
    vlm: VlmInterface,
    system_prompt: str,
    generation_config: dict[str, Any],
    args: InferenceArguments,
    rank: int,
    world_size: int,
) -> None:
    """Run inference for a single annotation JSON and save result.

    Args:
        ann_path (Path): Path to annotation JSON.
        vlm (VlmInterface): Vision-language model interface.
        generation_configs (dict): kwargs for model.generate.
        system_prompt (str): System prompt text.
        args (InferenceArguments): Parsed CLI arguments.
        rank (int): Distributed rank.
        world_size (int): World size.
    """

    t = int(ann_path.stem[-1])
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    out_name = model_name.replace("/", "--") + f"_{date}_t{t}.json"
    output_file_path = Path(args.output_dir) / out_name

    questions = json.loads(ann_path.read_text(encoding="utf-8"))
    if isinstance(args.limit, int) and args.limit > 0:
        questions = questions[: args.limit]
    questions = split_by_rank(questions, rank, world_size)

    time_buckets = ["prev_3", "prev_2", "prev_1", "current"]
    target_size = (args.width, args.height)

    outs: list[dict[str, Any]] = []
    progress = tqdm(
        questions,
        dynamic_ncols=True,
        total=len(questions),
        disable=(rank != 0),
        desc=f"Inference t{t}...",
    )

    for q in progress:
        validate_question(q)

        region_id = q["region_id"]
        images: list[Image.Image] = []
        for idx, image_file in enumerate(q["images"]):
            image_path = Path(args.image_folder) / image_file
            image_rgb = np.array(Image.open(image_path).convert("RGB").resize(target_size))
            bucket = time_buckets[idx]
            mask = restore_mask_from_rle(q["rle"][bucket])
            mask = resize_mask(mask, target_size)
            image_som_np = draw_som_on_image(
                image_rgb, [mask], [region_id], alpha=0.2, anno_mode=["Mask", "Mark"]
            )
            images.append(Image.fromarray(image_som_np))

        if args.save_video:
            save_video(
                images,
                Path(args.output_dir)
                / "video"
                / f"t{t}"
                / f"input_context_{q['question_id']}_{region_id}.mp4",
                fps=1,
                align_macro_block_size=True,
            )

        inputs = {
            "sample": q,
            "system_prompt": system_prompt,
            "images": images,
        }
        response = vlm.infer(inputs, generation_config)

        out_annotation = q.copy()
        out_annotation.update({"pred": response, "model_id": str(args.model_path), "t": t})
        outs.append(out_annotation)

    # Gather results
    if dist.is_initialized():
        gathered_outs = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_outs, outs)
        if rank == 0:
            outs = [item for sublist in gathered_outs for item in sublist]

    if rank == 0:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(outs, f, indent=4, ensure_ascii=False)
        print(f"[t{t}] results saved to {output_file_path}")


def run_inference(
    args: InferenceArguments,
    interface: VlmInterface,
    init_kwargs: dict[str, Any],
    generation_config: dict[str, Any],
) -> None:
    # Distributed init
    device, rank, world_size = init_distributed()

    args.dump(args.output_dir)

    if args.seed is not None:
        set_seed(args.seed, deterministic=True)

    # Prompt config
    system_prompt = yaml.safe_load(Path(args.prompt_config).read_text(encoding="utf-8"))[
        "system_prompt"
    ]
    if args.system_prompt:
        system_prompt = args.system_prompt

    if args.save_video:
        (Path(args.output_dir) / "video").mkdir(parents=True, exist_ok=True)

    init_kwargs.setdefault("device", device)
    vlm = interface(**init_kwargs)

    if args.annotation_dir is not None:
        annotation_files = sorted(Path(args.annotation_dir).glob("strideqa_bench*.json"))
    else:
        annotation_files = [args.annotation_file]

    if not annotation_files:
        raise FileNotFoundError("No annotation JSON found.")

    for ann_path in annotation_files:
        process_single_annotation_file(
            ann_path,
            vlm=vlm,
            system_prompt=system_prompt,
            generation_config=generation_config,
            args=args,
            rank=rank,
            world_size=world_size,
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
