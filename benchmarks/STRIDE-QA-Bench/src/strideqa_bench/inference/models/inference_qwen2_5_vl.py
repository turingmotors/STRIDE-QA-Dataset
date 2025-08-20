from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import HfArgumentParser

from strideqa_bench.inference.inference import run_inference
from strideqa_bench.inference.inference_args import InferenceArguments
from strideqa_bench.inference.utils.load import load_pretrained_qwen_vl
from strideqa_bench.inference.vlm_interface import VlmInterface

load_dotenv()


@dataclass
class InferenceArgumentsForQwen2_5_VL(InferenceArguments):
    model_base: Path = field(
        default="/data/models/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Base model name to use"},
    )
    height: int = field(
        default=336, metadata={"help": "Height of the image to be processed"}
    )
    width: int = field(
        default=532, metadata={"help": "Width of the image to be processed"}
    )


class QwenVL(VlmInterface):
    """Concrete adapter for Qwen-2.5-VL."""

    def __init__(
        self, model_path: Path | str, device: torch.device, **kwargs: Any
    ) -> None:
        self.device = device
        self.load(model_path, **kwargs)

    def load(self, model_path: Path | str, **kwargs: Any) -> None:
        """Load processor and model onto the given device."""
        self.processor, self.model = load_pretrained_qwen_vl(
            model_path,
            device_map=self.device.index if self.device.type == "cuda" else "cpu",
            device=self.device,
            load_4bit=False,
            load_8bit=False,
            **kwargs,
        )

    def format_inputs(
        self, sample: dict[str, Any], system_prompt: str
    ) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy_placeholder.jpg"},
                    {"type": "image", "image": "dummy_placeholder.jpg"},
                    {"type": "image", "image": "dummy_placeholder.jpg"},
                    {"type": "image", "image": "dummy_placeholder.jpg"},
                    {"type": "text", "text": sample["question"]},
                ],
            },
        ]
        return messages

    def preprocess(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self.format_inputs(inputs["sample"], inputs["system_prompt"])
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = inputs["images"]
        batch = self.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        return batch.to(self.device)

    def generate(
        self,
        inputs: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> str:
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **generation_config)
            ids = ids[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]


def main():
    parser = HfArgumentParser(InferenceArgumentsForQwen2_5_VL)
    args = parser.parse_args_into_dataclasses()[0]

    init_kwargs = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "use_flash_attn": args.use_flash_attn,
    }
    generation_config = QwenVL.load_generation_config(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
    )
    run_inference(
        args=args,
        interface=QwenVL,
        init_kwargs=init_kwargs,
        generation_config=generation_config,
    )


if __name__ == "__main__":
    main()
