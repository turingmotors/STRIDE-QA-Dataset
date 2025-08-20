import json
import os
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_pretrained_qwen_vl(
    model_path: str | Path,
    *,
    model_base: str | Path | None = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "auto",
    device: str = "cuda",
    use_flash_attn: bool = False,
    **kwargs: Any,
):
    model_path = str(model_path)
    model_base = str(model_base) if model_base is not None else None
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["_attn_implementation"] = "flash_attention_2"

    if is_lora_model(model_path) and model_base is None:
        raise ValueError(
            "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
        )

    if is_lora_model(model_path):
        print("Loading Qwen2-VL from base model...")
        ModelClass = (
            Qwen2_5_VLForConditionalGeneration
            if "Qwen2.5" in model_base or "Cosmos" in model_base
            else Qwen2VLForConditionalGeneration
        )

        base_kwargs = dict(low_cpu_mem_usage=True, **kwargs)
        model = ModelClass.from_pretrained(model_base, **base_kwargs)

        processor = AutoProcessor.from_pretrained(model_base, use_fast=True)

        # Resize lm_head and embed_tokens if they are not the same size as the base model
        token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(
                    token_num, token_dim, device=model.device, dtype=model.dtype
                )
            )
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(
                    token_num, token_dim, device=model.device, dtype=model.dtype
                )
            )

        # Non-LoRA additional weights (Vision, etc.)
        print("Loading additional Qwen2-VL weights...")
        non_lora_trainables = torch.load(
            os.path.join(model_path, "non_lora_state_dict.bin"), map_location="cpu"
        )
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        # Merge LoRA weights
        print("Loading LoRA weights and merging...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    else:
        print(
            f"Loading model from {model_path} as a standard model. Adapter files were not found, so it can't be merged"
        )
        config_path = Path(model_path) / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        if "Qwen2_5" in config["architectures"][0] or "Cosmos" in config["architectures"][0]:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

        else:
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    return processor, model


def is_lora_model(model_path: str | Path) -> bool:
    """
    Check if a model directory contains LoRA adapter files.

    Args:
        model_path: Path to the model directory

    Returns:
        bool: True if the directory contains LoRA adapter files
    """
    model_dir = Path(model_path)
    return (model_dir / "adapter_config.json").exists() and (
        model_dir / "adapter_model.safetensors"
    ).exists()


def get_model_name_from_path(model_path: str) -> str:
    """
    Get the model name from the model path.

    Args:
        model_path: Path to the model directory

    Returns:
        str: Model name
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
