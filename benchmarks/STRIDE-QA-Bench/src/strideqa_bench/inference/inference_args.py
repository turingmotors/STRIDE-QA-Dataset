import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from strideqa_bench import STRIDEQA_BENCH_ROOT


@dataclass
class InferenceArguments:
    """
    Inference hyper-parameters and I/O settings.

    You can inherit from this class and override or add new fields to customize the inference.
    """

    # Required fields
    image_folder: Path = field(metadata={"help": "Path to the image folder"})

    # Default fields
    annotation_file: Path | None = field(
        default=None, metadata={"help": "Path to a single annotation JSON file"}
    )
    annotation_dir: Path | None = field(
        default=None,
        metadata={
            "help": "Directory containing multiple annotation JSON files. If provided, annotation-file is ignored."
        },
    )
    prompt_config: Path = field(
        default=STRIDEQA_BENCH_ROOT / "config" / "prompt.yaml",
        metadata={"help": "Path to the prompt config file"},
    )
    output_dir: Path = field(default="results", metadata={"help": "Path to the output directory"})
    model_path: Path | None = field(default=None, metadata={"help": "Path to the model checkpoint"})
    use_flash_attn: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "System prompt to override the one in the prompt config file"},
    )
    seed: int | None = field(default=42, metadata={"help": "Seed for reproducibility"})
    do_sample: bool = field(default=False, metadata={"help": "Whether to sample"})
    temperature: float | None = field(default=0.0, metadata={"help": "Sampling temperature"})
    top_p: float | None = field(default=1.0, metadata={"help": "Top-p (nucleus) sampling value"})
    max_new_tokens: int | None = field(
        default=200, metadata={"help": "Maximum number of new tokens to generate"}
    )
    limit: int | None = field(
        default=None,
        metadata={"help": "Limit the number of annotations to process (useful for testing)"},
    )
    height: int | None = field(
        default=336, metadata={"help": "Height of the image to be processed"}
    )
    width: int | None = field(default=532, metadata={"help": "Width of the image to be processed"})
    save_video: bool = field(default=False, metadata={"help": "Whether to save the video"})

    def __post_init__(self) -> None:
        if (self.annotation_file is None) == (self.annotation_dir is None):
            raise ValueError("Specify exactly one of --annotation-file or --annotation-dir")

    @staticmethod
    def _is_main_process() -> bool:
        return int(os.getenv("RANK", "0")) == 0

    def dump(self, output_dir: Path) -> None:
        if not self._is_main_process():
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "inference_args.json", "w", encoding="utf-8") as f:
            data = asdict(self)
            for k, v in data.items():
                if isinstance(v, Path):
                    data[k] = str(v)
            json.dump(data, f, indent=4)
