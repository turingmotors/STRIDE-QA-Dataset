# STRIDE-QA Bench

STRIDE-QA Bench is a unified benchmarking suite for evaluating vision-language models (VLMs) on spatial-temporal reasoning tasks.
The provided scripts run inference on the dataset, compute evaluation metrics, and generate reports.
Results are saved as JSON/Markdown files and visualized as plots.

## 📦 Getting Started

### Prerequisites

- Python 3.10 is recommended (for one-shot install of detectron2 with `uv sync`).
  If you install detectron2 manually, Python 3.11+ could also be supported.
- CUDA 12.1+ is recommended. Ensure your `CUDA_HOME` is set.

### Install Dependencies

Install dependencies with `uv`:

```shell
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and install dependencies
git clone https://github.com/turingmotors/STRIDE-QA-Dataset.git
cd STRIDE-QA-Dataset/benchmarks/STRIDE-QA-Bench

uv sync --all-extras
```

### Download Dataset

Download the processed STRIDE-QA Bench dataset from [turing-motors/STRIDE-QA-Bench](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench) and put them in the `data/STRIDE-QA-Bench` folder, using `huggingface-cli`:

```shell
mkdir -p data/STRIDE-QA-Bench
uv run huggingface-cli login # if not logged in
uv run huggingface-cli download turing-motors/STRIDE-QA-Bench --repo-type dataset --local-dir ./data/STRIDE-QA-Bench --local-dir-use-symlinks False
```

The folder should look like the following:

```shell
$ tree -L 2 --dirsfirst -F
├── config/
│   ├── prompt.yaml
│   └── tolerance.yaml
├── data/
│   └── STRIDE-QA-Bench/
├── results/
├── scripts/
│   ├── common/
│   ├── inference_internvl2_5.sh*
│   ├── inference_qwen2_5_vl.sh*
│   └── strideqa_bench.sh*
├── src/
│   └── strideqa_bench/
├── pyproject.toml
├── README.md
└── uv.lock
```

## 📊 Evaluate Baseline Models

We provide scripts to run inference and evaluation for the following baseline models:

- Qwen2.5-VL-7B-Instruct: [inference script](./scripts/inference_qwen2_5_vl.sh)
- InternVL2_5-8B: [inference script](./scripts/inference_internvl2_5.sh)

Example:

```shell
# Run inference for Qwen2.5-VL-7B-Instruct, evaluate the responses, and visualize the results
# It may take around 4 hours to run on a single H100 GPU. Distributed inference is also supported by default.
./scripts/inference_qwen2_5_vl.sh
```

- Check the generated response JSON files and the JSON/Markdown reports saved under `./results/<MODEL>-<TIMESTAMP>`. You can override the output directory with `--output-dir`.
- Set `MODEL_PATH` in the script file, or pass it as an environment variable when invoking the script:

    ```shell
    MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct ./scripts/inference_qwen2_5_vl.sh
    ```

- Note that results may vary slightly across runs due to nondeterminism, even with a fixed seed.

## 🧪 Evaluate Your Models

Before running the evaluation, you must first run inference to produce model responses as JSON files. You can use the provided scripts in `scripts/` or your own inference pipeline, as long as the outputs match the expected format. Once predictions are ready, run the evaluator to generate metrics and reports:

```shell
uv run python -m strideqa_bench.benchmark \
  --input-dir /path/to/predictions \
  --output-dir /path/to/predictions \
  --annotation-dir ./data/STRIDE-QA-Bench/annotation_files \
  --config-path ./config/tolerance.yaml

# Example: evaluate Qwen2.5-VL-7B-Instruct responses
uv run python -m strideqa_bench.benchmark --input-dir ./results/strideqa_bench/Qwen2.5-VL-7B-Instruct-20250819_153601 --output-dir ./results/strideqa_bench/Qwen2.5-VL-7B-Instruct-20250819_153601 --annotation-dir ./data/STRIDE-QA-Bench/annotation_files  --config-path ./config/tolerance.yaml
```

Results include `metrics_report.json|md`, `raw_evaluation.json`, and visualization plots under the output directory.


## 💡 Write Your Own Inference Script (per model)

You should prepare an inference script per model under:
`src/strideqa_bench/inference/models/` (see existing ones in this folder).

Reference implementations:

- `src/strideqa_bench/inference/models/inference_qwen2_5_vl.py`
- `src/strideqa_bench/inference/models/inference_internvl2_5.py`

What to implement:

1) A model adapter class that conforms to `VlmInterface`

- Required methods:
  - `load(model_path, **kwargs)`: load processor/tokenizer/model to `self.device`
  - `preprocess(inputs) -> dict`: build a model-ready batch from `{images, sample, system_prompt}`
  - `generate(batch, generation_config) -> str`: run text generation and return a plain string

2) A CLI entry that reuses the common runner

- Create a dataclass extending `InferenceArguments`
- Parse with `HfArgumentParser`
- Build `init_kwargs` (paths, sizes, flags) and `generation_config`
- Call `run_inference(...)`

Minimal template example:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import HfArgumentParser

from strideqa_bench.inference.inference import run_inference
from strideqa_bench.inference.inference_args import InferenceArguments
from strideqa_bench.inference.vlm_interface import VlmInterface


@dataclass
class MyArgs(InferenceArguments):
    model_base: Path = field(default=Path("/path/to/model"))


class MyVLM(VlmInterface):
    def __init__(self, model_path: Path | str, device: torch.device, **kwargs: Any) -> None:
        self.device = device
        self.load(model_path, **kwargs)

    def load(self, model_path: Path | str, **kwargs: Any) -> None:
        # Load processor/model to self.device
        ...

    def preprocess(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # inputs: {"sample": {...}, "system_prompt": str, "images": list[PIL.Image]}
        # return: model-ready batch (tensors on self.device)
        ...

    def generate(self, inputs: dict[str, Any], generation_config: dict[str, Any]) -> str:
        # return a plain text response
        ...


def main() -> None:
    parser = HfArgumentParser(MyArgs)
    args = parser.parse_args_into_dataclasses()[0]

    init_kwargs = {"model_path": args.model_base}
    generation_config = MyVLM.load_generation_config(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
    )
    run_inference(args=args, interface=MyVLM, init_kwargs=init_kwargs, generation_config=generation_config)


if __name__ == "__main__":
    main()
```

How to run your script (example):

```shell
uv run torchrun --nproc-per-node <NUM_GPUS> --master_port 29500 \
  ./src/strideqa_bench/inference/models/inference_my_model.py \
  --annotation-dir ./data/STRIDE-QA-Bench/annotation_files \
  --image-folder   ./data/STRIDE-QA-Bench \
  --output-dir     ./results/strideqa_bench/MyModel-$(date +%Y%m%d_%H%M%S) \
  --model-path     /path/to/checkpoint \
  --max-new-tokens 200 --seed 42
```
