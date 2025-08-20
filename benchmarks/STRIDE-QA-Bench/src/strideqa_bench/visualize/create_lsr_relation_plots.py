"""
CLI wrapper to generate LSR relation plots from result directories.

Usage example:

```bash
python -m strideqa_bench.visualize.create_lsr_relation_plots \
  --model-dirs \
    "GPT-4o:/path/to/gpt-4o-results" \
    "Qwen2.5-VL-7B:/path/to/qwen-results" \
  --output-dir /path/to/out \
  --pdf-name lsr_relation_grid.pdf
```
"""

import argparse
from pathlib import Path

from strideqa_bench import STRIDEQA_BENCH_ROOT
from strideqa_bench.visualize.visualize import create_lsr_relation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LSR relation plots from results.")
    parser.add_argument(
        "--model-dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of model_name:dir_path pairs (e.g., 'ModelA:/path/to/dir')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(STRIDEQA_BENCH_ROOT / "results/lsr_relation_plots").resolve(),
        help="Directory to save the relation plots",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="raw_evaluation.json",
        help="Evaluation JSON file name inside each result directory",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="lsr_relation_grid.pdf",
        help="Output PDF file name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dirs: dict[str, Path] = {}
    for item in args.model_dirs:
        if ":" not in item:
            raise ValueError(
                f"Invalid format for --model-dirs entry: {item}. Expected 'name:/abs/path'"
            )
        name, dir_str = item.split(":", 1)
        dir_path = Path(dir_str)
        if not dir_path.exists():
            raise FileNotFoundError(f"Result directory not found: {dir_path}")
        model_dirs[name] = dir_path

    out_path = create_lsr_relation_plots(
        model_dirs=model_dirs,
        output_dir=args.output_dir,
        rel_thresh_minor=15,
        json_name=args.json_name,
        pdf_name=args.pdf_name,
    )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
