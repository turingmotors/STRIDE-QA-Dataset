"""Generic VLM interface with light default implementations."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class VlmInterface(Protocol):
    """
    Minimal interface all vision-language models should satisfy.

    You must implement the following methods:
    - load: Load the model and tokenizer/processor
    - preprocess: Preprocess the inputs
    - generate: Generate the text response

    You can optionally override the following methods:
    - infer
    - load_generation_config
    """

    device: torch.device
    processor: Any
    model: Any

    def load(self, model_path: Path | str, **kwargs: Any) -> None: ...

    def preprocess(self, inputs: Any) -> dict[str, Any]: ...

    def generate(self, inputs: Any, generation_config: dict[str, Any]) -> str: ...

    @torch.inference_mode()
    def infer(
        self,
        inputs: Any,
        generation_config: dict[str, Any],
    ) -> str:
        """End-to-end single-example inference."""
        inputs = self.preprocess(inputs)
        return self.generate(inputs, generation_config)

    @staticmethod
    def load_generation_config(
        do_sample: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        """Return a common generation-kwargs dict."""
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
        }
