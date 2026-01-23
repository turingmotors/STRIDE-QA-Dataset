# <img src="assets/favicon.png" alt="logo" width="36"/>STRIDE-QA: Ego-centric Spatiotemporal Reasoning in Urban Driving Scenes

<div align="center">

[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Oral-red)](https://arxiv.org/abs/2508.10427)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://turingmotors.github.io/stride-qa/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/turingmotors/STRIDE-QA-Dataset)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Dataset)
[![Benchmark](https://img.shields.io/badge/🤗%20HuggingFace-Benchmark-yellow)](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench)

</div>

We propose STRIDE-QA, a large-scale VQA dataset for ego-centric spatiotemporal reasoning in urban driving scenes, accompanied by a benchmark suite for evaluation.

<img src="https://turingmotors.github.io/stride-qa/static/images/pipeline.png" alt="teaser" width=100% height=100%>

## News

- `[2026-01-23]` STRIDE-QA Dataset is now released on Hugging Face (includes front camera images and JSONL annotations): [turing-motors/STRIDE-QA-Dataset](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Dataset)
- `[2025-12-22]` Full paper has been accepted to the **AAAI 2026** Main Track as an **Oral** presentation (Paper ID: 23443). (Oral: [Jan 24 3:00-4:00pm](https://aaai.org/wp-content/uploads/2025/12/Main-track-oral-talks.pdf), Poster: [Jan 24](https://aaai.org/wp-content/uploads/2025/12/Main-track-poster-presentations.pdf))
- `[2025-08-22]` Code and dataset for the **STRIDE-QA Bench** is released. Please refer to the [README](benchmarks/STRIDE-QA-Bench/README.md) for more details.
- `[2025-08-19]` [arXiv](https://arxiv.org/abs/2508.10427) paper released. Dataset/Benchmark/Models are coming soon. Please stay tuned! :coffee:
- `[2025-07-13]` Short paper has been accepted to the [ICCV 2025 End-to-End 3D Learning Workshop](https://e2e3d.github.io/iccv2025/).

## STRIDE-QA Bench

We provide STRIDE-QA Bench as an official framework for evaluating the spatiotemporal reasoning abilities of VLMs in urban driving contexts. The toolkit includes inference runners, evaluation scripts, and visualization utilities.

- Toolkit: [benchmarks/STRIDE-QA-Bench](benchmarks/STRIDE-QA-Bench)
- Dataset: [turing-motors/STRIDE-QA-Bench](https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench)

See the [README.md](benchmarks/STRIDE-QA-Bench/README.md) for installation, usage, and examples.

## Related Links

- Project Page: <https://turingmotors.github.io/stride-qa>
- Paper: <https://arxiv.org/abs/2508.10427>
- STRIDE-QA Dataset: <https://huggingface.co/datasets/turing-motors/STRIDE-QA-Dataset>
- STRIDE-QA Bench: <https://huggingface.co/datasets/turing-motors/STRIDE-QA-Bench>

## License

This project is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) License.

## Acknowledgments

This project is based on results obtained from a project, JPNP20017, subsidized by the New Energy and Industrial Technology Development Organization (NEDO).

## Citation

If you find STRIDE-QA is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@misc{ishihara2025strideqa,
      title={STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes},
      author={Keishi Ishihara and Kento Sasaki and Tsubasa Takahashi and Daiki Shiono and Yu Yamaguchi},
      year={2025},
      eprint={2508.10427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10427},
}
```
