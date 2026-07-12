# LLM Judge Consistency Under Perturbation

## Overview

This repository contains code and results for a study testing whether LLM judges (Llama 3.1 8B, Mistral 7B) are consistent when scoring perturbed model outputs. Four perturbation types are applied to 100 AlpacaEval instances: typos, filler phrases, bullet formatting, and sentence shuffling.

## Key Findings

- **Typos cause significant score drops:** Llama 3.1 drops to 47% exact match (p < 0.0001), Mistral to 74% (p = 0.003)
- **Sentence shuffling is most destructive:** 21% category flips in Llama 3.1 (p < 0.0001)
- **Formatting changes (filler, bullets) are benign:** >80% exact match, p > 0.05

## Repository Structure

```
├── src/
│   ├── experiment.py          # Main experiment code
│   └── visualize.py           # Figure generation
├── data/
│   └── alpaca_eval.json       # Dataset (downloaded automatically)
├── results/
│   ├── judge_consistency_100.csv   # Raw scores
│   └── consistency_summary.csv     # Computed metrics
├── figures/
│   ├── figure1_exact_match.png
│   ├── figure_llama_typos_heatmap.png
│   ├── figure_llama_shuffled_heatmap.png
│   └── figure4_category_flips.png
└── paper/
    └── main.tex
```

## Requirements

- Python 3.10+
- PyTorch 2.3+
- Transformers 4.43+
- CUDA-capable GPU (A100 recommended, 40GB VRAM)

Install dependencies:

```bash
pip install transformers accelerate bitsandbytes pandas numpy matplotlib seaborn scipy
```
