# Transformers Don't Need LayerNorm at Inference Time

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/TBD)

[Paper](https://arxiv.org/abs/TBD) | [Project Page](#) | [Models](https://huggingface.co/schaeff)

This repository contains the official implementation of "Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and the Implications for Mechanistic Interpretability" which demonstrates that all layer normalization (LN) layers can be removed from GPT-2 models with only a small increase in validation loss.

## Abstract

Layer-wise normalization (LN) is an essential component of virtually all transformer-based large language models. While its effects on training stability are well documented, its role at inference time is poorly understood. Additionally, LN layers hinder mechanistic interpretability by introducing additional nonlinearities and increasing the interconnectedness of individual model components.

We show that all LN layers can be removed from every GPT-2 model with only a small increase in validation loss (e.g. +0.03 cross-entropy loss for GPT-2 XL). Thus LN cannot play a substantial role in language modeling. We find that the amount of fine-tuning data needed for LN removal grows sublinearly with model parameters, suggesting scaling to larger models is feasible.

We release a suite of LN-free GPT-2 models on Hugging Face. Our work clarifies the role of LN layers in language modeling, showing that GPT-2-class models can function without LN layers. We hope that our LN-free analogues of the GPT-2 family of models will enable more precise interpretability research and improve our understanding of language models.

## Key Findings

- **LN is not essential for inference**: GPT-2 models can function without LN layers with minimal performance degradation
- **Sublinear scaling**: The data needed for LN removal grows sublinearly with model parameters
- **Interpretability benefits**: LN-free models enable more precise mechanistic interpretability research
- **Released models**: We provide a suite of LN-free GPT-2 models for the research community

## Released Models

We release a suite of LN-free GPT-2 models on Hugging Face that can be used for more precise interpretability research:

- [GPT-2 Small LN-free](https://huggingface.co/schaeff/gpt2-small_LNFree300)
- [GPT-2 Medium LN-free](https://huggingface.co/schaeff/gpt2-medium_LNFree500)
- [GPT-2 Large LN-free](https://huggingface.co/schaeff/gpt2-large_LNFree600)
- [GPT-2 XL LN-free](https://huggingface.co/schaeff/gpt2-xl_LNFree800)

### Usage

You can load the models with transformers:

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("schaeff/gpt2-small_ln-free300")
```

## Evaluation Results

Reported values are mean cross-entropy losses for 10.2M tokens for The Pile and The Pile filtered and 4.5M tokens for the OpenWebText (OWT) validation set.

| Model | FT steps | OWT (val) | The Pile | The Pile-filtered |
|-------|----------|-----------|----------|-------------------|
| OpenAI GPT-2 Small original | 0 | 3.1006 | 2.8450 | 2.7899 |
| schaeff GPT-2 Small vanilla | 300 | 3.0126 | 2.8511 | 2.8112 |
| schaeff GPT-2 Small LN-free | 300 | 3.0797 [+0.0671] | 2.8852 [+0.0402] | 2.8757 [+0.0858] |
| OpenAI GPT-2 Medium original | 0 | 2.8145 | 2.5163 | 2.5390 |
| schaeff GPT-2 Medium vanilla | 500 | 2.7390 | 2.5752 | 2.5724 |
| schaeff GPT-2 Medium LN-free | 500 | 2.7642 [+0.0252] | 2.6579 [+0.1416] | 2.6352 [+0.0962] |
| OpenAI GPT-2 Large original | 0 | 2.6623 | 2.5320 | 2.4347 |
| schaeff GPT-2 Large vanilla | 600 | 2.6240 | 2.6233 | 2.5074 |
| schaeff GPT-2 Large LN-free | 600 | 2.6384 [+0.0144] | 2.7504 [+0.2184] | 2.5159 [+0.0812] |
| OpenAI GPT-2 XL original | 0 | 2.5567 | 2.4436¹ | 2.3739 |
| schaeff GPT-2 XL vanilla | 800 | 2.4799 | 2.4673 | 2.3821 |
| schaeff GPT-2 XL LN-free | 800 | 2.5052 [+0.0253] | 130.2197² | 2.3992 [+0.0253] |

**Footnotes:**
1. GPT-2 XL original: Median: 1.0103, 95 Percentile Range: [0.0005, 10.6193], 99.9% Percentile Range [≈0.0000, 43.0064]
2. GPT-2 XL LN-free: Median: 1.0937, 95 Percentile Range: [0.0004, 10.7548], 99.9% Percentile Range [≈0.0000, 48.6459]

## Setup

You will need 80GB+ GPU memory for most schedules - typically A100.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

The training script implements the fine-tuning procedure for removing layer-wise normalization from GPT-2 models. It is driven by schedules encoded in `config.py`. Training can be resumed from checkpoint; it will respect partially removed layer norm.

### List training schedules:
```bash
python config.py list
```

### Show specific config:
```bash
python config.py show {config_name}
```

### Fine-tune away layer norm:
```bash
python train.py --mode without_ln --config {config_name}
```

## Evaluation

The evaluation script calculates CE loss for several datasets to evaluate the output model. The script supports three formats:
- Original Hugging Face transformers model
- LN-free checkpoint
- LN-free transformers model which has had its eps and weights scaled so that layer norm is effectively disabled

### Evaluate on The Pile:
```bash
python eval_pile.py -m gpt2 -f transformers -b 8 -d pile-apollo
```

### Reproduce paper results:
This task is compute intensive and takes about 4h on an A100.

```bash
chmod +x eval_all.sh
./eval_all.sh
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you have found our work useful please cite as:

```
@misc{gpt2layernorm2025,
  author = {Baroni, Luca and Khara, Galvin and Schaeffer, Joachim and Subkhankulov, Marat and Heimersheim, Stefan},
  title = {Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and the Implications for Mechanistic Interpretability},
  year = {2025},
  eprint = {2507.02559},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2507.02559v1}
}
```
