# Paper: Transformers Don't Need LayerNorm at Inference Time

Companion code for removing layer normalization from GPT-2 whilst preserving its next-token prediction performance.

## Setup

You will need 80GB+ GPU memory for most schedules - typically A100.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

The training script implements the fine-tuning procedure for removing layer-wise normalization from GPT-2 models.
It is driven by schedules encoded in config.py. Training can be resumed from checkpoint; it will respect partially
removed layer norm.

List training schedules:
```
python config.py list
```

Show specific config:
```shell
python config.py show {config_name}
```

Fine-tune away layer norm:
```shell
python train.py --mode without_ln --config {config_name}
```

## Evaluation

The evalution script calculates CE loss for several datasets to evaluate the output model.
The script supports three formats: original huggingface transformers model, LN-free checkpoint, LN-free transformers model
which has had its eps and weights scaled so that layer norm is effectively disabled.

Evaluate on The Pile:
```
python eval_pile.py -m gpt2 -f transformers -b 8 -d pile-apollo
```

To evaluate all models and to reproduce the results in the manuscript run:
This task is compute intensive and takes about 4h on an A100.

chmod +x eval_all.sh
./eval_all.sh
```

## Citation

If you have found our work useful please cite as:

```
@misc{bkss2025layernorm,
  author = {Baroni, Luca and Khara, Galvin and Schaeffer, Joachim and Subkhankulov, Marat and Heimersheim, Stefan},
  title = {Transformers Don't Need LayerNorm at Inference Time: Scaling LayerNorm Removal to GPT-2 XL and the Implications for Mechanistic Interpretability},
  year = {2025},
  eprint = {2507.02559},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2507.02559v1}
}
```
