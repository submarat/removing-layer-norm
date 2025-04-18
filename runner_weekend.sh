CUDA_VISIBLE_DEVICES=0 EXP_RECOMPUTE_STD_ON_REAL=1 EXP_CORRECT_BOS=1 python train.py --mode without_ln --config gpt2_standard_aux;
CUDA_VISIBLE_DEVICES=0 EXP_RECOMPUTE_STD_ON_REAL=1 EXP_CORRECT_BOS=1 python train.py --mode without_ln --config gpt2-medium_fasttune_aux;
CUDA_VISIBLE_DEVICES=0 EXP_RECOMPUTE_STD_ON_REAL=1 EXP_CORRECT_BOS=1 python train.py --mode without_ln --config gpt2-large_aux;
CUDA_VISIBLE_DEVICES=0 EXP_RECOMPUTE_STD_ON_REAL=1 EXP_CORRECT_BOS=1 python train.py --mode without_ln --config gpt2-xl_aux;