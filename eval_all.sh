# GPT2 Small:
python eval_pile.py --model openai-community/gpt2 --model-name gpt2 --format transformers -d pile-apollo;
python eval_pile.py --model openai-community/gpt2 --model-name gpt2 --format transformers -d pile-apollo-luca;
python eval_pile.py --model openai-community/gpt2 --model-name gpt2 --format transformers -d openwebtext;
python eval_pile.py --model schaeff/gpt2-small_vanilla300 --model-name gpt2 --format transformers -d pile-apollo;
python eval_pile.py --model schaeff/gpt2-small_vanilla300 --model-name gpt2 --format transformers -d pile-apollo-luca;
python eval_pile.py --model schaeff/gpt2-small_vanilla300 --model-name gpt2 --format transformers -d openwebtext;
python eval_pile.py --model submarat/gpt2-noln-ma-aux --model-name gpt2 --format noLN_HF_model -d pile-apollo;
python eval_pile.py --model submarat/gpt2-noln-ma-aux --model-name gpt2 --format noLN_HF_model -d pile-apollo-luca;
python eval_pile.py --model submarat/gpt2-noln-ma-aux --model-name gpt2 --format noLN_HF_model -d openwebtext;

# GPT2 Medium:
python eval_pile.py --model openai-community/gpt2-medium --model-name gpt2-medium --format transformers -d pile-apollo;
python eval_pile.py --model openai-community/gpt2-medium --model-name gpt2-medium --format transformers -d pile-apollo-luca;
python eval_pile.py --model openai-community/gpt2-medium --model-name gpt20medium --format transformers -d openwebtext;
python eval_pile.py --model schaeff/gpt-2medium_vanilla500 --model-name gpt2-medium --format transformers -d pile-apollo;
python eval_pile.py --model schaeff/gpt-2medium_vanilla500 --model-name gpt2-medium --format transformers -d pile-apollo-luca;
python eval_pile.py --model schaeff/gpt-2medium_vanilla500 --model-name gpt2-medium --format transformers -d openwebtext;
python eval_pile.py --model submarat/gpt2-medium-noln-ma-aux --model-name gpt2-medium --format noLN_HF_model -d pile-apollo;
python eval_pile.py --model submarat/gpt2-medium-noln-ma-aux --model-name gpt2-medium --format noLN_HF_model -d pile-apollo-luca;
python eval_pile.py --model submarat/gpt2-medium-noln-ma-aux --model-name gpt2-medium --format noLN_HF_model -d openwebtext;

# GPT2 Large:
python eval_pile.py --model openai-community/gpt2-large --model-name gpt2-large --format transformers -d pile-apollo;
python eval_pile.py --model openai-community/gpt2-large --model-name gpt2-large --format transformers -d pile-apollo-luca;
python eval_pile.py --model openai-community/gpt2-large --model-name gpt2-large --format transformers -d openwebtext;
python eval_pile.py --model schaeff/VanillaLarge600 --model-name gpt2-large --format transformers -d pile-apollo;
python eval_pile.py --model schaeff/VanillaLarge600 --model-name gpt2-large --format transformers -d pile-apollo-luca;
python eval_pile.py --model schaeff/VanillaLarge600 --model-name gpt2-large --format transformers -d openwebtext;
python eval_pile.py --model schaeff/LNFreeLarge600_v2 --model-name gpt2-large --format noLN_HF_model -d pile-apollo;
python eval_pile.py --model schaeff/LNFreeLarge600_v2 --model-name gpt2-large --format noLN_HF_model -d pile-apollo-luca;
python eval_pile.py --model schaeff/LNFreeLarge600_v2 --model-name gpt2-large --format noLN_HF_model -d openwebtext;

# GPT2 XLarge:
python eval_pile.py --model openai-community/gpt2-xl --model-name gpt2-xl --format transformers -d pile-apollo;
python eval_pile.py --model openai-community/gpt2-xl --model-name gpt2-xl --format transformers -d pile-apollo-luca;
python eval_pile.py --model openai-community/gpt2-xl --model-name gpt2-xl --format transformers -d openwebtext;
python eval_pile.py --model schaeff/VanillaXL800 --model-name gpt2-xl --format transformers -d pile-apollo;
python eval_pile.py --model schaeff/VanillaXL800 --model-name gpt2-xl --format transformers -d pile-apollo-luca;
python eval_pile.py --model schaeff/VanillaXL800 --model-name gpt2-xl --format transformers -d openwebtext;
python eval_pile.py --model schaeff/LNFreeXL800 --model-name gpt2-xl --format noLN_HF_model -d pile-apollo;
python eval_pile.py --model schaeff/LNFreeXL800 --model-name gpt2-xl --format noLN_HF_model -d pile-apollo-luca;
python eval_pile.py --model schaeff/LNFreeXL800 --model-name gpt2-xl --format noLN_HF_model -d openwebtext;