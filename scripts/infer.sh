#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False' and 'prompt_template' to 'ori_template'.
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b', and 'prompt_template' to 'alpaca'.
# If inferring with the llama-fin model, download the LORA weights and set 'lora_weights' to './lora-llama-fin-vx' (or the exact directory of LORA weights) and 'prompt_template' to 'fin_template'.

exp_tag="v1"
python3 infer.py \
    --base_model './base_models/llama-7b-hf' \
    --lora_weights './lora-llama-fin-'$exp_tag \
    --use_lora True \
    --instruct_dir './instruction_data/infer.json' \
    --prompt_template 'fin_template'
