#!/bin/bash


exp_tag="Meta"
python3 tuning_train.py \
    --base_model './base_models/llama-7b-hf' \
    --data_path './instruction_data/fin_data.json' \
    --output_dir './Fin-Alpaca-LoRA-7B-'$exp_tag \
    --prompt_template_name 'fin_template' \
    --micro_batch_size 64 \
    --batch_size 64 \
    --num_epochs 10 \
    --wandb_run_name $exp_tag


#exp_tag="Linly"
#python3 tuning_train.py \
#    --base_model './base_models/Linly-Chinese-LLaMA-7b-hf' \
#    --data_path './instruction_data/fin_data.json' \
#    --output_dir './Fin-Alpaca-LoRA-7B-'$exp_tag \
#    --prompt_template_name 'fin_template' \
#    --micro_batch_size 96 \
#    --batch_size 96 \
#    --num_epochs 10 \
#    --wandb_run_name $exp_tag