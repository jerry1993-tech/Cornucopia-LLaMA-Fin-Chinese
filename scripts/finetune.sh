#!/bin/bash


base_model_pir="./base_models/llama-7b-hf"
if [ ! -d $base_model_pir ];then
  cd ../base_models/ || exit
  git clone https://huggingface.co/decapoda-research/llama-7b-hf
  cd ../ || exit
fi


exp_tag="v1"
python3 tuning_train.py \
    --base_model './base_models/llama-7b-hf' \
    --data_path './instruction_data/fin_data.json' \
    --output_dir './lora-llama-fin-'$exp_tag \
    --prompt_template_name 'fin_template' \
    --micro_batch_size 64 \
    --batch_size 64 \
    --wandb_run_name $exp_tag