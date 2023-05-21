#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False' and 'prompt_template' to 'ori_template'.
# If inferring with the llama-fin model, download the LORA weights and set 'lora_weights' to './lora-llama-fin-'$exp_tag (or the exact directory of LORA weights) and 'prompt_template' to 'fin_template'.

BASE_MODEL="./base_models/llama-7b-hf"
# only ori llama
o_cmd="python3 infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora False \
    --prompt_template 'ori_template'"

# lora-llama-fin-ori-fb
exp_tag="ori-fb"
a_cmd="python3 infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora True \
    --lora_weights './lora-llama-fin-'$exp_tag \
    --prompt_template 'fin_template'"

# lora-llama-fin-Linly-zh
BASE_MODEL="./base_models/Linly-Chinese-LLaMA-7b-hf"
exp_tag="Linly-zh"
m_cmd="python3 infer.py \
    --base_model ${BASE_MODEL} \
    --use_lora True \
    --lora_weights './lora-llama-fin-'$exp_tag \
    --prompt_template 'fin_template'"

echo "only_ori_llama"
eval $o_cmd > infer_result/o_tmp.txt
echo "lora-llama-fin-ori-fb"
eval $a_cmd > infer_result/a_tmp.txt
echo "lora-llama-fin-Linly-zh"
eval $m_cmd > infer_result/m_tmp.txt