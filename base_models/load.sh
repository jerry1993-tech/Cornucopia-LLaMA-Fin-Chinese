#!/bin/bash


base_model_pir="./base_models/llama-7b-hf"
if [ ! -d $base_model_pir ];then
  cd ../base_models/ || exit
  git clone https://huggingface.co/decapoda-research/llama-7b-hf
  cd ../ || exit
fi

base_model_pir="./base_models/Linly-Chinese-LLaMA-7b-hf"
if [ ! -d $base_model_pir ];then
  cd ../base_models/ || exit
  git clone https://huggingface.co/P01son/Linly-Chinese-LLaMA-7b-hf
  cd ../ || exit
fi

