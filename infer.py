# -*- coding:utf-8 -*-

"""
This file is to infer the tuned LLaMa model.
"""

import sys
import json
import argparse

import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument("--load_8bit", default=False, type=bool)
parser.add_argument("--base_model", default='', type=str, required=True, help="original pretrained llama weights")
parser.add_argument("--instruct_dir", default='', type=str, help="dataset of infer.")
parser.add_argument("--use_lora", default=True, type=bool)
parser.add_argument("--lora_weights", default='tloen/alpaca-lora-7b', type=str, help="The lora weights of llama.")
parser.add_argument("--prompt_template", default='fin_template', type=str, help="The template of infer data.")
args = parser.parse_args()


def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data


def main():
    # -----------------------------------------
    # 加载 prompt 模板与 llama模型
    prompter = Prompter(args.prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.use_lora:
        print("using lora {}".format(args.lora_weights))
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16
        )
    # 重新配置 decapoda-research config
    model.config.pad_token_id, tokenizer.pad_token_id = 0, 0  # unk token
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not args.load_8bit:
        model.half()                                          # 开启半精度

    model.eval()                                              # 保证BN层直接利用之前训练阶段得到的均值和方差

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # -----------------------------------------
    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    def infer_from_json(instruct_dir):
        input_data = load_instruction(instruct_dir)
        for d in input_data:
            instruction = d["instruction"]
            output = d["output"]
            print("###infering###")
            model_output = evaluate(instruction)
            print("###instruction###")
            print(instruction)
            print("###golden output###")
            print(output)
            print("###model output###")
            print(model_output)

    if args.instruct_dir != "":
        infer_from_json(args.instruct_dir)
    else:
        for instruction in [
            "有哪些比较好的高端理财平台？",
            "手上有20万存款，可以作什么投资",
            "净值型和传统理财产品的区别，有什么不同？",
            "我想了解一下创业板股票怎么买？"
        ]:
            print("Instruction:", instruction)
            print("Response:", evaluate(instruction))
            print()


if __name__ == "__main__":
    main()
