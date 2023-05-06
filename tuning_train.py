# -*- coding:utf-8 -*-

"""
This file provides a method to tuning LLaMa model with financial data:
### Lora + int8_training ###
"""

import argparse
import os
import sys
from typing import List

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from utils.prompter import Prompter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser()
# model/data params
parser.add_argument("--base_model", default='', type=str, required=True, help="original pretrained llama weights")
parser.add_argument("--data_path", default='yahma/alpaca-cleaned', type=str, help="dataset of SFT.")
parser.add_argument("--output_dir", default='./lora-alpaca', type=str, help="The path of lora model.")
# training hyperparams
parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--micro_batch_size", default=8, type=int, help="Batch size per process for training.")
parser.add_argument("--num_epochs", default=8, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--cutoff_len", default=512, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--val_set_size", default=1200, type=int, help="Batch size for evaluate.")
# lora hyperparams
parser.add_argument("--lora_r", default=8, type=int, help="The number of Lora ranks.")
parser.add_argument("--lora_alpha", default=16, type=int, help="Set the alpha parameter of LORA.")
parser.add_argument("--lora_dropout", default=0.05, type=float, help="Set the dropout parameter of LORA.")
parser.add_argument("--lora_target_modules", default=["q_proj", "v_proj"], type=List[str],
                    help="Set the target module for the PEFT model.")
# llm hyperparams
parser.add_argument("--train_on_inputs", default=False, type=bool, help="if False, masks out inputs in loss.")
parser.add_argument("--group_by_length", default=False, type=bool,
                    help="faster, but produces an odd training loss curve.")
# wandb params
parser.add_argument("--wandb_project", default='llama_fin', type=str, help="The name of wandb_project.")
parser.add_argument("--wandb_run_name", default='', type=str)
parser.add_argument("--wandb_watch", default='', type=str, choices=['false', 'gradients', 'all'])
parser.add_argument("--wandb_log_model", default='', type=str, choices=['false', 'true'])
parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                    help="Either training checkpoint or final adapter.")
parser.add_argument("--prompt_template_name", default='alpaca', type=str,
                    help="The prompt template to use, will default to alpaca.")
args = parser.parse_args()


def do_tuning():
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            "Training Alpaca-LoRA model with params:\n"
            "base_model: {}\n".format(args.base_model),
            "data_path: {}\n".format(args.data_path),
            "output_dir: {}\n".format(args.output_dir),
            "batch_size: {}\n".format(args.batch_size),
            "micro_batch_size: {}\n".format(args.micro_batch_size),
            "num_epochs: {}\n".format(args.num_epochs),
            "learning_rate: {}\n".format(args.learning_rate),
            "cutoff_len: {}\n".format(args.cutoff_len),
            "val_set_size: {}\n".format(args.val_set_size),
            "lora_r: {}\n".format(args.lora_r),
            "lora_alpha: {}\n".format(args.lora_alpha),
            "lora_dropout: {}\n".format(args.lora_dropout),
            "lora_target_modules: {}\n".format(args.lora_target_modules),
            "train_on_inputs: {}\n".format(args.train_on_inputs),
            "group_by_length: {}\n".format(args.group_by_length),
            "wandb_project: {}\n".format(args.wandb_project),
            "wandb_run_name: {}\n".format(args.wandb_run_name),
            "wandb_watch: {}\n".format(args.wandb_watch),
            "wandb_log_model: {}\n".format(args.wandb_log_model),
            "resume_from_checkpoint: {}\n".format(args.resume_from_checkpoint or None),
            "prompt template: {}\n".format(args.prompt_template_name)
        )

    # --------------------------Check--------------------------
    # Check if parameters passed or set in the environment
    use_wandb = len(args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environment if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    # Check if the base_model exists
    assert args.base_model, "Please specify a base_model, for example: 'decapoda-research/llama-7b-hf'"

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:  # ddp adopts a multiprocessing approach
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # --------------------------Model--------------------------
    # 与全精度模型相比，以 8 位精度加载模型最多可节省 4 倍的内存
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = prepare_model_for_int8_training(model)

    # using lora to tuning
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Check the available weights and load
    if args.resume_from_checkpoint:
        # Full checkpoint
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )

        # only LoRA model
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
            # So the trainer won't try loading its state
            args.resume_from_checkpoint = None
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print("Restarting from {}".format(checkpoint_name))
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print("Checkpoint {} not found".format(checkpoint_name))

    # More transparent to x% of trainable parameters
    model.print_trainable_parameters()

    # --------------------------Tokenizer--------------------------
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0          # 「unk」 different from the eos token
    tokenizer.padding_side = "left"     # allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    # choose prompt template
    prompter = Prompter(args.prompt_template_name)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len \
                                              + tokenized_full_prompt["labels"][user_prompt_len:]  # Maybe faster
        return tokenized_full_prompt

    # load dataset from file(here is xx.json)
    if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.data_path)
    else:
        data = load_dataset(args.data_path)

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=2023
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # --------------------------Trainer--------------------------
    # 当多张显卡时，阻止 Trainer 使用自己的 DataParallelism
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=32 if args.val_set_size > 0 else None,
            save_steps=32,
            output_dir=args.output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=args.wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(args.output_dir)

    print("\n 若上面出现有关于keys丢失的警告，请忽略! o(^_^)o ～")


if __name__ == "__main__":
    do_tuning()
