# 提示词模板

此目录包含用于 LoRA 微调 LLaMa 模型的提示的模板样式。

## Format

模板是通过一个JSON文件描述的，该文件包含以下键：

- `prompt_input`: The template to use when input is not None. Uses `{instruction}` and `{input}` placeholders.
- `prompt_no_input`: The template to use when input is None. Uses `{instruction}` placeholders.
- `description`: A short description of the template, with possible use cases.
- `response_split`: The text to use as separator when cutting real response from the model output.

No `{response}` placeholder was used, since the response is always the last element of the template and is just to be concatenated to the rest.

## 模板案例

The default template, used unless otherwise specified, is `alpaca.json`

```json
{
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

```

## 现有模板

### alpaca

到目前为止，用于通用LoRA微调的默认模板。

### alpaca_legacy

原始羊驼使用的旧模板，响应字段后没有“\n”。保留以供参考和实验。

### alpaca_short

一个修剪过的羊驼模板，它似乎也表现得很好，并保留了一些 tokens。使用默认模板创建的模型似乎也可以通过短时间查询。
