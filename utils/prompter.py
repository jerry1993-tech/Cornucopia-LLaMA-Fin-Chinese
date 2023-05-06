# -*- coding:utf-8 -*-

"""
Choose template to build prompt.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    """
    prompt 构造器 -> chose template
    """
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose

        if not template_name:
            template_name = "alpaca"   # default
        file_name = osp.join("templates", "{}.json".format(template_name))

        if not osp.exists(file_name):
            raise ValueError("Can't read {}".format(file_name))

        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print("Chose prompt template {0}: {1}".format(template_name, self.template['description']))

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        """
        从指令和可选输入返回完整 prompt，如果提供了一个label (=response, =output)，也会被添加
        """
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = "{0}{1}".format(res, label)
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
