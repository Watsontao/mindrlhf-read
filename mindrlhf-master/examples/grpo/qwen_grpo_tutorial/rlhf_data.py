# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    data process
"""
import argparse
import json
import os
import jsonlines
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from mindformers import logger
from mindrlhf.models.qwen2_5.qwen2_5_tokenizer import Qwen2_5Tokenizer



# python里的数据格式

##
# list 列表  列表list[0]  list[1]   有序  可以重复  可变  []
# dic 字典  键值对   键不可重复   {} + key: value
# tuple 元组 不可修改的的列表  有序 可以重复 不可变       ()
# set 集合  无序 不可以重复（自动去重） 可变   用 {} 或 set() 定义
#
#
# #



def load_json_file(file_path):
    """
    Read data from json file
    """
    ext = os.path.splitext(file_path)[1].lower().replace(".", "")  # 拿到后缀名
    with open(file_path, "r", encoding="utf-8") as f: # "r" 表示 Read (只读) 模式   打开个文件 并命名为f
        if ext == "jsonl":  # 普通 JSON 是一个大对象，而 JSONL (JSON Lines) 是每一行都是一个独立的 JSON 对象。
            raw_data = [] # 定义一个列表
            for item in jsonlines.Reader(f):
                raw_data.append(item)
            return raw_data
        if ext == "json":
            return json.load(f)
        raise ValueError("data files should be jsonl or json")




def process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id, dataset_type):
    """
    process_data
    """

    for item in tqdm(raw_data):  # tqdm() 进度条工具  长度为raw_data的长度
        sample = {} # 定义一个字典
        if dataset_type == "gsm8k":
            prompt = item["question"]
            response = item["answer"].split("#### ")[-1]
            # 切分前："推理过程... #### 3"
            # 切分后（变成列表）：["推理过程...", "3"][-1] 是负数索引  代表倒数第一个数
        elif dataset_type == "dapo17k":
            prompt = item["prompt"][0]["content"].split("\n\n")[1]  # [0]：取这个列表的第 1 个元素。
            response = item["reward_model"]["ground_truth"] # 嵌套字典取值：
        elif dataset_type == "openR1math":
            prompt = item["question"]
            response = item["answer"]
        elif dataset_type == "deepscaler":
            template = ("A conversation between User and Assistant. The user asks a question, and the Assistant"
                        " solves it. The assistant first thinks about the reasoning process in the mind and then"
                        " provides the user with the answer. The reasoning process is enclosed within <think>"
                        " </think> tags, i.e., <think> reasoning process here </think> answer here. "
                        "User: {} Assistant: <think>\n")
            prompt = template.format(item["problem"])  # .format() 填空   上面的User:{} 是占位符
            response = item["answer"]
        else:
            prompt = item["question"]
            response = item["answer"]



        # tokenizer 把字变成数字列表  truncation=True 如果句子太长超过了限制，就咔嚓切断，防止报错
        # tokenizer() 返回的是一个list
        prompt_dict = tokenizer(prompt, truncation=True, max_length=max_prompt_length, add_special_tokens=False)

        response_dict = tokenizer(response, truncation=True, max_length=seq_length, add_special_tokens=False)


        # numpy数组，比 list好用
        # 把 list 转换为 numpy
        prompt_ids = np.array(prompt_dict["input_ids"])
        prompt_len = prompt_ids.shape[-1] # shape[-1] 获取数组最后一个维度的长度
        pretrain_ids = np.array(response_dict["input_ids"])
        loss_mask = np.array(response_dict["attention_mask"])  # attention_mask 告诉模型哪些是字，哪些是填充


        prompt_ids = np.pad(
            prompt_ids, (0, max_prompt_length - prompt_ids.shape[-1]), "constant", constant_values=(0, pad_token_id)
        )
        pretrain_ids = np.pad(
            pretrain_ids, (0, seq_length - pretrain_ids.shape[-1]), "constant", constant_values=(0, pad_token_id)
        )
        loss_mask = np.pad(
            loss_mask, (0, seq_length - loss_mask.shape[-1]), "constant", constant_values=(0, pad_token_id)
        )


        loss_mask[:prompt_len] = 0.0   #  loss_mask 原本全是 1（表示所有字都要计分）    这里设置下 题目部分的权重设置为0   这个代码loss_mask后面没有用到

        ##
        # loss_mask 它存储的是由 0 和 1 组成的掩码（Mask）， 主要是告诉模型 1表示重要 需要计算分 0表示不重要 不需要计算分
        # “算分” = “计算误差” = “决定要不要根据这个字的表现去修改模型参数”
        # loss_mask 原本全是 1（表示所有字都要计分）。
        #
        # [:prompt_len] 选中了题目部分。
        #
        # = 0.0 把这部分的权重设为 0。
        # #

        sample["prompt_ids"] = prompt_ids
        sample["pretrain_ids"] = pretrain_ids
        sample["loss_mask"] = loss_mask

        yield sample  ##

        ##
        # 这是 Python 中非常高效的**生成器（Generator）**语法。
        #
        # 普通做法 (return / list.append) —— 仓库堆积模式：
        #
        # 如果是 list.append(sample)，程序会把几万条数据全部处理完，全部堆在内存（仓库）里，最后一次性发货。
        #
        # 缺点：如果数据有 100GB，你的电脑内存直接爆炸（Out of Memory）。
        #
        # 生成器做法 (yield) —— 传送带模式：
        #
        # yield 的意思是**“产出这一个”，然后暂停**。
        #
        # 当训练程序喊“下一条！”时，它才继续跑下一轮循环，处理下一条数据。
        #
        # 优点：无论数据有多少亿条，它同一时间只占用一条数据的内存。这就是为什么处理大数据集时一定要用 yield。#


def write_mindrecord(args):
    """
    write_mindrecord
    """

    raw_data = load_json_file(args.file_path)

    tokenizer = Qwen2_5Tokenizer(args.vocab_path, args.merges_file_path, add_bos_token=False, add_eos_token=False)

    max_prompt_length = int(args.max_prompt_length) # prompt长度
    seq_length = int(args.seq_length)  # response长度
    if args.pad_token_id is None:  # # 如果没指定填充用的 ID，就问分词器要一个默认的
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = int(args.pad_token_id)

    # 定义一个蓝图 以防计算机看不懂
    schema = {
        # 第一列   shape是尺寸"shape": [10] 表示是10个数字 [-1]表示一维数组
        "prompt_ids": {"type": "int64", "shape": [-1]},   # 存放题目ID
        # 第二列
        "pretrain_ids": {"type": "int64", "shape": [-1]}, # 存放答案ID
        # 第三列
        "loss_mask": {"type": "int64", "shape": [-1]},   # 存放打分掩码
    }

    # shard_num=1 只生成一个文件
    writer = FileWriter(file_name=args.output_path, shard_num=1, overwrite=True)
    writer.add_schema(schema)

    count = 0
    for sample in process_data(tokenizer, raw_data, max_prompt_length, seq_length, pad_token_id, args.dataset_type):
        try:
            writer.write_raw_data([sample])
        except Exception as e:
            logger.error(f"Error occurred when writing sample {sample}, error: {e}")
        else:
            # 没出错就 +1
            count += 1

        if args.trunc_num > 0 and count >= args.trunc_num:
            break

    logger.info(f"Total number of samples: {count}")

    writer.commit()
    logger.info(f"Transformation finished! Output file refer: {args.output_path}")


def get_args():
    """
    get args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", required=True, help="path to vocab.json")
    parser.add_argument("--merges_file_path", required=True, help="path to merges.txt")
    parser.add_argument("--file_path", required=True, help="file path to raw data.")
    parser.add_argument("--output_path", required=True, help="file path to output mindrecord file.")
    parser.add_argument("--max_prompt_length", default=2048, help="max prompt encode length.")
    parser.add_argument("--seq_length", default=4096, help="encoded sequence length.")
    parser.add_argument("--pad_token_id", default=None, help="pad token id.")
    parser.add_argument("--dataset_type", default=None, help="your dataset type?")
    parser.add_argument("--trunc_num", default=-1, type=int,
                        help="max number of samples to be written to mindrecord. -1 means all.")
    args_opt = parser.parse_args()  # 返回一个Namespace对象，这是argparse库中自定义的一个类
    return args_opt # arguments options


if __name__ == "__main__":
    my_args = get_args()
    write_mindrecord(my_args)
