import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Mapping
import os, argparse, sys
from itertools import chain
import datasets
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
)
import torch
from transformers.testing_utils import CaptureLogger

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
    handlers=[logging.StreamHandler(sys.stdout)],)
log_level = logging.INFO
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

model_args = argparse.Namespace()
model_args.cache_dir = None
model_args.use_fast_tokenizer = True
model_args.model_revision = "main"
model_args.use_auth_token = False

# data_args = argparse.Namespace()
# data_args.block_size = 1024
# block_size = data_args.block_size
# data_args.preprocessing_num_workers = None
# data_args.validation_split_percentage = 0.002

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}

tokenizer = None


# tokenizer.add_eos_token = True  # 指令微调没有，只让模型加bos
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

IGNORE_INDEX = -100
PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    )

# 指令微调必须分别限制input和target，防止input本身已经达到了max_seq_length导致output被全部截断
# max_seq_length = 2048
# max_input_length = 1536
# max_target_length = 512

max_seq_length = 1024
max_input_length = 768
max_target_length = 256

def tokenize_function(examples):
    sources = []
    targets = []
    prompt = PROMPT_TEMPLATE
    for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input is not None and input !="":
            instruction = instruction+'\n'+input
        source = prompt.format_map({'instruction':instruction})
        target = f"{output}{tokenizer.eos_token}"  # 这里加了eos即</s>

        sources.append(source)
        targets.append(target)

    tokenized_sources = tokenizer(sources, return_attention_mask=False)  # 句头加bos即<s>
    tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)  # 句头不加bos即<s>
    # 因为是分开来token所以target前面都是以▁A 相当于空格开头[/INST] A
    # 其实最终是相当于sources和targets加了空格去拼接  真无语了 
    # <s>[INST]......[/INST] ABCxxx

    all_input_ids = []
    all_labels = []
    for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
        # 保证有output
        if len(s) > max_input_length:
            s = s[:max_input_length]

        input_ids = torch.LongTensor(s + t)[:max_seq_length]
        labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
        assert len(input_ids) == len(labels)
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    results = {'input_ids':all_input_ids, 'labels': all_labels}
    return results


def gen_arrow(files:List, output_arrow_dir, tokenizer_name_or_path, cache_dir=None):
    global tokenizer
    global tokenizer_kwargs
    if tokenizer is None:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    lm_datasets = []
    for idx, file in enumerate(files):
        # print(file)
        file = Path(file)
        file_dir = file.parent
        if cache_dir is None:
            cache_dir = file_dir
        else:
            cache_dir = Path(cache_dir)
        file_name = file.stem
        arrow_cache_path = cache_dir / file_name
        arrow_cache_path.mkdir(parents=True, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(str(arrow_cache_path), keep_in_memory=False)
            logger.info(f'training datasets-{file_name} has been loaded from disk')
        except Exception:
            load_cache_path = cache_dir / (file_name + '_load')
            load_cache_path.mkdir(parents=True, exist_ok=True)
            map_cache_path = cache_dir / (file_name + '_map')
            map_cache_path.mkdir(parents=True, exist_ok=True)
            raw_dataset = load_dataset("json", data_files=str(file), cache_dir=str(load_cache_path), keep_in_memory=False)
            logger.info(f"{file} has been loaded")
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=None,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
                load_from_cache_file=True,
                cache_file_names = {k: str(map_cache_path / 'tokenized.arrow') for k in raw_dataset},
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(arrow_cache_path)
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
    print(lm_datasets)

    print('split train test')
    # lm_datasets = lm_datasets.train_test_split(test_size = 0.05, seed=1234)
    lm_datasets = lm_datasets.train_test_split(test_size = 0.05, seed=1234)
    # lm_datasets = lm_datasets.train_test_split(test_size = 1000, seed=1234)
    print(lm_datasets)
    # lm_datasets.save_to_disk(cache_dir.parent / 'final_merge_split3', num_proc=1)
    lm_datasets.save_to_disk(cache_dir.parent / output_arrow_dir, num_proc=1)
    
if __name__=='__main__':

    root_path = '/disk0/fin_group/zyn/'
    root_path = '/home/yss/'
    root_path = './'

    files = [
        f'{root_path}instr_data/cj_instr.json',
    ]
    gen_arrow(files, "arrow_data1219", 'tokenizer_chinese_llama', cache_dir=f'{root_path}instr_data/1219/cache')  # 使用原chinese_llama词表
    # gen_arrow(files, "arrow_data1219", 'tokenizer_lamber', cache_dir=f'{root_path}instr_data/1219_lamber/cache')  # 使用lamber词表
        
