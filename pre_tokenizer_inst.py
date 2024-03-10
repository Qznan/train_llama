import logging
from pathlib import Path
from typing import *
import os, argparse, sys
import datasets
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import LlamaTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
                    handlers=[logging.StreamHandler(sys.stdout)], )

log_level = logging.INFO
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

IGNORE_INDEX = -100
PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)
num_tokens_prompt_prefix = None
num_tokens_prompt_suffix = None

PROMPT_TEMPLATE_PREFIX = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
)

PROMPT_TEMPLATE_SUFFIX = (
    "[/INST]"  # 因为是句子开头[会默认被编码为_[，刚好相当于原来的空格
)


def init_prompt_token_num():
    global tokenzier
    # 默认微调是句尾不加</s>
    tokens_prompt_prefix = tokenizer(PROMPT_TEMPLATE_PREFIX, return_attention_mask=False)['input_ids']  # 句头加<s>
    tokens_prompt_suffix = tokenizer(PROMPT_TEMPLATE_SUFFIX, return_attention_mask=False, add_special_tokens=False)['input_ids']  # 句头不加<s>

    global num_tokens_prompt_prefix
    global num_tokens_prompt_suffix
    num_tokens_prompt_prefix = len(tokens_prompt_prefix)
    num_tokens_prompt_suffix = len(tokens_prompt_suffix)
    print(f'num_tokens_prompt_prefix: {num_tokens_prompt_prefix}\nnum_tokens_prompt_suffix: {num_tokens_prompt_suffix}')


def tokenize_function(examples):
    sources = []
    targets = []
    prompt = PROMPT_TEMPLATE
    for instruction, input_, output in zip(examples['instruction'], examples['input'], examples['output']):
        if input_ is not None and input_ != "":
            instruction = instruction + '\n' + input_
        source = prompt.format_map({'instruction': instruction})
        target = f"{output}{tokenizer.eos_token}"  # 这里加了eos即</s>

        sources.append(source)
        targets.append(target)

    tokenized_sources = tokenizer(sources, return_attention_mask=False)  # 句头加<s>
    tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)  # 句头不加<s>
    # 因为是分开来token所以target前面都是以▁A 相当于空格开头[/INST] A
    # 其实最终是相当于sources和targets加了空格去拼接  真无语了
    # <s>[INST]......[/INST] ABCxxx

    all_input_ids = []
    all_labels = []
    for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
        # 保证最终input_ids有target, 所以source和target分别截断
        if len(s) > max_input_length:
            ori_len = len(s)
            # s = s[:max_input_length]  # 可能导致"\n\n{instruction} [/INST]"这些被截断，故做以下处理：

            max_instruction_length = max_input_length - num_tokens_prompt_prefix - num_tokens_prompt_suffix

            # max_instruction_length 第1)种选择：考虑到指令一般会在最前面或者最后面，最大允许超过1000，优先保留前后各500token
            if max_instruction_length > 1000:
                start, end = 500, 500
                s = s[:num_tokens_prompt_prefix + start + (max_instruction_length - 1000)] + \
                    s[-num_tokens_prompt_suffix - end:]
            else:  # 最大不允许1000,则前后对半
                start = max_instruction_length // 2
                end = max_instruction_length - start
                s = s[:num_tokens_prompt_prefix + start] + \
                    s[-num_tokens_prompt_suffix - end:]

            # # max_instruction_length 第2)种选择：直接保留instruction的前面部分
            # s = s[:num_tokens_prompt_prefix+max_instruction_length] + \
            #     s[-num_tokens_prompt_suffix:]
            # # max_instruction_length 第3)种选择：直接保留instruction的后面部分
            # s = s[:num_tokens_prompt_prefix] + \
            #     s[-num_tokens_prompt_suffix-max_instruction_length:]

            assert len(s) <= max_input_length
            print(f'\nsource length {ori_len} is larger than {max_input_length}, have been truncated to {len(s)}')
            # print(f'{tokenizer.decode(s)}\n')
            # input()

        input_ids = (s + t)[:max_seq_length]
        labels = ([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
        assert len(input_ids) == len(labels)
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    results = {'input_ids': all_input_ids, 'labels': all_labels}
    return results


def gen_arrow(files: List, output_dir, merge_arrow_dir='merge_arrow_data'):
    lm_datasets = []
    cache_load_dir = os.path.join(output_dir, 'cache_load')
    cache_map_dir = os.path.join(output_dir, 'cache_map')
    arrow_dir = os.path.join(output_dir, 'single_arrow_data')
    merge_arrow_dir = os.path.join(output_dir, merge_arrow_dir) if merge_arrow_dir is not None else None  # 存放合并后的Dataset/DatasetDict的arrow格式文件的目录
    (os.makedirs(d, exist_ok=True) for d in [cache_load_dir, cache_map_dir, arrow_dir])

    for idx, file in enumerate(files):
        logger.info(f'loading {file}...')
        file_name = Path(file).stem
        _arrow_dir = os.path.join(arrow_dir, file_name)

        try:
            processed_dataset = datasets.load_from_disk(_arrow_dir, keep_in_memory=False)
            logger.info(f'Find cache of single file {file}')

        except Exception:

            _cache_load_dir = os.path.join(cache_load_dir, file_name)  # 单个文件的cache目录
            raw_dataset = load_dataset("json", data_files=file, cache_dir=_cache_load_dir, keep_in_memory=False)
            logger.info(f"{file} has finished loaded, load cache file: {_cache_load_dir}")

            # 去除其余字段
            raw_dataset['train'] = raw_dataset['train'].select_columns(["instruction", "input", "output"])

            _cache_map_dir = os.path.join(cache_map_dir, file_name)  # 单个文件的cache目录
            os.makedirs(_cache_map_dir, exist_ok=True)
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=None,  # 量太小不需要
                # num_proc=32,
                remove_columns=["instruction", "input", "output"],
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(_cache_map_dir, 'tokenized.arrow') for k in raw_dataset},
                desc="preprocessing on dataset",
            )
            logger.info(f"{file} has finished map func (tokenizer), map cache file: {_cache_load_dir}")

            processed_dataset = tokenized_dataset

            processed_dataset.save_to_disk(_arrow_dir)

        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

    if merge_arrow_dir is None:
        logger.info(f'Finish process all files. not merge because merge_arrow_dir is None')
        return

    logger.info(f'Finish process all files. merge output datasets: {lm_datasets}')

    if validation_split_percentage is not None:
        logger.info(f'split train and test, test ratio or num: {validation_split_percentage} seed=1234')
        lm_datasets = lm_datasets.train_test_split(test_size=validation_split_percentage, seed=1234)
        logger.info(f'Finish split train and test. merge output datasets: {lm_datasets}')

    lm_datasets.save_to_disk(merge_arrow_dir, num_proc=1)
    logger.info(f'Finish saved merge output datasets path: {merge_arrow_dir}')

    with open(output_dir + f'/{Path(tokenizer_path).stem}.info', 'w', encoding='U8') as f:
        f.write(tokenizer_path + '\n')


if __name__ == '__main__':
    validation_split_percentage = None
    validation_split_percentage = 1000  # 小数是比例，整数则是测试样本数量
    validation_split_percentage = 0.05  # 小数是比例，整数则是测试样本数量

    # 指令微调必须分别限制input和target，防止input本身已经达到了max_seq_length导致output被全部截断
    # max_seq_length = 2048
    # max_input_length = 1536
    # max_target_length = 512
    max_seq_length = 1024
    max_input_length = 768
    max_target_length = 256
    max_seq_length = 4096
    max_input_length = 2048
    max_target_length = 2048

    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }

    tokenizer_path = 'tokenizer_chinese_llama'  # 使用chinese_alpca2词表

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    # tokenizer.add_eos_token = True  # 指令微调没有，只让模型加bos
    init_prompt_token_num()

    root_path = '/disk0/fin_group/zyn/'
    root_path = '/home/yss/'
    root_path = './'

    files = [
        f'{root_path}instr_data/cj_instr.json',
    ]
    output_dir = 'instr_data/0111'

    """
    会生成：
    cache_load load时生成，files中每个文件单独一份
    cache_map map时生成，每个文件单独一份
    single_arrow_data map完成后生成，每个文件单独一份
    merge_arrow_data 将上述所有文件合并 并且train test split.
    """
    gen_arrow(files, output_dir)
    # gen_arrow(files, output_dir, merge_arrow_dir=None)
