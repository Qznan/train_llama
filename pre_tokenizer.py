import logging
from pathlib import Path
from typing import *
import os, argparse, sys
from itertools import chain
import datasets
from datasets import load_dataset, concatenate_datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)
from transformers.testing_utils import CaptureLogger

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


def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples["text"])  # 因为add_eos_token=True,所以将会增加sos和eos即 <s>  </s>给每行句子
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # # customed 保留最后的遗留数据 small remainde
    # if len(concatenated_examples[list(examples.keys())[0]]) % block_size != 0:
    #     for k in result.keys():
    #         result[k] += [concatenated_examples[k][-block_size:]]

    result["labels"] = result["input_ids"].copy()
    return result


def gen_arrow(files: List, output_dir, merge_arrow_dir='merge_arrow_data'):
    lm_datasets = []
    cache_load_dir = os.path.join(output_dir, 'cache_load')  # 存放各个小文件的load即(load_dataset)产生的cache目录
    cache_map_dir = os.path.join(output_dir, 'cache_map')  # 存放各个小文件的map产生的cache目录
    arrow_dir = os.path.join(output_dir, 'single_arrow_data')  # 存放每个处理好的Dataset/DatasetDict的arrow格式文件的目录
    merge_arrow_dir = os.path.join(output_dir, merge_arrow_dir)  # 存放合并后的Dataset/DatasetDict的arrow格式文件的目录
    (os.makedirs(d, exist_ok=True) for d in [cache_load_dir, cache_map_dir, arrow_dir])

    for idx, file in enumerate(files):
        logger.info(f'loading {file}...')
        file_name = Path(file).stem
        # _arrow_dir = os.path.join(arrow_dir, file_name + f'_{block_size}')
        _arrow_dir = os.path.join(arrow_dir, file_name)  # 每单个处理好的Dataset/DatasetDict的arrow格式文件的目录。e.g.single_arrow_data/test1w_1/

        try:
            processed_dataset = datasets.load_from_disk(_arrow_dir, keep_in_memory=False)  # e.g.single_arrow_data/test1w_1/
            logger.info(f'training datasets-{file_name} has been loaded from disk')

        except Exception:
            """ load_dataset https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/loading_methods#datasets.packaged_modules.text.TextConfig """
            _cache_load_dir = os.path.join(cache_load_dir, file_name)  # 单个文件的load即(load_dataset)产生的cache目录
            raw_dataset = load_dataset("text", data_files=file, cache_dir=cache_load_dir, keep_in_memory=False,
                                       keep_linebreaks=False,  # 是否保持\n
                                       sample_by='line'  # line(\n分割) | paragraph(\n\n分割) | document(整个文件整篇一起)
                                       )  # 默认是生成只有train split的DatasetDict
            logger.info(f"{file} has finished loaded, [load] cache file: {_cache_load_dir}")

            _cache_map_dir = os.path.join(cache_map_dir, file_name)  # 单个文件的map产生的cache目录
            os.makedirs(_cache_map_dir, exist_ok=True)
            """ map https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.DatasetDict.map """
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,  # batch_size默认是1000
                num_proc=32,  # 多线程，默认是1
                remove_columns="text",
                load_from_cache_file=True,  # 如检测到有相同函数的计算结果缓存，是否直接读取
                keep_in_memory=False,
                # 将map中函数的计算结果缓存，可不同split(train/test)中指定不同
                cache_file_names={k: os.path.join(_cache_map_dir, 'tokenized.arrow') for k in raw_dataset},
                desc="Running tokenizer on dataset",
            )
            logger.info(f"{file} has finished map func (tokenizer), [map] cache file: {_cache_load_dir}")

            grouped_datasets = tokenized_dataset.map(
                group_texts,
                batched=True,  # batch_size默认是1000
                num_proc=32,  # 多线程，默认是1
                load_from_cache_file=True,  # 如检测到有相同函数的计算结果缓存，是否直接读取
                keep_in_memory=False,
                # 将map中函数的计算结果缓存，可不同split(train/test)中指定不同
                cache_file_names={k: os.path.join(_cache_map_dir, 'grouped.arrow') for k in tokenized_dataset},
                desc=f"Grouping texts in chunks of {block_size}",
            )
            logger.info(f"{file} has finished map func (group), [map] cache file: {_cache_load_dir}")

            processed_dataset = grouped_datasets
            """ save_to_disk https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.Dataset.save_to_disk """
            processed_dataset.save_to_disk(_arrow_dir)  # 处理好的单个arrow输出目录 e.g.single_arrow_data/test1w_1/

        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

    logger.info(f'Finish process all files. merge output datasets: {lm_datasets}')

    if validation_split_percentage is not None:
        logger.info(f'split train and test, test ratio or num: {validation_split_percentage} seed=1234')
        lm_datasets = lm_datasets.train_test_split(test_size=validation_split_percentage, seed=1234)
        logger.info(f'Finish split train and test. merge output datasets: {lm_datasets}')

    lm_datasets.save_to_disk(merge_arrow_dir, num_proc=32)  # 存储也需要多线程不然很慢，默认按每个分片shard最大500M自动分片存，个数取分片后数量和线程数最大值
    logger.info(f'Finish saved merge output datasets path: {merge_arrow_dir}')

    with open(output_dir + f'/{Path(tokenizer_path).stem}.info', 'w', encoding='U8') as f:
        f.write(tokenizer_path + '\n')


if __name__ == '__main__':
    validation_split_percentage = None
    validation_split_percentage = 1000  # 小数是比例，整数则是测试样本数量
    validation_split_percentage = 0.05  # 小数是比例，整数则是测试样本数量
    validation_split_percentage = 0.002  # 小数是比例，整数则是测试样本数量

    block_size = 1024
    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": False,
    }

    tokenizer_path = 'tokenizer_chinese_llama'  # 使用chinese_alpca2词表
    tokenizer_path = 'tokenizer_chinese_llama'

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    tokenizer.add_eos_token = True  # 预训练有，让模型加bos和eos

    root_path = '/disk0/fin_group/zyn/'
    root_path = '/home/yss/'
    root_path = './'

    files = [
        f'{root_path}pt_data/test100_1.txt',
        f'{root_path}pt_data/test100_2.txt',
    ]

    output_dir = 'pt_data/0111'

    """
    会生成：
    cache_load load时生成，files中每个文件单独一份
    cache_map map时生成，每个文件单独一份，包括tokenize和group
    single_arrow_data map完成后生成，每个文件单独一份
    merge_arrow_data 将上述所有文件合并 并且train test split.
    """
    gen_arrow(files, output_dir)
