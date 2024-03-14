import datasets
from pathlib import Path
import json
import ipdb
from transformers import LlamaForCausalLM, LlamaTokenizer

"""
stats：统计arrow文件的token、样本数量
inspect_arrow：查看arrow文件的id或token
"""
tokenizer = None


def printjson(object, value_func=None, sort=True):
    object = object.copy()
    if value_func is not None and isinstance(object, dict):
        for k, v in object.items():
            object[k] = value_func(object[k])
    if sort:
        object = {key: object[key] for key in sorted(object.keys())}  # sorted了之后输出的json也是sorted的
    return print(json.dumps(object, indent=4, ensure_ascii=False))


def aggreate_stats(stats: dict):
    """ 合并统计前缀相同的key"""
    agg_stats = {}
    for k, v in stats.items():
        agg_stats.setdefault(k.split('_')[0], []).append(v)
    agg_stats = {k: sum(v) for k, v in agg_stats.items()}
    return agg_stats


def stats(datasets_dir):
    dirs = [d for d in Path(datasets_dir).iterdir() if d.is_dir()]
    print(dirs)
    tokens_stats = {}  # calc nums_token
    nums_stats = {}  # calc nums_token
    for d in dirs:
        try:
            dataset = datasets.load_from_disk(d / 'train', keep_in_memory=False)
            print(f'{d}\ndatasets (train split)-{dataset} has been loaded from disk')
            num = len(dataset)
            num_per_example = len(dataset[0][dataset.column_names[0]])  # 2048
            total_num = num * num_per_example
            print(f'guess num_per_example is {num_per_example} by {dataset.column_names[0]}, nums:{num} total:{total_num}\n')
            tokens_stats[str(d.name)] = total_num
            nums_stats[str(d.name)] = num
        except KeyboardInterrupt as e:
            print(e)
            exit(0)
        except Exception as e:
            print(e)
            continue

    print('tokens总量', sum(tokens_stats.values()))
    printjson(tokens_stats, lambda x: x / (10 ** 6))
    agg_tokens_stats = aggreate_stats(tokens_stats)
    printjson(agg_tokens_stats, lambda x: x / (10 ** 6))

    print('nums总量', sum(nums_stats.values()))
    printjson(nums_stats, lambda x: f'{x:,}')
    agg_nums_stats = aggreate_stats(nums_stats)
    printjson(agg_nums_stats, lambda x: f'{x:,}')


def decode_ids(ids):
    global tokenizer
    ret = [tokenizer.convert_ids_to_tokens(i) if i != -100 else 'igx' for i in ids]
    return ret


def inspect_arrow(dataset_dir, tokenizer_path):
    global tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=True)

    dataset = datasets.load_from_disk(dataset_dir + '/train', keep_in_memory=False)
    # dataset.select(range(10000))
    # ipdb.set_trace()
    print(f'{dataset_dir}\ndatasets (train split)-{dataset} has been loaded from disk')
    num = len(dataset)

    def print_info(i):
        if i >= num:
            print('i is too large (num:{num})')
            return

        exm = dataset[i]
        print(exm)
        print(f"\n====input_ids:\n{decode_ids(exm['input_ids'])}")
        print(f"\n====labels:\n{decode_ids(exm['labels'])}")

        print('\n====input_ids(txt):\n' + tokenizer.decode(exm['input_ids']))
        processed_labels = [i if i != -100 else 0 for i in exm['labels']]
        print('\n====labels(txt):\n' + tokenizer.decode(processed_labels))

        print('\n====input_ids(str):\n' + repr(tokenizer.decode(exm['input_ids'])))
        print('\n====labels(str):\n' + repr(tokenizer.decode(processed_labels)))

    print_info(10)
    ipdb.set_trace()


if __name__ == '__main__':
    inspect_arrow('pt_data/0111/merge_arrow_data', 'tokenizer_chinese_llama')

    stats('pt_data/0111/single_arrow_data')
