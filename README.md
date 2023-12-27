<h1 align="center">Customed scripts for training llama</h1>
<p align="center">
  <a href="https://github.com/Qznan/train_llama">
    <img src="https://img.shields.io/github/stars/Qznan/train_llama.svg?colorA=orange&colorB=orange&logo=github" alt="GitHub stars">
  </a>
  <a href="https://github.com/Qznan/train_llama/issues">
        <img src="https://img.shields.io/github/issues/Qznan/train_llama.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/Qznan/train_llama/">
        <img src="https://img.shields.io/github/last-commit/Qznan/train_llama.svg">
  </a>
   <a href="https://github.com/Qznan/train_llama/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/Qznan/train_llama.svg">
  </a>
  
</p>

### News置顶
- 🔥 更新vllm部署脚本（改进：一次部署langchain和openai均可使用）

### 背景
由于项目需要，整理出llama模型的微调代码，并发布以备份。代码参考来源是[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)，主要是将：微调数据tokenizer为arrow数据的逻辑部分独立出来了，解绑数据准备与微调步骤。修改了部分代码。增加了一些注释

目前只整理lora微调部分。全参微调只需再修改一点逻辑而已，待后续更新。

### 使用说明如下
```
首先根据requirements下载需要的包

lora微调说明
主要步骤为：
1、读取微调数据生成tokenzier后的数据，保存为arrow文件格式（运行python pre_tokenizer_inst.py）
2、修改01_run_sft.sh中运行脚本中的dataset_dir参数，指定为上一步输出的arrow文件目录，设置其余参数运行进行模型训练训练（运行01_run_sft.sh）
3、将训练得到的pt_lora_model跟原模型进行参数合并，得到完整模型（运行python merge_llama2_with_chinese_lora_low_mem.py）
4、可将完整模型路径写入到vllm启动脚本进行部署。

详细步骤：
1、详情查看pre_tokenizer_inst.py中的__main__方法，
    1.files中指定微调数据，这里给出的示例数据是instr_data/cj_instr.json。主要包括instruction、input和output字段。
    2.gen_arrow(files, "arrow_data1219", 'tokenizer_chinese_llama', cache_dir=f'{root_path}instr_data/1219/cache')  # 使用原chinese_llama词表
        tokenizer主要函数，参数由files,处理完的数据保存目录名arrow_data1219,所采用的分词器tokenizer_chinese_llama,处理中间过程数据的保存路径instr_data/1219/cache，
        最终会生成instr_data/1219/arrow_data1219文件目录

2、运行01_run_sft脚本
    1、01_run_sft.sh中指定dataset_dir参数为上一步的/instr_data/1219/arrow_data1219目录。并保持chinese_tokenizer_path参数与上一步所采用的的分词器相同。
    2、其余常见修改包括使用单机几卡的GPU。在CUDA_VISIBLE_DEVICES=0,1,2中设置，这个例子使用3卡，故nproc-per-node也要一并修改为3
    3、设置模型保存输出路径output_dir


3、训练完后运行python merge_llama2_with_chinese_lora_low_mem.py
    python merge_llama2_with_chinese_lora_low_mem.py \
    --base_model path/to/llama2-hf-model \  # 基座模型路径
    --lora_model path/to/chinese-llama2-or-alpaca2-lora \  # 上一步模型模型保存路径中的pt_lora_model目录
    --output_type [huggingface|pth|] \  # 一般设置为huggingface格式
    --output_dir path/to/output-dir # 合并后的新模型保存路径
```
