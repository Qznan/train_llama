# model: 模型ckpt路径
# port/host: 按自己需求设置
# gpu-memory-utilization：预分配的占用GPU比例，已验证13b模型可只用0.4*80G=32G即可部署
# tensor-parallel-size：模型并行数量，根据提供的GPU数量确定

model=/new_disk/models_for_all/chinese-alpaca-2-13b
model=/new_disk/models_for_all/llama-2-13b-hf
model=/new_disk/models_for_all/icrc_lambergpt/round1/lambergpt_13b_sft_1/sft
port=8200

CUDA_VISIBLE_DEVICES="0" \
nohup python api_server.py \
--model ${model} \
--port ${port} \
--host "0.0.0.0" \
--gpu-memory-utilization 0.4 \
--tensor-parallel-size 1 > log.txt 2>&1 &
