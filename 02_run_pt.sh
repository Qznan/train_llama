# root_path=/home/yss
# root_path=/disk0/fin_group/zyn
root_path=.

# export NCCL_DEBUG=INFO  # 打印NCCL信息
# export NCCL_P2P_DISABLE=0

pretrained_model=${root_path}/pretrained_models/chinese-alpaca-2-13b
pretrained_model=${root_path}/pretrained_models/llama-2-tiny-testing

chinese_tokenizer_path=tokenizer_chinese_llama  # chinese-llama原生词表 5w

dataset_dir=${root_path}/pt_data/0111/merge_arrow_data  # 这里修改原代码，指定pre_tokenizer_inst生成的arrow文件目录
output_dir=${root_path}/saved_models/pt_test0111_1

mkdir -p ${output_dir}
output_log_file=${output_dir}/run.log

use_lora=False

if [ "$use_lora" = "True" ]; then
    full_finetuning=False
    lr=1e-4  # lora
else
    full_finetuning=True
    lr=5e-5 # full_finetuning pt
fi

BATCH_SIZE="
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
"

STRATEGY="
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_first_step True \
    --save_strategy epoch \
    --save_steps 200 \
    --save_total_limit 5 \
    --evaluation_strategy epoch \
    --eval_steps 100 \
"

lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"  # 不微调emb和llm_head就在下面命令中注释掉--modules_to_save
lora_dropout=0.05
LORA="
    --lora_rank 64 \
    --lora_alpha 128 \
    --trainable ${lora_trainable} \
    --lora_dropout 0.05 \
"
    # --modules_to_save ${modules_to_save}

# 以下在pre_tokenizer时已经指定，这里无需管，不起作用
validation_file=validation_file_name  # 在pre_tokenizer_inst时已经切分训练测试在arrow文件，这里无需管理
unused_params="
    --block_size 1024 \
    --validation_split_percentage 0.1 \
    --validation_file ${validation_file} \
    --preprocessing_num_workers 8 \
"

deepspeed_config_file=ds_zero2_no_offload.json
# deepspeed_config_file=ds_zero3_no_offload.json  # 使用zero3的话需要在保持的ckpt目录中运行例如python zero_to_fp32.py checkpoint-126 final_model.bin才能得到训练后的文件

report_to=none  # 不记录到wandb
#report_to=wandb  # 记录到wandb run_name即记录的名字

CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
run_clm_pt_with_peft2.py \
    --deepspeed ${deepspeed_config_file} \
    --full_finetuning ${full_finetuning} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    $BATCH_SIZE \
    --do_train \
    --do_eval \
    --seed 1234 \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --gradient_checkpointing \
    $STRATEGY \
    $unused_params \
    $LORA \
    --torch_dtype auto \
    --load_in_kbits 16 \
    --save_safetensors False \
    --ddp_timeout 30000 \
    --ddp_find_unused_parameters ${full_finetuning} \
    --run_name pt \
    --report_to ${report_to} \
    2>&1 | tee -a ${output_log_file}

    # --resume_from_checkpoint ${resume_checkpoint}
# nohup bash 02_run_pt.sh > pt0220.log 2>&1 &