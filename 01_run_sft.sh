# root_path=/home/yss
# root_path=/disk0/fin_group/zyn
root_path=.

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"  # 不微调emb和llm_head就在下面命令中注释掉--modules_to_save
lora_dropout=0.05

pretrained_model=${root_path}/pretrained_models/chinese-alpaca-2-13b
pretrained_model=/new_disk/models_for_all/chinese-alpaca-2-13b

chinese_tokenizer_path=tokenizer_chinese_llama  # chinese-llama原生词表 5w
# chinese_tokenizer_path=tokenizer_lamber  # lamberGPT扩增词表 10w  注意要跟pre_tokenizer_inst使用的词表保持一致

dataset_dir=${root_path}/instr_data/1219/arrow_data1219  # 这里修改原代码，指定pre_tokenizer_inst生成的arrow文件目录
# dataset_dir=/home/yss/data_fin/ins_1219_lamber/arrow_data1219_lamber  # 这里修改原代码，指定pre_tokenizer_inst生成的arrow文件目录

validation_file=validation_file_name  # 在pre_tokenizer_inst时已经切分训练测试在arrow文件，这里无需管理

per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=1024  # 在pre_tokenizer_inst时已经指定，这里无需管，不起作用

output_dir=${root_path}/saved_models/1218_3gpu_zero2_test

deepspeed_config_file=ds_zero2_no_offload.json
# deepspeed_config_file=ds_zero3_no_offload.json  # 使用zero3的话需要在保持的ckpt目录中运行例如python zero_to_fp32.py checkpoint-126 final_model.bin才能得到训练后的文件

# run_name=tmp  #wandb的run名字
report_to=none  # 不记录到wandb

CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --standalone --nnodes 1 --nproc-per-node 3 \
run_clm_sft_with_peft2.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 1234 \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --evaluation_strategy epoch \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --report_to ${report_to}

    # --modules_to_save ${modules_to_save} \