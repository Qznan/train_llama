"""
对ckpt进行算术操作如加减乘除
"""
import os
import torch
import shutil


def ari_tensors(source_ckpt_path, target_ckpt_path, output_ckpt_path, ops='-'):
    # 创建输出文件夹
    # if os.path.exists(output_ckpt_path):
    #     i = ''
    #     while i not in ['y', 'n']:
    #         i = input(f'{output_ckpt_path} 已经存在 必须先删除，是否删除。y删除|n不删除并退出:' )
    #     if i == 'n':
    #         exit(0)
    #     elif i == 'y':
    #         shutil.rmtree(output_ckpt_path)
    #     else:
    #         pass

    # # 复制源ckpt文件夹的所有内容到输出文件夹
    # shutil.copytree(source_ckpt_path, output_ckpt_path)

    # 不这样就手动将所有源文件包括index和tokenizer啥的都复制到output文件夹

    # 加载目标ckpt文件夹中的所有bin文件
    target_tensors = {}
    for f in os.listdir(target_ckpt_path):
        if f.endswith('.bin') and f.startswith('pytorch_model'):
            target_tensors.update(torch.load(os.path.join(target_ckpt_path, f)))
    print(f'load target_tensor from {target_ckpt_path} success! total have {len(target_tensors)} params')

    # output_bin_files = [f for f in os.listdir(output_ckpt_path) if f.endswith('.bin') and f.startswith('pytorch_model')]
    # for file in output_bin_files:
    if not os.path.exists(output_ckpt_path):
        os.makedirs(output_ckpt_path, exist_ok=True)

    source_bin_files = [f for f in os.listdir(source_ckpt_path) if f.endswith('.bin') and f.startswith('pytorch_model')]
    for file in source_bin_files:
        print(f'loaded source tensors from {os.path.join(source_ckpt_path, file)}')
        source_tensors = torch.load(os.path.join(source_ckpt_path, file))
        for k, v in source_tensors.items():
            # ('model.layers.14.input_layernorm.weight', tensor([0.3516, 0.3789, 0.3574,  ..., 0.3652, 0.3496, 0.3438], dtype=torch.bfloat16))
            if k in target_tensors:
                s_shape, t_shape = source_tensors[k].shape, target_tensors[k].shape
                # lm_head.weight [98830, 5120]
                # model.embed_tokens.weight [98830, 5120]
                if s_shape != t_shape:
                    print('params: ', k)
                    print('source: ', s_shape)
                    print('target: ', t_shape)
                    print('检查到参数大小不一致，自动修复lm_head.weight model.embed_tokens.weight')
                    if k in ['lm_head.weight', 'model.embed_tokens.weight']:
                        if s_shape[0] < t_shape[0]:
                            source_tensors[k] = torch.cat([source_tensors[k], torch.zeros(t_shape[0] - s_shape[0], s_shape[1])])
                        elif s_shape[0] > t_shape[0]:
                            target_tensors[k] = torch.cat([target_tensors[k], torch.zeros(s_shape[0] - t_shape[0], s_shape[1])])
                        if source_tensors[k].shape != target_tensors[k].shape:  # 还是不一样
                            print('修复不成功，这里先不改变直接用source参数, 但请重新修改代码以重跑')
                            continue
                        else:
                            print('修复成功')
                ori_dtype = v.dtype
                if ops == '-':
                    source_tensors[k] = (v.float() - target_tensors[k].float()).to(ori_dtype)
                elif ops == '+':
                    source_tensors[k] = (v.float() + target_tensors[k].float()).to(ori_dtype)
                else:
                    raise NotImplementedError
                print(f'param_name:  {k} {v.shape} processing ok')

        torch.save(source_tensors, os.path.join(output_ckpt_path, file))
        print(f'file:  {file} total params processing ok')


if __name__ == "__main__":
    # ari_tensors(  # 测试 金融效果差的fullsft 减去通用base
    #     '/data_net/med_group/LamberGPT/experiments/outputs/uni3/llama-2-13b-uni_3_sft_wo_warmup_full_epoch2_v3',
    #     '/home/hgd/fin_group/zyn/data3/fin_group/saved_models/fin_sft_0605_v22/checkpoint-8392',
    #     '/home/hgd/fin_group/zyn/data3/fin_group/saved_models/fin_sft_0605_v22_test_base_minus_this',
    # )

    # ari_tensors(  # 医疗减通用base
    #     '/data_net/med_group/LamberGPT/experiments/outputs/uni3/llama-2-13b-uni_3_sft_wo_warmup_full_epoch2_v3',
    #     '/home/hgd/fin_group/med/llama-2-13b-med_sft_short_ehr_lora_epoch2_v3-12_merged',
    #     '/home/hgd/fin_group/med/llama-2-13b-med_sft_short_ehr_lora_epoch2_v3-12_merged_minus_uni3base',
    # )

    ari_tensors(  # 金融+医疗减base
        '/home/hgd/fin_group/zyn/data3/fin_group/saved_models/fin_sft_0605_v22_lora/checkpoint-8392',
        '/home/hgd/fin_group/med/llama-2-13b-med_sft_short_ehr_lora_epoch2_v3-12_merged_minus_uni3base',
        '/home/hgd/fin_group/zyn/data3/fin_group/saved_models/fin_sft_0605_v22_lora_add_medlorav312',
        ops='+'
    )


