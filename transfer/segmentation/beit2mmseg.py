# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_beit(ckpt):
    new_ckpt = OrderedDict()

    if "rel_pos_bias.relative_position_bias_table" in ckpt:
        print("Expand the shared relative position embedding to each transformer block. ")
        last_layer = -1
        for k in ckpt.keys():
            if k.startswith('blocks'):
                layer_id = int(k.split('.')[1])
                last_layer = max(layer_id, last_layer)
        num_layers = last_layer + 1
        rel_pos_bias = ckpt["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            ckpt["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        
        ckpt.pop("rel_pos_bias.relative_position_bias_table")

    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            k = k.replace('encoder.', '')
        if k.startswith('blocks'):
            new_key = k.replace('blocks', 'layers')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            elif 'mlp.fc1' in new_key:
                new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_key:
                new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
            elif new_key.endswith('gamma'):
                gamma_id = int(new_key[new_key.find('ls') + 2])
                new_key = new_key.replace(f'ls{gamma_id}.gamma', f'gamma_{gamma_id}')
            new_ckpt[new_key] = v
        elif k.startswith('patch_embed'):
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
            new_ckpt[new_key] = v
        elif k in ['pos_embed', 'cls_token'] or k.startswith('norm'):
            new_key = k
            new_ckpt[new_key] = v
        else:
            print(f"Unused key: {k}")
            continue
    
    all_keys = list(new_ckpt.keys())
    for k in all_keys:
        if 'relative_position_index' in k:
            new_ckpt.pop(k)
    
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained beit models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    weight = convert_beit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()