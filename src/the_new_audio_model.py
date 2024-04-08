#!/vol/research/fmodel_av/tony/anaconda3/envs/dtfat/bin/python

"""
We used the timm folder(no timm installation required) for network and imagenet pretrained weight
https://github.com/huggingface/pytorch-image-models/tree/main/timm

"""


################# to_remove ######################
import os
MNT_PATH = "absolute-path//pytorch_home/"
VOL_PATH = '/vol/research/fmodel_av/tony/pytorch_home/'


to_set_var = ["TORCH_HOME","HF_HOME","PIP_CACHE_DIR"]
SET_PATH=None
if os.path.isdir(MNT_PATH):
  SET_PATH = MNT_PATH
elif os.path.isdir(VOL_PATH):
  SET_PATH = VOL_PATH
if SET_PATH is not None:
  print(f"SET_PATH {SET_PATH}")
  for v in to_set_var:
      print(f"Setting {v} to {SET_PATH}")
      os.environ[v]=SET_PATH
else:
 print(f"Both {MNT_PATH} and {VOL_PATH} not present Using default")


#######################################




import logging
import pdb
import math
import random
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


from itertools import repeat
from typing import List




import torch

from audio_model_timm.models.maxxvit import MaxxVit,MaxxVitCfg,MaxxVitConvCfg,MaxxVitTransformerCfg,checkpoint_filter_fn
from audio_model_timm.models import load_pretrained_apr22
import copy


#######################################
def check_params_changed(w_b4,w_init):
  not_changed = []
  for k,v in w_init.items():
      if torch.equal(v,w_b4[k]):
          not_changed.append(k)
  skip_formats = ['.se_weight_t','.se_weight_f','.rel_pos.','head.','.num_batches_tracked','.t_f_weight']

  remove_skip_formats = []
  
  for i in not_changed:
      for skp in skip_formats:
          if skp in i:
              remove_skip_formats.append(i)

  after_skip_formats = [i for i in not_changed if i not in remove_skip_formats]


  print(f"\n Params not init after after_skip_formats {after_skip_formats}")
  assert len(after_skip_formats)==0, 'some params not inint'
#######################################


def get_timm_pretrained_model(n_classes, imgnet=True):



    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def load_pretrained_maxvit_small(m2, n_classes):


        pretrained_cfg = {'url': '', 'hf_hub_id': 'timm/maxvit_small_tf_384.in1k', 'architecture': 'maxvit_small_tf_224', 'tag': 'in1k', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.95, 'crop_mode': 'center', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'num_classes': 1000, 'pool_size': (7, 7), 'first_conv': 'stem.conv1', 'classifier': 'head.fc'}
        pretrained_strict=False ## realative poskeyas missing

        load_pretrained_apr22(
            m2,
            pretrained_cfg=pretrained_cfg,
            num_classes=n_classes,
            in_chans=3,
            filter_fn=checkpoint_filter_fn, ### adapt between conv2d / linear layers attention2d and attencl
            strict=pretrained_strict,
        )

        return m2

    img_size = (1024,128)
    # window_size = (16,4)
    in_channel = 1
    window_size_l = [(8,8),(8,8),(8,8),(8,4)] ## used for window and grid attn.
    n_classes = n_classes
    use_nchw = False ## model was pretrained with use_nchw=False and set as defualt but same effect for 2d conv
    window_size_time_freq = [[(6,3),(3,6)],[(6,3),(3,6)],[(6,3),(3,6)],[(4,3),(3,4)]]## [time,freq]
    #window_size_time_freq = [[(64,4),(16,32)],[(64,4),(16,16)],[(64,4),(16,8)],[(32,4),(16,4)]]## [time,freq]
    feat_map_size_list = [(512, 64),(256, 32),(128, 16),(64, 8)] ## not used in network building


    drop_path_rate=0.3

    m2 = MaxxVit(cfg=MaxxVitCfg(embed_dim=(96, 192, 384, 768), depths=(2, 2, 5, 2), block_type=('M', 'M', 'M', 'M'), stem_width=64, stem_bias=True, conv_cfg=MaxxVitConvCfg(block_type='mbconv', expand_ratio=4.0, expand_output=True, kernel_size=3, group_size=1, pre_norm_act=False, output_bias=True, stride_mode='dw', pool_type='avg2', downsample_pool_type='avg2', padding='same', attn_early=False, attn_layer='se', attn_act_layer='silu', attn_ratio=0.25, init_values=1e-06, act_layer='gelu_tanh', norm_layer='batchnorm2d', norm_layer_cl='', norm_eps=0.001), transformer_cfg=MaxxVitTransformerCfg(dim_head=32, head_first=False, expand_ratio=4.0, expand_first=True, shortcut_bias=True, attn_bias=True, attn_drop=0.0, proj_drop=0.0, pool_type='avg2', rel_pos_type='bias_tf', rel_pos_dim=512, partition_ratio=32, window_size=None, grid_size=None, no_block_attn=False, use_nchw_attn=use_nchw, init_values=None, act_layer='gelu_tanh', norm_layer='layernorm2d', norm_layer_cl='layernorm', norm_eps=1e-05), head_hidden_size=768, weight_init='vit_eff'),num_classes= n_classes, in_chans= in_channel, img_size=img_size,window_size_list=window_size_l,drop_path_rate=drop_path_rate,window_size_time_freq=window_size_time_freq,feat_map_size_list=feat_map_size_list)

    # # torch.save(m2.state_dict(),'maxvit_small_tf_224_in1k.pth')
    # # print(m2)
    m2=m2.train(False)
    device = 'cuda:0'
    # wb4= list(m2.parameters())[0][2]
    if imgnet:
        print(" Loading ImgNet Pretrained weight ")
        w_b4 = copy.deepcopy(m2.state_dict())
        m2 = load_pretrained_maxvit_small(m2,n_classes)
        check_params_changed(w_b4=w_b4,w_init=m2.state_dict())
    
                
    m2=m2.train(True)

    # assert torch.equal(list(m2.parameters())[0][2],wb4), "weight same hence not loaded"
    x = torch.randn(1,in_channel,img_size[0],img_size[1])
    out = m2(x)

    print("timm forward_features out ", out.shape)

    print("count_parameters ",count_parameters(m2)/1e6)


    return m2


if __name__=="__main__":
    m = get_timm_pretrained_model(n_classes=527, imgnet=True)

    import time

    ts = 0
    rep = 10
    device = 'cuda:0'
    BS=2

    m = m.to(device)


    x = torch.randn(BS,1,1024,128).to(device)
    out = m(x)



