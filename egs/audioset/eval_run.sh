#!/bin/bash

dataset=audioset



imagenetpretrain=True

bal=bal 
lr=5e-4
epoch=11
main_dir="DTFAT"
as_ckpt="../../pretrained_models/audioset_fullset/best_model/models/best_audio_model.pth"
debug_mode=False

working_dir=absolute-path/${main_dir}/egs/audioset/
py_file=absolute-path/${main_dir}/src/eval.py
python_path=absolute-path/dtfat/bin/python
log_dir=absolute-path/tensorboard_logs/
exp_name=${main_dir}


batch_size=64
n_workers=8
tr_data=not_needed
te_data=absolute-path/audioset/eval_data.json

CUDA_CACHE_DISABLE=1 ${python_path} -u ${py_file} --dataset ${dataset} --data-train ${tr_data} --data-val ${te_data} --working_dir ${working_dir} --label-csv ${working_dir}/data/class_labels_indices.csv --lr $lr --n-epochs ${epoch} --batch-size $batch_size --num-workers ${n_workers} --save_model True --bal ${bal} --imagenet_pretrain $imagenetpretrain --debug ${debug_mode} --log_dir ${log_dir} --exp_name ${exp_name} --as2m_ckpt ${as_ckpt}  > ${working_dir}/$(date +%s).log   






