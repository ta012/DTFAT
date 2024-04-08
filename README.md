
# DTF-AT



## Introduction  

<p align="center"><img src="complete_arch_v2.png" alt="Illustration of AST." width="1200"/></p>

Pytorch Implementation of **DTF-AT: Decoupled Time-Frequency Audio Transformer for Event Classification**

### Setting Up  
 Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd DTFAT/ 
conda env create -f dtfat.yml
conda activate dtfat
```

#### Data Preparation Audioset  
Since the AudioSet data is downloaded from YouTube directly, videos get deleted and the available dataset decreases in size over time. So you need to prepare the following files for the AudioSet copy available to you.

Prepare data files as mentioned in [AST](https://github.com/YuanGongND/ast.git)

#### Validation 
We have provided the best model. Please download the [model weight](https://drive.google.com/file/d/1U3Esc7Ftn-wyCAsg4bSAws_bGVhNAFAH/view?usp=sharing) and keep it in `DTFAT/pretrained_models/best_model/model`. 

You can validate the model performance on your AudioSet evaluation data as follows,
```
cd DTFAT/egs/audioset
bash eval_run.sh
```
This script create log file with date time stamp in the same directory(eg:1692289183.log). You can find the mAP in the end of the log file.




## Citing  
We are using the [AST](https://github.com/YuanGongND/ast) repo for model training and [timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)(do not install timm) for model implementation and ImageNet-1K pretrained weights.
```  
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```  
```  
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```  
  

