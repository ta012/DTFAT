
# DTF-AT



## Introduction  

<p align="center"><img src="complete_arch_v2.png" alt="Illustration of AST." width="1200"/></p>

PyTorch Implementation of [DTF-AT: Decoupled Time-Frequency Audio Transformer for Event Classification (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29716)

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




## Acknowledgements
We are using the [AST](https://github.com/YuanGongND/ast) repo for model training and [timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm)(do not install timm) for model implementation and ImageNet-1K pretrained weights.


## Citation

If you find our work useful, please cite it as:  

```bibtex
@inproceedings{alex2024dtf,
  title={DTF-AT: decoupled time-frequency audio transformer for event classification},
  author={Alex, Tony and Ahmed, Sara and Mustafa, Armin and Awais, Muhammad and Jackson, Philip JB},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17647--17655},
  year={2024}
}