# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

############### Setting Path for to replace cache ########################
import os
MNT_PATH = "absolute-path//pytorch_home/"
VOL_PATH = 'absolute-path/pytorch_home/'


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

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
# import models
# from models import audio_model_timm
from the_new_audio_model import get_timm_pretrained_model
import numpy as np
import json 

from datetime import datetime
import time 

######## Reproducibility ###########
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  

seed = 56
import random
random.seed(seed)

np.random.seed(seed)

torch.use_deterministic_algorithms(True)

 # You can choose any integer value as the seed

torch.manual_seed(seed)  # Set the seed for generating random numbers
torch.cuda.manual_seed(seed)  # Set the seed for generating random numbers on GPU (if available)
torch.cuda.manual_seed_all(seed)  # Set the seed for generating random numbers on all GPUs (if available)
torch.backends.cudnn.deterministic = True  # Ensure reproducibility by disabling some GPU-specific optimizations
torch.backends.cudnn.benchmark = False  # Disable benchmarking to improve reproducibility

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

###################################

timestamp = str(datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')).replace('-','_').replace(':','_').replace(' ','_')


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
# parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--working_dir", type=str, default='', help="directory to dump experiments")
parser.add_argument("--exp_dir", type=str, default='', help="Keep it empty")


parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=-1, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")

parser.add_argument('--ensemble',help='if ensemble', type=ast.literal_eval, default='False')


parser.add_argument('--debug',help='if debug', type=ast.literal_eval, default='False')

parser.add_argument("--log_dir", type=str, default='', help="log_dir")
parser.add_argument("--exp_name", type=str, default='', help="experiment name")

parser.add_argument("--as2m_ckpt", type=str, default='', help="as2m_ckpt")



args = parser.parse_args()

args.resume_ckpt = None 


from utilities import *

def validate_best(audio_model, val_loader, args, epoch):
    """
    Same as validate just printing during validation and added args.loss_fn = to avoid error validation loop
    """
    args.loss_fn = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    len_val_loader = len(val_loader)
    print("\nVal loader size ",len_val_loader)
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            print(f"{i}/{len_val_loader}")

            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)        
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        audio_input = audio_input.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        target = target.astype(int)

        stats = calculate_stats(audio_output, target)



    return stats, loss

if __name__ == '__main__':


  # dataset spectrogram mean and std, used to normalize the input
  norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
  target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
  # if add noise for data augmentation, only use for speech commands
  noise = {'audioset': False, 'esc50': False, 'speechcommands':True}
  freqm_dict = {'audioset':48}
  timem_dict = {'audioset':192}
  mixup_dict = {'audioset':0.5}
  num_classes_dict = {'audioset':527}

  audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': freqm_dict[args.dataset], 'timem': timem_dict[args.dataset], 'mixup': mixup_dict[args.dataset], 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                'noise':noise[args.dataset]}
  val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False}


  val_ds = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)
  val_loader = torch.utils.data.DataLoader(val_ds,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g)
  

  
  print(f"\nval dataset size {len(val_loader.dataset)} \n")

      


  audio_model = get_timm_pretrained_model(num_classes_dict[args.dataset],imgnet=args.imagenet_pretrain)



  ### load the AudioSet full set ckpt
  print(f"\nLoading {args.as2m_ckpt}")    
  wt = torch.load(args.as2m_ckpt,map_location='cpu')
  new_state = dict()
  for k,v in wt.items():
      new_k = k.replace('module.','')
      new_state[new_k]=v
  audio_model.load_state_dict(new_state)
  print('\n Now starting validating with best model ')

  stats,_ = validate_best(audio_model, val_loader, args, 'best_model')


  mAP = np.mean([stat['AP'] for stat in stats])

  print(f"Val mAP: {mAP}")
  print("Validation Completed ")

