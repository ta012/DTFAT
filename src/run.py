# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

#######################################
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
from traintest import train, validate
import json 

from datetime import datetime
import time 

######## Reproducibility ###########
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  

seed = 56
print(f"Using seed {seed}")
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



args = parser.parse_args()

args.resume_ckpt = None 


# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
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

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_ds = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker,generator=g)
    else:

        print('balanced sampler is not used')
        train_ds = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker,generator=g)

    val_ds = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,worker_init_fn=seed_worker,generator=g)
    

    
    if args.debug:
        print("\n\n Running in DEBUG Mode \n")
        ds = dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
        ds = torch.utils.data.Subset(ds,np.linspace(1, len(ds)-1, num=500,dtype=int))
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = train_loader
        print(f"\ntrain dataset size {len(train_loader.dataset)} val dataset size {len(val_loader.dataset)} \n")
    else:
        print(f"\ntrain dataset size {len(train_loader.dataset)} val dataset size {len(val_loader.dataset)} \n")

        
    ## override args for as2m
    if len(train_loader.dataset) > 2e5:
        args.ensemble = True


    audio_model = get_timm_pretrained_model(num_classes_dict[args.dataset],imgnet=args.imagenet_pretrain)


if args.debug:

    args.exp_dir = args.working_dir + 'to_del'
    print(f"\nCreating experiment directory: {args.exp_dir}")

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
        os.mkdir(args.exp_dir+'/models')
    else:
        print(f"{args.exp_dir} already exists")  

else:
    args.exp_dir = args.working_dir + timestamp
    print(f"\nCreating experiment directory: {args.exp_dir}")

    os.mkdir(args.exp_dir)
    os.mkdir(args.exp_dir+'/models')


args.timestamp = timestamp


with open(args.exp_dir+"/args.pkl", "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))

train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

