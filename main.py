import os
import random
import sys
import socket
import torch.multiprocessing

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '16')

torch.multiprocessing.set_sharing_strategy('file_system')
import warnings

warnings.filterwarnings("ignore")

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train, get_client_config
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle

import torch
import uuid
import datetime
import numpy as np


def set_random(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(debug_command=None):
    parser = ArgumentParser(description='Parameter Setting', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=50, help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch', type=int, default=1, help='The Local Epoch for each Participant')
    parser.add_argument('--num_clients', type=int, default=0,
                        help='Total number of clients (0 = use the largest preset in dataset_config). '
                             '>0 hits a preset directly, otherwise scaled proportionally.')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    parser.add_argument('--model', type=str, default='fipl_da',
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, default='fl_pacs',
                        choices=DATASET_NAMES, help='Dataset choice.')
    parser.add_argument('--backbone', type=str, default='lightresnet',
                        choices=['lightresnet'],
                        help='Backbone network (default: lightresnet).')

    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')
    parser.add_argument('--averaing', type=str, default='weight', help='The Option for averaging strategy')

    # Differential Privacy
    parser.add_argument('--eps', type=float, default=0.0,
                        help='Privacy budget eps for DP-SGD. If <=0, no DP is applied.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient / feature clipping norm C for DP-SGD.')
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='Privacy parameter delta for DP (default: 1e-4).')
    parser.add_argument('--proto_eps_ratio', type=float, default=0.0,
                        help='Ratio of eps allocated to prototypes (0.0~1.0). '
                             '0.0 = unified-sigma mode (backward compatible). '
                             '>0 = explicit split: eps_proto = eps * ratio, eps_model = eps * (1-ratio).')

    # FIPL-DA
    parser.add_argument('--mu', type=float, default=None,
                        help='FIPL-DA prototype loss weight.')
    parser.add_argument('--mu_decay', type=float, default=0.9,
                        help='Decay factor for mu per round (mu *= mu_decay).')
    parser.add_argument('--scale', type=float, default=10.0,
                        help='FIPL-DA cosine classifier logits scaling (softmax temperature).')
    parser.add_argument('--etf_lambda', type=float, default=1.0,
                        help='FIPL-DA server prototype separation regularization strength: balance between '
                             'fitting client prototypes and separating too-close class pairs. 0 = plain FedAvg.')
    parser.add_argument('--proto_margin', type=float, default=-0.1,
                        help='FIPL-DA prototype separation margin: only penalize class pairs with cosine > margin.')
    parser.add_argument('--etf_lr', type=float, default=0.1,
                        help='FIPL-DA GD step size for server prototype separation (decays by etf_lr_decay each round).')
    parser.add_argument('--etf_lr_decay', type=float, default=1.0,
                        help='FIPL-DA decay factor for etf_lr per round.')
    parser.add_argument('--etf_init', action='store_true', dest='etf_init', default=True,
                        help='FIPL-DA: initialize classifier weights with a simplex ETF (default on).')
    parser.add_argument('--no_etf_init', action='store_false', dest='etf_init', default=True,
                        help='FIPL-DA: disable ETF initialization.')
    parser.add_argument('--local_lr', type=float, default=None,
                        help='Local training learning rate.')
    parser.add_argument('--local_batch_size', type=int, default=None,
                        help='Local training batch size.')

    torch.set_num_threads(4)
    add_management_args(parser)
    if debug_command is not None:
        args = parser.parse_args(debug_command.split())
    else:
        args = parser.parse_args()

    best = best_args[args.dataset][args.model]

    if debug_command is not None:
        cmd_tokens = debug_command.split()
    else:
        cmd_tokens = sys.argv[1:]

    explicit_keys = set()
    for token in cmd_tokens:
        if token.startswith('--'):
            key = token[2:].replace('-', '_')
            explicit_keys.add(key)

    for key, value in best.items():
        if key not in explicit_keys:
            setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    if args is None:
        args = parse_args()
    set_random(args.seed)
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)

    selected_domain_list, actual_parti_num = get_client_config(args)
    args.parti_num = actual_parti_num

    backbone_names = [args.backbone] * args.parti_num
    backbones_list = priv_dataset.get_backbone(args.parti_num, backbone_names)

    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.arch_name

    total_params = sum(p.numel() for p in model._model_template.parameters())
    trainable_params = sum(p.numel() for p in model._model_template.parameters() if p.requires_grad)
    print(f'Network: {model.arch_name} | Total params: {total_params:,} | Trainable params: {trainable_params:,}')

    print('\n{}_eps{}_{}_{}_{}_{}'.format(args.model, args.eps, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))
    setproctitle.setproctitle('{}_eps{}_{}_{}_{}_{}'.format(args.model, args.eps, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train(model, priv_dataset, args, selected_domain_list)
