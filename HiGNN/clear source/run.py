import argparse
import os
import time

import torch
from dataset import load_dataset_random
from model import HiGNN
from trainer import Trainer

class Option(object):
    def __init__(self, d):
        self.__dict__ = d

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="trimnet_drug/data/", help="all data dir"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bace",
        help="muv,tox21,toxcast,sider,clintox,hiv,bace,bbbp",
    )
    parser.add_argument("--seed", default=68, type=int)
    parser.add_argument("--gpu", type=int, nargs="+", default=0, help="CUDA device ids")  # noqa: E501

    parser.add_argument(
        "--hid", type=int, default=32, help="hidden size of transformer model"
    )
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument(
        "--batch_size", type=int, default=128, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_patience", default=10, type=int)
    parser.add_argument("--early_stop_patience", default=-1, type=int)
    parser.add_argument("--lr_decay", default=0.98, type=float)
    parser.add_argument("--focalloss", default=False, action="store_true")

    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--exps_dir", default="../test", type=str,
                        help="out dir")
    parser.add_argument("--exp_name", default=None, type=str)

    d = vars(parser.parse_args())
    args = Option(d)

    args.parallel = True if args.gpu and len(args.gpu) > 1 else False
    args.parallel_devices = args.gpu
    args.tag = time.strftime("%m-%d-%H-%M") if args.exp_name is None else args.exp_name  # noqa: E501
    args.exp_path = os.path.join(args.exps_dir, args.tag)

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    args.code_file_path = os.path.abspath(__file__)

    if args.dataset == "hiv":
        args.tasks = ["HIV_active"]
        train_dataset, valid_dataset, test_dataset = load_dataset_random(
            args.data, args.dataset, args.seed, args.tasks
        )
        args.out_dim = 1
    elif args.dataset == "bace":
        args.tasks = ["Class"]
        train_dataset, valid_dataset, test_dataset = load_dataset_random(
            args.data, args.dataset, args.seed, args.tasks
        )
        args.out_dim = 1
    elif args.dataset == "bbbp":
        args.tasks = ["BBBP"]
        train_dataset, valid_dataset, test_dataset = load_dataset_random(
            args.data, args.dataset, args.seed, args.tasks
        )
        args.out_dim = 1
    else:  # Unknown dataset error
        raise Exception("Unknown dataset, please enter the correct --dataset option")  # noqa: E501

    args.in_dim = train_dataset.num_node_features
    args.edge_in_dim = train_dataset.num_edge_features
    option = args.__dict__

    model = HiGNN(in_channels=45,
                  hidden_channels=128,
                  out_channels=1,
                  edge_dim=10,
                  num_layers=2,
                  dropout=0,
                  slices=2,
                  f_att=True,
                  r=4,
                  brics=False,)
    
    trainer = Trainer(
        option,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    train()
