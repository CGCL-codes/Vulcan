import warnings
warnings.filterwarnings("ignore")

import os
import GPUtil
def select_gpu_with_most_free_memory():
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPU found!")
    gpu_id = max(gpus, key=lambda gpu: gpu.memoryFree).id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Selected GPU: {gpu_id}, Free Memory: {gpus[gpu_id].memoryFree} MB")
select_gpu_with_most_free_memory()

import torch
import argparse
from dataset.utils import *
from model.utils import *
from utils.utils import *
from engine.eval import *
from engine.train import *
# edgeseed
from method.cluster import *
from method.collapse_loss import *
from method.pruning import *
from method.erank import *
from method.utils import *

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="EdgeSeed", help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_name", type=str, default="deit_base_patch16_224", help="Model to use.")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset to use.")
    parser.add_argument("--task_name", type=str, default="T1-25", help="Sub-task of edge devices.")
    parser.add_argument("--task_type", type=str, default="recognition", help="Sub-task type of edge devices.")
    parser.add_argument("--pruning_rate", default=0.2, type=float,help="Pruning Rate.")
    parser.add_argument("--output_dir", default="/workspace/project/edgeseed/data", type=str, help="The output directory where checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Training Arguments
    parser.add_argument("--train_batch_size", default=256, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_epochs", default=5, type=int, help="Total number of training iteration to perform.")
    parser.add_argument("--num_steps", default=-1, type=int, help="Total number of training iteration to perform.")
    parser.add_argument("--eval_every", default=100, type=int, help="Run prediction on validation set every so many steps.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--penalty_param", default=1.0, type=float, help="The initial penalty parameter for SGD.")
    parser.add_argument("--weight_decay", default=5e-2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--accum_steps", default=1, type=float, help="Number of updates steps to accumulate before performing a backward/update pass.")

    return parser

def get_args():
    parser = get_args_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.local_rank = -1
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    name2abb = {
        "deit_base_patch16_224": "deit_base",
        "deit_small_patch16_224": "deit_small",
        "deit_tiny_patch16_224": "deit_tiny",
        "mask_rcnn_swin_tiny": "swin_tiny",
    }
    args.name = f"{name2abb[args.model_name]}({args.dataset_name}-{args.task_name}-{args.pruning_rate})"
    args.num_steps = rate2step(args.pruning_rate)*args.accum_steps
    args.eval_epoch = True
    args.sub_label = get_sub_task(args)
    return args

def wandb_init(args):
    wandb.init(project='edgeseed', name=args.name)
    config = wandb.config
    config.batch_size = args.train_batch_size
    config.test_batch_size = args.eval_batch_size
    config.epochs = args.num_epochs
    config.lr = args.learning_rate
    config.use_cuda = True  
    config.seed = args.seed  
    config.log_interval = 10
    wandb.watch_called = False 

def vulcan():
    args = get_args()
    # get model and dataset
    model = get_model(
        args.model_name,args.dataset_name,
        root=os.path.join(args.output_dir,"param"),
        task_type = args.task_type
    )
    wandb_init(args)
    model = class_specific_model_derivation(args, model = model)

if __name__ == '__main__':
    vulcan()
    
    
