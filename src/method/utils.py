from peft import get_peft_model, LoraConfig, TaskType
from types import MethodType
from peft.tuners.lora import layer

from .cluster import *
from .collapse_loss import *
from .erank import *
from .pruning import *

from dataset.utils import get_dataloader
from utils.utils import get_sub_task,get_lagrange_multiplier
from engine.train import train_vulcan
from engine.utils import load_model,load_clusters

def class_specific_model_derivation(
        args,model,mode = "weight",
        use_collapse_loss=True,use_rank_loss=True,
        early_stop=False,lora=False,anchor=True,
        norm_type="z_score"
    ):
    train_loader, test_loader = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = get_sub_task(args),
        task_type = args.task_type,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
    )
    clusters = get_clusters(args,model,train_loader,norm_type)
    qk_dim_list,vo_dim_list = mha_adaptive_arch_config(args,model)
    if lora:
        def forward(self,x):
            return self.base_model(x)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=256,            
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["qkv", "proj", "fc1", "fc2", "head"],
            bias="all",
        )
        model = get_peft_model(model, lora_config)
        model.forward = MethodType(forward, model)
    # get lagrange multiplier
    get_lagrange_multiplier(model)
    # post-training
    train_vulcan(
        args=args, model=model,
        train_loader=train_loader, test_loader=test_loader,
        use_collapse_loss=use_collapse_loss, use_rank_loss=use_rank_loss, 
        clusters=clusters, qk_dim_list=qk_dim_list, vo_dim_list=vo_dim_list,
        mode = mode, eval_epoch = args.eval_epoch, early_stop=early_stop,
        lora = lora,anchor=anchor,
    )
    # pruning
    if lora:
        model = model.merge_and_unload()
    load_model(args,model)
    clusters = load_clusters(args)
    model = pruning_vit(model,clusters,qk_dim_list,vo_dim_list)
    return model
    
