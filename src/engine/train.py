import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import copy
import time
from mmengine.optim import OptimWrapper

from .eval import evaluate
from .utils import *
from method.collapse_loss import *
from method.pruning import *

def train_vulcan(
        args,model,
        train_loader,test_loader,
        use_collapse_loss=False,
        use_rank_loss=False,
        clusters=None,
        qk_dim_list=None,
        vo_dim_list=None,
        mode = "weight",
        eval_epoch = True,
        early_stop = False,
        lora = False,
        anchor = True,
    ):

    # optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "lambda" not in n and p.requires_grad], 
            "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": [model.lambda_1,model.lambda_2], 
            "lr": args.penalty_param*(-1), "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False
    )

    # get config
    if hasattr(model, "blocks"):
        if not lora:
            depth = len(model.blocks)
            # register hook
            activation = [None for _ in range(depth)]
            def register_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output [b,N,e]
                    if mode == "weight":
                        activation[layer_idx] = (output.sum(dim=(0,1))).detach().cpu()
                    elif mode == "activation":
                        activation[layer_idx] = (output.mean(dim=(0,1)))
                return hook_fn
            hooks = []
            for layer_idx in range(depth):
                hooks.append(model.blocks[layer_idx].mlp.act.register_forward_hook(register_hook(layer_idx)))
        else:
            depth = len(model.base_model.model.blocks)
            # register hook
            activation = [None for _ in range(depth)]
            def register_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output [b,N,e]
                    if mode == "weight":
                        activation[layer_idx] = (output.sum(dim=(0,1))).detach().cpu()
                    elif mode == "activation":
                        activation[layer_idx] = (output.mean(dim=(0,1)))
                return hook_fn
            hooks = []
            for layer_idx in range(depth):
                hooks.append(
                    model.base_model.model.blocks[layer_idx].mlp.act.register_forward_hook(register_hook(layer_idx))
                )

    elif hasattr(model, "backbone"):
        activation = [
            [torch.zeros(block.ffn.layers[1].in_features) for block in stage.blocks] 
            for stage in model.backbone.stages
        ]
        def register_hook(stage_idx,block_idx):
            def hook_fn(module, input, output):
                # output [b,N,e]
                activation[stage_idx][block_idx] = (output.sum(dim=(0,1))).detach().cpu()
            return hook_fn
        hooks = []
        for stage_idx in range(len(model.backbone.stages)):
            for block_idx in range(len(model.backbone.stages[stage_idx].blocks)):
                hooks.append(
                    model.backbone.stages[stage_idx].blocks[block_idx].ffn.layers[0][1].register_forward_hook(
                        register_hook(stage_idx,block_idx)
                    )
                )

    # prepare training
    criterion = nn.CrossEntropyLoss()
    device = args.device
    model.to(device)
    model.zero_grad()
    global_setp = 0
    can_save = False
    if args.num_steps>0:
        num_steps = args.num_steps
    else:
        num_steps = args.num_epochs*len(train_loader)
    
    if not lora:
        acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    else:
        acc = evaluate(copy.deepcopy(model).merge_and_unload(),test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    
    wandb_log_acc(global_setp,acc)
    # acc_pruned
    if not lora:
        model_pruned = copy.deepcopy(model)
    else:
        model_pruned = copy.deepcopy(model).merge_and_unload()
    pruning_vit(model_pruned,clusters,qk_dim_list,vo_dim_list)
    acc_pruned = evaluate(model_pruned,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    wandb_log_acc(global_setp,acc_pruned,pruned=True)
    best_acc = acc_pruned

    accum_steps = args.accum_steps

    print("\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # start training
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader, 
            desc="Training (X / X Steps) (loss=X.X)", 
            bar_format="{l_bar}{r_bar}", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            # forward+backward
            if args.task_type == "recognition":
                xb, yb = batch[:2]
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss_ce = criterion(logits, yb)
            elif args.task_type == "detection":
                data = model.data_preprocessor(batch, True)
                losses = model._run_forward(data, mode='loss')
                loss_ce = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox']
            elif args.task_type == "segmentation":
                data = model.data_preprocessor(batch, True)
                losses = model._run_forward(data, mode='loss')
                loss_ce = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox'] + losses['loss_mask']
            '''=========== Calculate the loss ============='''
            update_clusters(activation,clusters)
            if use_collapse_loss:
                if mode == "weight":
                    loss_collapse = weight_collapse_loss(model,clusters,model.lambda_1,model.lambda_2,lora,anchor=anchor) 
                elif mode == "activation":
                    loss_collapse = activation_collapse_loss(model,clusters,activation,model.lambda_1,model.lambda_2)
            else:
                loss_collapse = torch.zeros(1).to(device)
            if use_rank_loss:
                loss_rank = rank_loss(model,qk_dim_list,vo_dim_list,model.lambda_1,model.lambda_2,lora) 
            else:
                loss_rank = torch.zeros(1).to(device)
            loss = torch.zeros(1).to(device)
            if torch.isfinite(loss_ce):
                loss+=loss_ce
            if torch.isfinite(loss_collapse):
                loss+=loss_collapse
            if torch.isfinite(loss_rank):
                loss+=loss_rank
            '''============================================'''
            loss.backward()
            if (global_setp+1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_setp += 1

            if isinstance(best_acc,dict):
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss_ce=%2.4f) (loss_collapse=%2.4f) (loss_rank=%2.4f) (bbox_mAP=%.1f) (segm_mAP=%.1f)" % (
                        global_setp, num_steps, loss_ce.item(), loss_collapse.item(), loss_rank.item(),best_acc["bbox_mAP"],best_acc["segm_mAP"]
                    )
                )
            else:
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss_ce=%2.4f) (loss_collapse=%2.4f) (loss_rank=%2.4f) (best_acc=%.2f)" % (
                        global_setp, num_steps, loss_ce.item(), loss_collapse.item(), loss_rank.item(),best_acc
                    )
                )
            # log the training process
            wandb_log(
                global_setp,loss=loss.item(),
                loss_ce=loss_ce.item(),
                loss_collapse=loss_collapse.item(),
                loss_rank=loss_rank.item()
            )
            wandb_log(
                global_setp,
                lambda1=model.lambda_1.item(),
                lambda2=model.lambda_2.item(),
            )

            # evaluate
            if global_setp % args.eval_every == 0:
                if not lora:
                    acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                else:
                    acc = evaluate(copy.deepcopy(model).merge_and_unload(),test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                wandb_log_acc(global_setp,acc)
                # acc_pruned
                if not lora:
                    model_pruned = copy.deepcopy(model)
                else:
                    model_pruned = copy.deepcopy(model).merge_and_unload()
                pruning_vit(model_pruned,clusters,qk_dim_list,vo_dim_list)
                acc_pruned = evaluate(model_pruned,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                wandb_log_acc(global_setp,acc_pruned,pruned=True)

                if not can_save and get_delta_acc(acc,acc_pruned) < 1.0:
                    can_save = True
                if can_save and better_acc(acc_pruned,best_acc):
                    best_acc = acc_pruned
                if not lora:
                    save_model(args,model)
                else:
                    save_model(args,copy.deepcopy(model).merge_and_unload())
                save_clusters(args,clusters)
            '''=============== Early Stop ==============='''
            if can_save and early_stop:
                num_steps = global_setp + 1000
                early_stop = False
            '''=========================================='''

            if global_setp >= num_steps: break
        # evaluate
        if eval_epoch:
            if not lora:
                acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
            else:
                acc = evaluate(copy.deepcopy(model).merge_and_unload(),test_loader,visual_task=args.task_type,sub_label=args.sub_label)
            wandb_log_acc(global_setp,acc)
            # acc_pruned
            if not lora:
                model_pruned = copy.deepcopy(model)
            else:
                model_pruned = copy.deepcopy(model).merge_and_unload()
            pruning_vit(model_pruned,clusters,qk_dim_list,vo_dim_list)
            acc_pruned = evaluate(model_pruned,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
            wandb_log_acc(global_setp,acc_pruned,pruned=True)
            if not can_save and get_delta_acc(acc,acc_pruned) < 1.0:
                can_save = True
            if can_save and better_acc(acc_pruned,best_acc):
                best_acc = acc_pruned
            if not lora:
                save_model(args,model)
            else:
                save_model(args,copy.deepcopy(model).merge_and_unload())
            save_clusters(args,clusters)
        if global_setp >= num_steps: break
    
    if isinstance(best_acc,dict):
        logger.info("Best bbox_mAP: \t%f" % best_acc["bbox_mAP"])
        logger.info("Best segm_mAP: \t%f" % best_acc["segm_mAP"])
    else:
        logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("=================================================================================\n")

    for hook in hooks:
        hook.remove()

    return best_acc

def train(
        args,model,
        train_loader,test_loader,
        model_name=None,
        early_stop = True,
    ):

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "lambda" not in n and p.requires_grad], 
            "lr": args.learning_rate, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False
    )

    # prepare training
    criterion = nn.CrossEntropyLoss()
    device = args.device
    model.to(device)
    model.zero_grad()
    global_setp = 0
    if args.num_steps>0:
        num_steps = args.num_steps
    else:
        num_steps = args.num_epochs*len(train_loader)
    
    best_acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    last_acc = best_acc
    cnt = 0
    wandb_log(global_setp,accuracy=best_acc)
    if model_name is None:
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}-FT).pt"
    print("\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # start training
    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader, 
            desc="Training (X / X Steps) (loss=X.X)", 
            bar_format="{l_bar}{r_bar}", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            # forward+backward
            if args.task_type == "recognition":
                xb, yb = batch[:2]
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
            elif args.task_type == "detection":
                data = model.data_preprocessor(batch, True)
                losses = model._run_forward(data, mode='loss')
                loss = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_setp += 1
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.4f) (best_acc=%.2f)" % (
                    global_setp, num_steps, loss.item(), best_acc
                )
            )
            # log the training process
            wandb_log(
                global_setp,loss=loss.item(),
            )
            # evaluate
            if global_setp % args.eval_every == 0:
                acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                wandb_log(global_setp,accuracy=acc)
                if acc > best_acc:
                    best_acc = acc
                    save_model(args,model,model_name)
                '''============ Early Stop ============='''
                if abs(acc-last_acc)<1.0:
                    cnt+=1
                    last_acc = acc
                else:
                    cnt = 0
                    last_acc = acc
                if cnt>=3 and early_stop:
                    return acc
                '''====================================='''

            if global_setp >= num_steps: break
        # evaluate
        acc = evaluate(model,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
        wandb_log(global_setp,accuracy=acc)
        if acc > best_acc:
            best_acc = acc
            save_model(args,model,model_name)
        '''============ Early Stop ============='''
        if abs(acc-last_acc)<1.0:
            cnt+=1
            last_acc = acc
        else:
            cnt = 0
            last_acc = acc
        if cnt>=3 and early_stop:
            return acc
        '''====================================='''
        if global_setp >= num_steps: break
    
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("=================================================================================\n")

    return acc

def train_kd(
        args,student,teacher,
        train_loader,test_loader,
        model_name=None,
    ):

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in student.named_parameters() if "lambda" not in n and p.requires_grad], 
            "lr": args.learning_rate, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False
    )

    for param in teacher.parameters():
        param.requires_grad = False

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    device = args.device
    student.to(device)
    teacher.to(device)
    teacher.eval()
    student.zero_grad()
    global_setp = 0
    if args.num_steps>0:
        num_steps = args.num_steps
    else:
        num_steps = args.num_epochs*len(train_loader)
    best_acc = evaluate(student,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
    wandb_log(global_setp,accuracy=best_acc)
    if model_name is None:
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}-KD).pt"

    print("\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    while True:
        student.train()
        epoch_iterator = tqdm(
            train_loader, 
            desc="Training (X / X Steps) (loss=X.X)", 
            bar_format="{l_bar}{r_bar}", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            # forward+backward
            xb, yb = batch[:2]
            xb = xb.to(device)
            yb = yb.to(device)
            logits_s = student(xb)
            loss_ce = criterion_ce(logits_s, yb)
            with torch.no_grad():
                logits_t = teacher(xb)
            loss_kd = criterion_kd(
                F.log_softmax(logits_s, dim=1),
                F.softmax(logits_t, dim=1)
            )
            loss = loss_ce + loss_kd
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_setp += 1
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.4f) (best_acc=%.2f)" % (
                    global_setp, num_steps, loss.item(), best_acc
                )
            )
            # log the training process
            wandb_log(
                global_setp,loss=loss.item(),
            )
            # evaluate
            if global_setp % args.eval_every == 0:
                acc = evaluate(student,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
                wandb_log(global_setp,accuracy=acc)
                if acc > best_acc:
                    best_acc = acc
                    save_model(args,student,model_name)

            if global_setp >= num_steps: break
        # evaluate
        acc = evaluate(student,test_loader,visual_task=args.task_type,sub_label=args.sub_label)
        wandb_log(global_setp,accuracy=acc)
        if acc > best_acc:
            best_acc = acc
            save_model(args,student,model_name)
        if global_setp >= num_steps: break
    
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("=================================================================================\n")

    return acc
        
        
        





    

    







