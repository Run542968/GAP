#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File :  main.py
@Time :  2023/09/12 15:37:33
@Author :  Jia-Run Du
@Version :  1.0
@Contact :  dujr6@mail2.sysu.edu.cn
@License :  Copyright (c) ISEE Lab
@Desc :  main file
"""


from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from utils.misc import collate_fn
import random
import options
from options import merge_cfg_from_file
import numpy as np
from tqdm import tqdm
from utils.util import get_logger, setup_seed
import dataset
from models.ConditionalDetr import build_model
from train import train
from test import test
from tqdm import tqdm
import shutil

import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts, log_metrics


# torch.set_default_tensor_type('torch.cuda.FloatTensor')
def check_directory(args):
    # contruct ckpt directory
    if not os.path.exists(os.path.join('./ckpt/',args.dataset_name)):
        os.makedirs(os.path.join('./ckpt/',args.dataset_name))

    # contruct logger
    if not os.path.exists(os.path.join('./logs/',args.dataset_name)):
        os.makedirs(os.path.join('./logs/',args.dataset_name))
    if os.path.exists(os.path.join('./logs/', args.dataset_name, args.model_name+'.log')):
        os.remove(os.path.join('./logs/', args.dataset_name, args.model_name+'.log'))
    logger = get_logger(os.path.join('./logs/', args.dataset_name, args.model_name+'.log'))

    # contruct results directory
    if not os.path.exists(os.path.join('./results/',args.dataset_name,args.model_name)):
        os.makedirs(os.path.join('./results/',args.dataset_name,args.model_name))
    if os.path.exists(os.path.join('./results/',args.dataset_name,args.model_name)): 
        shutil.rmtree(os.path.join('./results/',args.dataset_name,args.model_name))
    return logger

if __name__ == '__main__':
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)
    seed=args.seed
    setup_seed(seed)
    logger = check_directory(args)
    logger.info('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
    logger.info(args)

    if args.use_mlflow:
        #### mlflow ####
        if args.task == 'zero_shot':
            if args.eval_proposal: 
                experiment_name = "_".join([args.dataset_name,args.feature_type,args.task,str(args.split),"binary"])
            else:
                experiment_name = "_".join([args.dataset_name,args.feature_type,args.target_type,args.task,str(args.split)])
        elif args.task == 'close_set':
            if args.eval_proposal: 
                experiment_name = "_".join([args.dataset_name,args.feature_type,args.task,"binary"])
            else:
                experiment_name = "_".join([args.dataset_name,args.feature_type,args.target_type,args.task])
        else:
            raise ValueError("don't define this setting.")
        if mlflow.get_experiment_by_name(experiment_name) == None: 
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Create experiment id by name, id:{experiment_id}")
        else:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id # if the experiment of per-set name exist, get its id
            print(f"Get experiment id by name, id:{experiment_id}")
            
        run_name = args.model_name
        mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id
        )
        log_params(vars(args)) # NameSpace -> dict
        #### mlflow ####


    # load dataset
    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    train_val_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    # load model
    model, criterion, postprocessor = build_model(args,device)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters/1000000} M')

    param_dicts = [
        # the parameters in transformaer
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and \
                                                                "sematnic_visual_head" not in n and \
                                                                "sematnic_text_head" not in n and \
                                                                p.requires_grad]
         },
        # the parameters in backbone
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        # the parameters in semantic head: instance_visual_head
        {
            "params": [p for n, p in model.named_parameters() if "sematnic_visual_head" in n and p.requires_grad],
            "lr": args.lr_semantic_head,
        },
        # the parameters in semantic head: instance_text_head
        {
            "params": [p for n, p in model.named_parameters() if "sematnic_text_head" in n and p.requires_grad],
            "lr": args.lr_semantic_head,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    best_stats = {}
    for epoch in tqdm(range(args.epochs)):
        epoch_loss_dict_scaled = train(model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_max_norm)
        if args.use_mlflow: # for mlflow
            log_metrics(epoch_loss_dict_scaled,step=epoch)
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'last_' + args.model_name + '.pkl'))

        if epoch % args.train_interval == 0 and args.train_interval != -1:
            train_stats = test(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=train_val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args)
            logger.info('||'.join(['Train map @ {} = {:.3f} '.format(train_stats['iou_range'][i],train_stats['per_iou_ap_raw'][i]*100) for i in range(len(train_stats['iou_range']))]))
            logger.info('Intermediate Train mAP Avg ALL: {}'.format(train_stats['mAP_raw']*100))
            if args.use_mlflow: # for mlflow
                    res_dict = {'train_IoU_'+str(k):v*100 for k,v in zip(train_stats['iou_range'],train_stats['per_iou_ap_raw'])}
                    res_dict.update({"train_mAP":train_stats['mAP_raw']*100})
                    log_metrics(res_dict,step=epoch)

        
        if epoch % args.test_interval == 0:
            test_stats = test(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args)
            logger.info('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_stats['iou_range'][i],test_stats['per_iou_ap_raw'][i]*100) for i in range(len(test_stats['iou_range']))]))
            logger.info('Intermediate mAP Avg ALL: {}'.format(test_stats['mAP_raw']*100))
            
            if args.use_mlflow: # for mlflow
                res_dict = {'IoU_'+str(k):v*100 for k,v in zip(test_stats['iou_range'],test_stats['per_iou_ap_raw'])}
                res_dict.update({"mAP":test_stats['mAP_raw']*100})
                log_metrics(res_dict,step=epoch)

            # update best
            if test_stats['mAP_raw'] > best_stats.get('mAP_raw',0.0):
                best_stats = test_stats
                logger.info('new best metric {:.4f}@epoch{}'.format(best_stats['mAP_raw']*100, epoch))
                torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'best_' + args.model_name + '.pkl'))
        
    
    iou = best_stats['iou_range']
    max_map = best_stats['per_iou_ap_raw']
    max_Avg = best_stats['mAP_raw']
    logger.info('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
    logger.info('MAX mAP Avg ALL: {}'.format(max_Avg*100))
    
    if args.use_mlflow:     # for mlflow
        best_res_dict = {'best_IoU_'+str(k):v*100 for k,v in zip(best_stats['iou_range'],best_stats['per_iou_ap_raw'])}
        best_res_dict.update({"best_mAP":best_stats['mAP_raw']*100})
        best_res_dict.update({"best_epoch":best_stats['epoch']})
        log_metrics(best_res_dict)
        mlflow.end_run()


