import os.path as osp
import torch
import logging
from eval import tad_eval
logger = logging.getLogger()


@torch.no_grad()
def test(model, 
         criterion, 
         postprocessor, 
         data_loader, 
         dataset_name, 
         epoch,
         device, 
         args):
    model.eval()
    criterion.eval()
    postprocessor.eval()

    iou_range = [0.3, 0.4, 0.5, 0.6, 0.7] if dataset_name == 'Thumos14' else [num/100 for num in range(50, 100, 5)]
    logger.info(f'iou range {iou_range} in dataset {dataset_name}.')

    # action_evaluator = None
    action_evaluator = getattr(tad_eval,dataset_name+"Evaluator")(dataset_name=dataset_name, epoch=epoch, dataset=data_loader.dataset, iou_range=iou_range,
                                    nms_mode=['raw'],
                                    eval_proposal=args.eval_proposal,
                                    filter_threshold=args.filter_threshold
                                    )

    epoch_loss_dict = {}
    count = 0
    # res_dict = {}
    for samples, targets in data_loader:
        samples = samples.to(device)
        # targets = [{k: v.to(device) if k in ['segments', 'labels'] else v for k, v in t.items()} for t in targets] # Not Required in inferene stage
        
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(samples, classes, description_dict,targets,epoch)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # loss_dict_unscaled = {f'{k}_unscaled': v.item() for k, v in loss_dict.items()} # logging all losses thet are computed (note that some of these are not allpied for backward)
        # loss_dict_scaled = {k: v.item() * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        # loss_value = sum(loss_dict_scaled.values())

        # # update the epoch_loss
        # epoch_loss_dict.update({k: epoch_loss_dict.get(k,0.0) + v for k, v in loss_dict_unscaled.items()})
        # epoch_loss_dict.update({k: epoch_loss_dict.get(k,0.0) + v for k, v in loss_dict_scaled.items()})
        
        count = count + len(targets)
        logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)})")
        # logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_value:{loss_value}, loss_dict_scaled:{loss_dict_scaled}")
        # logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_dict_unscaled:{loss_dict_unscaled}")
        
        # post process
        video_duration = torch.FloatTensor([t["video_duration"] for t in targets]).to(device)
        results = postprocessor(outputs, video_duration, args.eval_proposal)
        res = {target['video_name']: output for target, output in zip(targets, results)}
        # res_dict.update(res)

        if action_evaluator is not None:
            action_evaluator.update(res)

    # epoch_loss_dict.update({k: v/count for k, v in epoch_loss_dict.items()})
    # logger.info(f"Inference Epoch: {epoch}, epoch_loss_dict:{epoch_loss_dict}")

    # accumulate predictions from all videos
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate()
        # dump detections
        if args.save_result:
            save_path = osp.join('./results/',args.dataset_name,args.model_name,'detection_{}_'+str(epoch)+'_{}.json')
            action_evaluator.dump_detection(save_path)
        action_evaluator.summarize()

    # summary the all stats for logger
    stats = {}
    stats['iou_range'] = iou_range
    stats['epoch'] = epoch
    if action_evaluator is not None: # fusion the nms_mode to the name of key
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v) for k, v in stats.items() if k.startswith('mAP')])
        logger.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    
    return stats

if __name__ == "__main__":
    import options
    from options import merge_cfg_from_file
    from utils.util import get_logger, setup_seed
    import dataset
    from torch.utils.data import DataLoader
    from utils.misc import collate_fn
    from models.ConditionalDetr import build_model
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)
    seed=args.seed
    setup_seed(seed)

    # load dataset
    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    # train_val_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    # load model
    model, criterion, postprocessor = build_model(args,device)
    ckpt_path = os.path.join("./ckpt",args.dataset_name,"best_"+args.model_name+".pkl")
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    iters = iter(train_loader)
    samples, targets = next(iters)
    samples = samples.to(device)
    # targets = [{k: v.to(device) if k in ['segments', 'labels'] else v for k, v in t.items()} for t in targets] # Not Required in inferene stage
    
    classes = train_loader.dataset.classes
    description_dict = train_loader.dataset.description_dict
    outputs = model(samples, classes, description_dict,targets)

    memory = outputs['memory'][-1] # [enc_layers, b,t,c]
    idx = 9
    save_dir = os.path.join('./heatmap',args.target_type+"_memory",targets[idx]['video_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 可视化特征的self-similarity matirx
    self_similarity = torch.einsum("td,ld->tl",memory[idx],memory[idx])
    self_similarity = self_similarity.cpu().detach().numpy()
    fig = plt.figure(figsize=(16,6))
    sns.heatmap(self_similarity,cmap="YlGnBu")
    plt.savefig(os.path.join(save_dir,'memory_self_similarity.png'))
    
    print(targets[idx]['video_name'])
    print(targets[idx]['mask_labels'])
    print(targets[idx]['semantic_labels'])
    print(targets[idx]['segments'])

# CUDA_VISIBLE_DEVICES=2 python test.py --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --batch_size 16 --target_type "prompt" --model_name "ActivityNet13_CLIP_prompt_zs_v6_1" --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone
