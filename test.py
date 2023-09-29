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
                                    eval_proposal=args.eval_proposal
                                    )

    epoch_loss_dict = {}
    count = 0
    # res_dict = {}
    for samples, targets in data_loader:
        samples = samples.to(device)
        # targets = [{k: v.to(device) if k in ['segments', 'labels'] else v for k, v in t.items()} for t in targets] # Not Required in inferene stage
        
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(samples, classes, description_dict,targets)

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
