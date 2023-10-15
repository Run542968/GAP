#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File :  dataset.py
@Time :  2023/09/12 15:44:44
@Author :  Jia-Run Du
@Version :  1.0
@Contact :  dujr6@mail2.sysu.edu.cn
@License :  Copyright (c) ISEE Lab
@Desc :  TAL Dataset 
"""

import json
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import options
from options import merge_cfg_from_file
from utils.segment_ops import segment_t1t2_to_cw
from utils.misc import collate_fn
from utils.util import id2onehot
import os
import math

logger = logging.getLogger()

      
class BaseDataset(Dataset):
    def __init__(self, subset, mode, args):
        '''
        BaseDataset
        Parameters:
            subset: train/val/test, the subset to specific dataset
            mode: train/inference, the stage of model, only effecitive for zero_shot, since the classes between train and inference are different
            task: "close_set" or "zero_shot"
            anno_file_path: the path to annotation file
            feature_info_path: the path to feature_info information, which includes the feature length and second
            feature_path: the path to obtain video feature (i.e. 16 frames-1 snippet)
            feature_type: the type of feature, "I3D" or "CLIP"
            feature_stride: the stride to form snippet feature from frame feature
            description_file_path: in zero_shot setting, the description file includes the text description for class name
            split: the proportion of train/inference in classes name
            split_id: the id of split file is used
            # binary: whether to just distinguish foreground and background
        '''

        super(BaseDataset).__init__()
        self.subset = subset
        self.mode = mode
        self.task = args.task
        self.anno_file_path = args.anno_file_path
        self.feature_info_path = args.feature_info_path
        self.feature_path = args.feature_path
        self.description_file_path = args.description_file_path
        self.feature_type = args.feature_type
        self.feature_stride = args.feature_stride
        self.rescale_length = args.rescale_length
        self.split = args.split
        self.split_id = args.split_id
        self.target_type = args.target_type
        self.complete_loss = args.complete_loss
        # self.binary = args.binary

        self.slice_size = args.slice_size
        self.slice_overlap = args.slice_overlap if self.mode=="train" else args.inference_slice_overlap

        self.classes, self.valid_anno_dict, self.valid_video_list, self.anno_dict, self.feature_info, self.src_valid_anno_dict = self._prepare_gt()
        self.description_dict = self._get_description(self.description_file_path)
        
        # if self.binary:
        #     logger.info(f"The num of valid videos is {len(self.valid_anno_dict)} in subset {subset}, there are {sum([len(v['annotations']) for k,v in self.valid_anno_dict.items()])} instances of {1} classes.")
        # else:
        logger.info(f"The num of valid videos is {len(self.valid_anno_dict)} in subset {subset}, there are {sum([len(v['annotations']) for k,v in self.valid_anno_dict.items()])} instances of {len(self.classes)} classes.")

    def _get_description(self, description_file_path):
        description_dict = json.load(open(description_file_path))
        return description_dict

    def _prepare_gt(self):
        '''parse annotation file'''
        anno_dict = json.load(open(self.anno_file_path))
        feature_info = json.load(open(self.feature_info_path))

        # get classes list
        if self.task == 'close_set':
            classes = anno_dict['classes'] # get all class names, without "ambiguous" 
        elif self.task == 'zero_shot':
            classes = self._get_classes_for_zero_shot(self.mode,self.split,self.split_id) # implementation in child class
        else:
            raise ValueError("Don't define this task_setting.")

        # get training labels: valid_anno_dict, a dict that contains the video information; valid_video_list, a list that contain video name
        valid_anno_dict, valid_video_list, src_valid_anno_dict = self._parse_anno(anno_dict,
                                                             feature_info,
                                                             classes,
                                                             self.subset,
                                                             self.task,
                                                             self.feature_stride,
                                                             self.mode,
                                                             self.slice_overlap,
                                                             self.slice_size) # implementation in child class
        
        return classes,valid_anno_dict,valid_video_list,anno_dict,feature_info,src_valid_anno_dict

    def __len__(self):
        return len(self.valid_video_list)

    def _rescale_feature(self,feats):
        
        if self.rescale_length == 0:
            return feats
        else:
            feats = feats.permute(1,0) # [C,T]
            # interpolate
            feats = F.interpolate(feats.unsqueeze(0),size=self.rescale_length)[0] # [C,self.slice_len]
            feats = feats.permute(1,0) # [T,C]
            return feats

    def _get_feature(self,video_name, feature_path, feature_type):
        
        if feature_type == "CLIP":
            feat_path = os.path.join(feature_path, video_name+".npy")
        elif feature_type == "I3D":
            raise NotImplementedError
        
        if os.path.exists(feat_path):
            video_feat = np.load(feat_path, allow_pickle=True) # T,dim
        else:
            raise ValueError(f"Don't exist the feat: {feat_path}")
        
        feature_data = torch.from_numpy(video_feat).float().contiguous()

        # re-scale feature to uniform length
        feature = self._rescale_feature(feature_data)

        return feature

    def _get_train_label(self, video_name, valid_anno_dict, classes, feat_length):
        
        '''get normalized target'''
        video_anno = valid_anno_dict[video_name]
        segments_anno = video_anno['annotations']
        feature_duration  = video_anno['feature_duration']
        # update the valid_anno_dict by a hook approach
        self.valid_anno_dict[video_name]['feature_length'] = feat_length
        num_classes = len(self.classes)
 
        target = {
            'segments': [], 
            'labels': [], # the category labels for detector
            'semantic_labels':[],
            'label_names': [],
            'video_name': video_name,
            'video_duration': feature_duration,   # only used in inference
            'instance_masks':{}, # the mask to get action instance
            'segmentation_labels': np.full(feat_length,num_classes), # [T]
            'segmentation_onehot_labels': np.full((feat_length,num_classes),1/num_classes), # [T,num_classes], the label for snippet-level semantic segmentation
            'mask_labels':np.full(feat_length,0) # [T]
            }
        
        # sort the segments follow time sequence
        segments_anno = list(sorted(segments_anno, key=lambda x: sum(x['segment'])))
        bg_start = 0
        bg_end = feature_duration
        for seg_anno in segments_anno: # a list of dict [{'segment': ,'labels': }, ]
            
            segment = seg_anno['segment'] 

            if self.complete_loss and len(segments_anno)>0:
                # add bg instance
                if segment[0]-bg_start > 0.5: # guarantee the bg instance more than 1 snippet
                    bg_segment = [bg_start,segment[0]]
                    bg_semantic_labels = num_classes
                    bg_start = segment[1] # update bg_start
                    target['segments'].append(bg_segment)
                    target['semantic_labels'].append(bg_semantic_labels)
                    target['labels'].append(1)  # the category labels for detector to classify
                    target['label_names'].append("background")


            # special rule for thumos14, treat ambiguous instances as negatives, although the ambiguous has been dropped in self.parse_gt()
            if seg_anno['label'] not in classes:
                continue

            if self.target_type != "none":
                label_id = 0
                label_name = 'foreground'
            else:
                # the label id of first forground class is 0
                label_id = classes.index(seg_anno['label']) # the index come from the classes in anno_file
                label_name = seg_anno['label']

            target['segments'].append(segment)
            target['labels'].append(label_id)  # the category labels for detector to classify
            target['label_names'].append(label_name)

            semantic_label = classes.index(seg_anno['label']) # the category labels for semantic classification
            target['semantic_labels'].append(semantic_label)
            
            # add instance_masks to target dict
            if seg_anno['label'] not in target['instance_masks'].keys():
                target['instance_masks'][seg_anno['label']] = {'label_id':semantic_label,'mask':np.ones(feat_length,dtype=bool)}
            
            start_float, end_float = np.array(segment)/feature_duration*feat_length
            start, end = np.floor(start_float).astype(int), np.ceil(end_float).astype(int)
            start_idx, end_idx = max(start,0), min(end + 1,feat_length)
            target['instance_masks'][seg_anno['label']]['mask'][start_idx:end_idx] = False

            # update segmentation labels 
            target['segmentation_onehot_labels'][start_idx:end_idx,:] = np.repeat(id2onehot(num_classes,semantic_label).reshape(1,-1),end_idx-start_idx,axis=0)
            target['segmentation_labels'][start_idx:end_idx] = semantic_label

            # update class-agnostic mask labels
            target['mask_labels'][start_idx:end_idx] = 1


        if self.complete_loss and len(segments_anno)>0:
            # add bg instance 
            if bg_end-segment[1] > 0.5: # guarantee the bg instance more than 1 snippet
                bg_segment = [segment[1],bg_end]
                bg_semantic_labels = num_classes
                target['segments'].append(bg_segment)
                target['semantic_labels'].append(bg_semantic_labels)
                target['labels'].append(1)  # the category labels for detector to classify
                target['label_names'].append("background")

        # normalized the coordinate
        target['segments'] = np.array(target['segments']) / feature_duration
        
        if len(target['segments']) > 0:
            target['segments'] = segment_t1t2_to_cw(target['segments'])

            # convert to torch format
            for k, dtype in zip(['segments', 'labels'], ['float32', 'int64']):
                if not isinstance(target[k], torch.Tensor):
                    target[k] = torch.from_numpy(np.array(target[k], dtype=dtype))
            
            # add instance_masks to target dict, cover to torch format
            for label_name in target['instance_masks'].keys():
                target['instance_masks'][label_name]['mask'] = torch.from_numpy(target['instance_masks'][label_name]['mask'])
                target['instance_masks'][label_name]['label_id'] = torch.from_numpy(np.array(target['instance_masks'][label_name]['label_id'],dtype='int64'))

            # convert 'segmentation_onehot_labels' to torch format
            target['segmentation_onehot_labels'] = torch.from_numpy(target['segmentation_onehot_labels'])
            target['segmentation_labels'] = torch.from_numpy(target['segmentation_labels'])

            # covert 'mask_labels' to torch format
            target['mask_labels'] = torch.from_numpy(target['mask_labels'])

            # covert 'semantic_labels' to torch format
            target['semantic_labels'] = torch.from_numpy(np.array(target['semantic_labels'],dtype='int64'))
        return target

    def __getitem__(self, index):
        video_name = self.valid_video_list[index]
        video_feat = self._get_feature(video_name, self.feature_path, self.feature_type) # T,dim
        T,dim = video_feat.shape
        target = self._get_train_label(video_name, self.valid_anno_dict, self.classes, T)

        return video_feat, target

class Thumos14Dataset(BaseDataset):
    def __init__(self, subset, mode, args):
        subset_mapping = {"train":"val","inference":"test"}
        super(Thumos14Dataset,self).__init__(subset_mapping[subset], mode, args)

    def _get_classes_for_zero_shot(self,mode,split,split_id):
        if split == 75:
            split_75_train = [] ## 75:25 split
            split_75_test = [] ## 75:25 split

            with open('./splits/train_75_test_25/THUMOS14/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_75_train.append(line[:-1]) 

            with open('./splits/train_75_test_25/THUMOS14/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_75_test.append(line[:-1]) 

            if mode == "train":
                classes = split_75_train
            elif mode == "inference":
                classes = split_75_test
            else:
                raise ValueError(f"Don't define this mode: {mode}.")
            
        elif split == 50:
            split_50_train = [] ## 50:50 split
            split_50_test = [] ## 50:50 split

            with open('./splits/train_50_test_50/THUMOS14/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_50_train.append(line[:-1]) 

            with open('./splits/train_50_test_50/THUMOS14/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_50_test.append(line[:-1]) 
            
            if mode == "train":
                classes = split_50_train
            elif mode == "inference":
                classes = split_50_test
            else:
                raise ValueError(f"Don't define this mode: {mode}.")
            
        else:
            raise ValueError(f"Don't have this split: {split}")

        return classes

    def get_valid_anno(self, gt_instances, slice, thr=0.75,
            start_getter=lambda x: x['segment'][0],
            end_getter=lambda x: x['segment'][1]):
        '''Perform integrity based instance filtering'''
        start, end = slice
        kept_instances = []
        for inst in gt_instances:
            # ignore insts outside the time window (slice)
            if end_getter(inst) <= start or start_getter(inst) >= end:
                continue
            else:
                # clamped inst
                new_start = max(start_getter(inst), start)
                new_end = min(end_getter(inst), end)
                integrity = (new_end - new_start) * 1.0 / (end_getter(inst) - start_getter(inst)) # the ratio of (new segment length)/(origin segment length)
                
                if integrity >= thr:
                    new_inst = {k:v for k,v in inst.items()}
                    new_inst['segment'] = [new_start - start, new_end - start]
                    kept_instances.append(new_inst)
        return kept_instances

    def _parse_anno(self,anno_dict,feature_info,classes,subset,task,feature_stride,mode,slice_overlap,slice_size,exclude_videos=None):
        src_valid_anno_dict = {} # store the origin video annotations
        valid_anno_dict = {} # store the video annotations
        valid_video_list = [] # store the video name

        anno_data = anno_dict['database']

        video_set = set([x for x in anno_data if anno_data[x]['subset'] in subset]) # get the video name that belongs to 'subset'
        video_set = video_set.intersection(feature_info.keys()) # get the video name belongs both anno_dict and feature_info
        
        exclude_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
        if exclude_videos is not None:
            assert isinstance(exclude_videos, (list, tuple))
            video_set = video_set.difference(exclude_videos)
        
        video_list = list(sorted(video_set)) # the video name that belongs to 'subset' and not belongs to excude_videos

        for video_name in video_list:
            feat_info = feature_info[video_name]
            # number of frames or snippets
            feature_length = int(feat_info['snippet_length'])   
            feature_duration = float(feat_info['snippet_duration']) # the second that feature mathched
            video_fps = float(feat_info['video_fps'])
            feature_fps = float(video_fps/feature_stride)

            if task == "close_set": 
                # remove ambiguous instances on THUMOS14
                annotations = [x for x in anno_data[video_name]['annotations'] if x['label'] != 'Ambiguous'] # discard the segments(instance) that belongs to "Ambiguous"
                annotations = list(sorted(annotations, key=lambda x: sum(x['segment']))) # sort segment based on the timestamp
            elif task == "zero_shot":
                # remove ambiguous instances and zero-shot instances on THUMOS14
                annotations = [x for x in anno_data[video_name]['annotations'] if x['label'] != 'Ambiguous' and x['label'] in classes] # choose the segments(instance) that not belongs to "Ambiguous" and inside classes 
                annotations = list(sorted(annotations, key=lambda x: sum(x['segment']))) # sort segment based on the timestamp
            else:
                raise ValueError(f"Don't define this task: {task}")

            # because a snippet need stride frames, the fps is 30. it must meet the minimum snippet length
            valid_annotations = [x for x in annotations if x['segment'][1] - x['segment'][0] > (feature_stride/video_fps)]
            

            # crop video into slices of fixed window length, i.e., 128
            # slice is the index to get feat
            slide = int(slice_size * (1 - slice_overlap))
            if feature_length <= slice_size:
                slices = [[0,feature_length-1]]
            else:
                # (length-kernel)/stride + 1
                num_complete_slices = math.floor((feature_length-slice_size)/slide) + 1
                slices = [[int(i*slide), int((i*slide)+slice_size-1)] for i in range(num_complete_slices)]

                if (num_complete_slices-1)*slide + slice_size < feature_length:
                    if mode == "inference":
                        # take the last incomplete slice
                        last_slice_start = int(slide * num_complete_slices)
                    else:
                        # move left to get a complete slice.
                        # This is a historical issue. The performance might be better
                        # if we keep the same rule for training and inference 
                        last_slice_start = max(0, feature_length - slice_size)
                    slices.append([last_slice_start, feature_length-1])
            
            for slice in slices:
                time_slices = [slice[0] / feature_fps, slice[1] / feature_fps]
                feature_second = time_slices[1] - time_slices[0]
                # perform integrity-based instance filtering
                valid_window_annotations = self.get_valid_anno(valid_annotations, time_slices)
                
                if mode == "inference" or len(valid_window_annotations) >= 1: # test phrase will add all slices, although it has empty annotation
                    # rename the video slice
                    new_vid_name = video_name + '_window_{}_{}'.format(*slice)
                    new_vid_info = {
                        'annotations': valid_window_annotations, 'src_video_name': video_name, 
                        'video_fps': video_fps, 'feature_fps': feature_fps,
                        # 'feature_length': slice_size, 
                        'subset': anno_data[video_name]['subset'], 'feature_duration': feature_second, 'time_offset': time_slices[0]}
                    valid_anno_dict[new_vid_name] = new_vid_info
                    valid_video_list.append(new_vid_name)
            
            
            # store the origin annotation for evaluate
            if len(valid_annotations) > 0: # avoid the empty annotations (mainly applied for zero-set) 
                src_valid_anno_dict[video_name] = {
                    'video_name': video_name, 'annotations': valid_annotations, 
                    'video_fps': video_fps, 
                    # 'feature_length': feature_length,
                    'subset': anno_data[video_name]['subset'], 'feature_duration': feature_duration}


        return valid_anno_dict, valid_video_list, src_valid_anno_dict
 
    def _get_feature(self,video_name, feature_path, feature_type):
        
        src_video_name = self.valid_anno_dict[video_name]['src_video_name']
        if feature_type == "CLIP":
            feat_path = os.path.join(feature_path, src_video_name+".npy")
        elif feature_type == "I3D":
            raise NotImplementedError
        
        if os.path.exists(feat_path):
            video_feat = np.load(feat_path, allow_pickle=True) # T,dim
        else:
            raise ValueError(f"Don't exist the feat: {feat_path}")
        
        feature_data = torch.from_numpy(video_feat).float().contiguous() # Txdim

        slice_start, slice_end = [int(x) for x in video_name.split('_')[-2:]]
        assert slice_end  > slice_start
        assert slice_start < feature_data.shape[0]
        feature_data = feature_data[slice_start:slice_end+1,:] # Txdim

        # if feature_data.shape[0] < self.slice_size: # only appearing in inference stage and feature_length < slice_size
        #     diff = self.slice_size - feature_data.shape[0]
        #     feature_data = np.pad(
        #         feature_data, ((0, diff), (0, 0)), mode='constant') # padding zero in tail

        #     # IMPORATANT: if padded is done, the length info must be modified
        #     self.valid_anno_dict[video_name]['feature_length'] = self.slice_size
        #     self.valid_anno_dict[video_name]['feature_duration'] = self.slice_size / self.valid_anno_dict[video_name]['feature_fps']
        
        feature = torch.Tensor(feature_data).float().contiguous()

        return feature





class ActivityNet13Dataset(BaseDataset):
    def __init__(self, subset, mode, args):
        subset_mapping = {"train":"train","inference":"val"}
        super(ActivityNet13Dataset,self).__init__(subset_mapping[subset], mode, args)

    def _get_classes_for_zero_shot(self,mode,split,split_id):
        if split == 75:
            split_75_train = [] ## 75:25 split
            split_75_test = [] ## 75:25 split

            with open('./splits/train_75_test_25/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_75_train.append(line[:-1]) 

            with open('./splits/train_75_test_25/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_75_test.append(line[:-1]) 

            if mode == "train":
                classes = split_75_train
            elif mode == "inference":
                classes = split_75_test
            else:
                raise ValueError(f"Don't define this mode: {mode}.")
            
        elif split == 50:
            split_50_train = [] ## 50:50 split
            split_50_test = [] ## 50:50 split

            with open('./splits/train_50_test_50/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_50_train.append(line[:-1]) 

            with open('./splits/train_50_test_50/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
                for line in filehandle.readlines():
                    split_50_test.append(line[:-1]) 
            
            if mode == "train":
                classes = split_50_train
            elif mode == "inference":
                classes = split_50_test
            else:
                raise ValueError(f"Don't define this mode: {mode}.")
            
        else:
            raise ValueError(f"Don't have this split: {split}")

        return classes

    def _parse_anno(self,anno_dict,feature_info,classes,subset,task,feature_stride,mode,slice_overlap=None,slice_size=None,exclude_videos=None):
        valid_anno_dict = {} # stor the video annotations
        valid_video_list = [] # store the video name

        anno_data = anno_dict['database']

        video_set = set([x for x in anno_data if anno_data[x]['subset'] in subset]) # get the video name that belongs to 'subset'
        video_set = video_set.intersection(feature_info.keys()) # get the video name belongs both anno_dict and feature_info
        
        if exclude_videos is not None:
            assert isinstance(exclude_videos, (list, tuple))
            video_set = video_set.difference(exclude_videos)
        
        video_list = list(sorted(video_set)) # the video name that belongs to 'subset' and not belongs to excude_videos

        for video_name in video_list:
            feat_info = feature_info[video_name]
            # number of frames or snippets
            feature_length = int(feat_info['snippet_length'])   
            feature_duration = float(feat_info['snippet_duration']) # the second that feature mathched
            video_fps = float(feat_info['video_fps'])

            if task == "close_set": 
                annotations = [x for x in anno_data[video_name]['annotations']] # discard the segments(instance) that belongs to "Ambiguous"
                annotations = list(sorted(annotations, key=lambda x: sum(x['segment']))) # sort segment based on the timestamp
            elif task == "zero_shot":
                # remove ambiguous instances and zero-shot instances on THUMOS14
                annotations = [x for x in anno_data[video_name]['annotations'] if x['label'] in classes] # choose the segments(instance) that inside classes 
                annotations = list(sorted(annotations, key=lambda x: sum(x['segment']))) # sort segment based on the timestamp
            else:
                raise ValueError(f"Don't define this task: {task}")

            # Remove incorrect annotions on ActivityNet (0.02 duration). Beside, because a snippet need 16 frames, the fps is 30. it must meet the minimum snippet length
            valid_annotations = [x for x in annotations if x['segment'][1] - x['segment'][0] > (feature_stride/video_fps) and x['segment'][1] - x['segment'][0] > 0.02]


            # fliter zero instance video
            if len(valid_annotations) > 0: # avoid the empty annotations (mainly applied for zero-set) 
                
                valid_anno_dict[video_name] = {
                    'video_name': video_name, 'annotations': valid_annotations, 
                    'video_fps': video_fps, 'feature_length': feature_length,
                    'subset': anno_data[video_name]['subset'], 'feature_duration': feature_duration}
                valid_video_list.append(video_name)
            else:
                continue
        
        return valid_anno_dict, valid_video_list, None




if __name__ == "__main__":
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    print(args)
    train_dataset = Thumos14Dataset(subset='train', mode='train', args=args)
    # print(f"dataset.description:{train_dataset.description_dict}")
    # print(f"dataset.classes:{train_dataset.classes}")
    
    data_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, num_workers=2, pin_memory=True, shuffle=True)
    iters = iter(data_loader)
    feat, target = next(iters)
    # print(f"feat.tensors:{feat.tensors.shape}")
    # print(f"feat.mask:{feat.mask}")
    # print(f"target:{target}")
    # print(f"target[0]['segmentation_onehot_labels']: {target[0]['segmentation_onehot_labels']}")
    # print(f"target[0]['segmentation_onehot_labels'].shape:{target[0]['segmentation_onehot_labels'].shape}")
    # print(f"target[0]['segmentation_onehot_labels'].dtype:{target[0]['segmentation_onehot_labels'].dtype}")
    # print(f"target[0]['segmentation_labels'].dtype:{target[0]['segmentation_labels']}")
    # gt_coordinations = [t['segments'] for t in target]
    # print(f"gt_coordinations:{gt_coordinations}")
    # gt_coordinations = torch.cat(gt_coordinations,dim=0) # [batch_instance_num,2]->"center,width"
    # print(f"gt_coordinations.shape:{gt_coordinations.shape}")
    # gt_labels = [t['labels'] for t in target]
    # print(f"gt_labels:{gt_labels}")
    # gt_labels = torch.cat(gt_labels,dim=0) # [batch_instance_num,1]->"class id"
    # print(f"gt_labels.shape:{gt_labels.shape}")
    print(f"target[0]['mask_labels']:{target[0]['mask_labels']}")
    print(f"target[0]['semantic_labels']:{target[0]['semantic_labels']}")
    print(f"target[0]['segments']:{target[0]['segments']}")

# CUDA_VISIBLE_DEVICES=4 python dataset.py --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml"
