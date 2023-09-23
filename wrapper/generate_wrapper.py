import options
from options import merge_cfg_from_file
import torch
from utils.util import setup_seed
from models.clip import build_text_encoder
from models.clip import clip as clip_pkg
import json

def generate_wrapper_v1(text_encoder, description_dict, target_type, split, split_id, device):

    def get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of a person doing"+" "+c)
        return temp_prompt
    
    def get_description(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # 0 is specific to description_v4.json
        return temp_prompt

    # load unseen and seen class_name
    split_seen = [] 
    split_unseen = []
    if split == 75:
        with open('./splits/train_75_test_25/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_seen.append(line[:-1]) 

        with open('./splits/train_75_test_25/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_unseen.append(line[:-1]) 
    elif split == 50:
        with open('./splits/train_50_test_50/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_seen.append(line[:-1]) 

        with open('./splits/train_50_test_50/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_unseen.append(line[:-1]) 
    else:
        raise ValueError(f"Don't have this split: {split}")


    if target_type == 'prompt':
        seen_prompt = get_prompt(split_seen)
        unseen_prompt = get_prompt(split_unseen)

    elif target_type == 'description':
        seen_prompt = get_description(split_seen)
        unseen_prompt = get_description(split_unseen)
    else: 
        raise ValueError("Don't define this text_mode.")
    
    seen_tokens = clip_pkg.tokenize(seen_prompt).long().to(device)
    unseen_tokens = clip_pkg.tokenize(unseen_prompt).long().to(device)

    seen_feats = text_encoder(seen_tokens).float() # s,dim
    unseen_feats = text_encoder(unseen_tokens).float() # u,dim

    transfer_wrapper = torch.einsum("sd,sh->dh",seen_feats,seen_feats) # [dim,dim]


    return transfer_wrapper, seen_feats, unseen_feats


def generate_wrapper_v2(text_encoder, description_dict, target_type, split, split_id, device):

    def get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of a person doing"+" "+c)
        return temp_prompt
    
    def get_description(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # 0 is specific to description_v4.json
        return temp_prompt

    # load unseen and seen class_name
    split_seen = [] 
    split_unseen = []
    if split == 75:
        with open('./splits/train_75_test_25/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_seen.append(line[:-1]) 

        with open('./splits/train_75_test_25/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_unseen.append(line[:-1]) 
    elif split == 50:
        with open('./splits/train_50_test_50/ActivityNet/train/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_seen.append(line[:-1]) 

        with open('./splits/train_50_test_50/ActivityNet/test/split_' +str(split_id)+ '.list', 'r') as filehandle:
            for line in filehandle.readlines():
                split_unseen.append(line[:-1]) 
    else:
        raise ValueError(f"Don't have this split: {split}")


    if target_type == 'prompt':
        seen_prompt = get_prompt(split_seen)
        unseen_prompt = get_prompt(split_unseen)

    elif target_type == 'description':
        seen_prompt = get_description(split_seen)
        unseen_prompt = get_description(split_unseen)
    else: 
        raise ValueError("Don't define this text_mode.")
    
    seen_tokens = clip_pkg.tokenize(seen_prompt).long().to(device)
    unseen_tokens = clip_pkg.tokenize(unseen_prompt).long().to(device)

    seen_feats = text_encoder(seen_tokens).float() # s,dim
    unseen_feats = text_encoder(unseen_tokens).float() # u,dim

    transfer_wrapper = torch.einsum("sd,ud->su",seen_feats,unseen_feats) # [seen,unseen]


    return transfer_wrapper, seen_feats, unseen_feats

if __name__ == '__main__':    
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)
    seed=args.seed
    setup_seed(seed)

    # load CLIP model
    text_encoder, logit_scale = build_text_encoder(args,device)
    
    # load description file
    description_dict = json.load(open(args.description_file_path))
    # obtain wrapper
    transfer_wrapper, seen_feats, unseen_feats = generate_wrapper(text_encoder,description_dict,args.split,args.split_id,args.target_type)
    print(f"transfer_wrapper.shape:{transfer_wrapper.shape}")
    print(f"seen_feats.shape:{seen_feats.shape}")
    print(f"unseen_feats.shape:{unseen_feats.shape}")

    # comute similarity
    
