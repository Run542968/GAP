import torch
import torchvision.ops.roi_align as roialign
from models.clip import clip as clip_pkg
import options
from options import merge_cfg_from_file
from models.clip import build_text_encoder
import dataset
from torch.utils.data import Dataset, DataLoader
from utils.misc import collate_fn


# input = torch.randn((3,4,1,20))
# print(input)
# proposal = torch.tensor([[[0,1,0,10,0],[0,11,0,15.5,0]],[[1,2.4,0,3.5,0],[1,2.3,0,4.5,0]],[[2,1.2,0,1.5,0],[2,1.2,0,1.5,0]]]).float()
# print(input.shape, proposal.shape)
# B,Q,_ = proposal.shape
# box = proposal.reshape(B*Q,-1)
# roi = roialign(input,box,output_size=(1,3))
# print(roi.reshape(B,Q,4,-1))
# print(roi.shape)

def get_text_feats(cl_names, description_dict, device, target_type, text_encoder):
    def get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of a person doing"+" "+c)
        return temp_prompt
    
    def get_description(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # NOTE: default the idx of description is 0.
        return temp_prompt
    
    if target_type == 'prompt':
        act_prompt = get_prompt(cl_names)
    elif target_type == 'description':
        act_prompt = get_description(cl_names)
    elif target_type == 'name':
        act_prompt = cl_names
    else: 
        raise ValueError("Don't define this text_mode.")
    
    tokens = clip_pkg.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
    text_feats = text_encoder(tokens).float()

    return text_feats


if __name__ == "__main__":
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.
    
    device = torch.device(args.device)
    text_encoder, logit_scale = build_text_encoder(args,device)

    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    data_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, num_workers=2, pin_memory=True, shuffle=True)
    iters = iter(data_loader)
    feat, target = next(iters)

    classes = train_dataset.classes
    description_dict = train_dataset.description_dict
    text_feats = get_text_feats(classes, description_dict, device, args.target_type, text_encoder)

    
