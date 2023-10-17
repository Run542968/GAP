import torch
import torchvision.ops.roi_align as roialign
from models.clip import clip as clip_pkg
import options
from options import merge_cfg_from_file
from models.clip import build_text_encoder
import dataset
from torch.utils.data import Dataset, DataLoader
from utils.misc import collate_fn
import seaborn as sns
import matplotlib.pyplot as plt
from utils.util import get_logger, setup_seed
import os
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
import numpy as np

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
    
    seed=args.seed
    setup_seed(seed)
    device = torch.device(args.device)
    text_encoder, logit_scale = build_text_encoder(args,device)

    data_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    # data_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    data_loader = DataLoader(data_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, shuffle=True)
    iters = iter(data_loader)
    feat, target = next(iters)

    classes = data_dataset.classes
    print(f"classes_list:{classes}")
    description_dict = data_dataset.description_dict
    text_feats = get_text_feats(classes, description_dict, device, args.target_type, text_encoder)

    visual_feats, mask = feat.to(device).decompose()
    visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
    logit_scale = logit_scale.exp()
    logits = torch.einsum("btd,cd->btc",logit_scale*visual_feats,text_feats)

    for idx in range(len(logits)):
        # idx = 5
        save_dir = os.path.join('./heatmap',args.target_type,target[idx]['video_name'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 可视化分类logits
        # logits = logits.sigmoid()
        semantic_logits = logits.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(semantic_logits[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'Semantic.png'))
        plt.close()

        # 可视化经过softmax后的分类logits
        semantic_logits_softmax = logits.softmax(dim=-1)
        semantic_logits_softmax = semantic_logits_softmax.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(semantic_logits_softmax[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'semantic_logits_softmax.png'))
        plt.close()

        loc_logits,_ = logits.max(dim=-1,keepdim=True)
        loc_logits = loc_logits.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(loc_logits[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'Loc.png'))
        plt.close()

        mean_loc_logits = logits.mean(dim=-1,keepdim=True)
        mean_loc_logits = mean_loc_logits.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(mean_loc_logits[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'MeanLoc.png'))
        plt.close()

        fig = plt.figure(figsize=(16,6))
        sns.heatmap(target[idx]['mask_labels'].unsqueeze(0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'GT.png'))
        plt.close()

        # 可视化视频的self-similarity matirx
        self_similarity = torch.einsum("td,ld->tl",visual_feats[idx],visual_feats[idx])
        self_similarity = self_similarity.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(self_similarity,cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'self_similarity.png'))
        plt.close()

        # neighbor similarity
        nerghbor_similarity = torch.einsum("td,td->td",visual_feats[idx][:-1,:],visual_feats[idx][1:,:]).sum(dim=1)
        nerghbor_similarity = nerghbor_similarity.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        plt.plot(np.arange(len(nerghbor_similarity)),nerghbor_similarity)
        plt.savefig(os.path.join(save_dir,'nerghbor_similarity.png'))
        plt.close()


        # 可视化视频的L2 norm
        norm_visual = visual_feats.norm(dim=-1,keepdim=True) # [B,T,1]
        norm_visual = norm_visual.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(norm_visual[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'visual_norm.png'))
        plt.close()

        # 可视化模型特征的self-similarity matirx
        model, criterion, postprocessor = build_model(args,device)
        ckpt_path = os.path.join("./ckpt",args.dataset_name,"best_"+args.model_name+".pkl")
        model.load_state_dict(torch.load(ckpt_path))
        model.to(device)
        model.eval()

        samples = feat.to(device)
        # targets = [{k: v.to(device) if k in ['segments', 'labels'] else v for k, v in t.items()} for t in targets] # Not Required in inferene stage
        
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict
        outputs = model(samples, classes, description_dict,target,-1)
        memory = outputs['memory'][-1] # [enc_layers, b,t,c]
        memory = memory / memory.norm(dim=-1,keepdim=True)
        memory = torch.einsum("btc,bt->btc",memory, ~mask)

        memory_self_similarity = torch.einsum("td,ld->tl",memory[idx],memory[idx])
        memory_self_similarity = memory_self_similarity.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(memory_self_similarity,cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'memory_self_similarity.png'))
        plt.close()

        # 可视化memory的分类结果
        memory_cls = torch.einsum("btc,nc->btn",memory, text_feats)
        memory_cls = memory_cls.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(memory_cls[idx].transpose(1,0),cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'memory_cls.png'))
        plt.close()


        # 可视化query和memory的相似度
        hs = outputs['hs'] # [dec_layers,b,num_queries,c]
        hs = hs[-1]
        hs = hs / hs.norm(dim=-1,keepdim=True)

        query_memory = torch.einsum("bqc,btc->bqt",hs,memory)
        query_memory = query_memory.cpu().detach().numpy()
        fig = plt.figure(figsize=(16,6))
        sns.heatmap(query_memory[idx],cmap="YlGnBu")
        plt.savefig(os.path.join(save_dir,'query_memory.png'))
        plt.close()

        
        print(target[idx]['video_name'])
        print(target[idx]['mask_labels'])
        print(target[idx]['semantic_labels'])
        print(target[idx]['segments'])


    # # 单独拿一个视频出来可视化
    # video_name = "v_"+"-VcxQ6i6Ejk"
    # vid_feat_path = os.path.join("/mnt/Datasets/ActivityNet13/CLIP_feature",video_name+".npy")
    # vid_feat_np = np.load(vid_feat_path,allow_pickle=True)
    # vid_feat = torch.from_numpy(vid_feat_np).to(args.device).float().contiguous()
    # vid_feat = vid_feat / vid_feat.norm(dim=1,keepdim=True)
    # vid_feat_cls = torch.einsum("tc,nc->tn", logit_scale*vid_feat, text_feats).softmax(-1)
    # vid_feat_cls = vid_feat_cls.cpu().detach().numpy()
    # fig = plt.figure(figsize=(16,6))
    # sns.heatmap(vid_feat_cls.transpose(1,0),cmap="YlGnBu")
    # plt.savefig(os.path.join(save_dir,video_name+'_CLS.png'))
    # plt.close()

# CUDA_VISIBLE_DEVICES=7 python debug.py --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --batch_size 16 --target_type "prompt" --model_name "ActivityNet13_CLIP_prompt_zs_v7_3" --num_queries 5 --enc_layers 2 --dec_layers 2 --enable_backbone --enable_classAgnostic

# CUDA_VISIBLE_DEVICES=7 python debug.py --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --batch_size 16 --target_type "prompt" --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_3" --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic
  