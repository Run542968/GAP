from .clip import *

def build_text_encoder(args,device):
    if args.CLIP_model_name in clip.available_models():
        model_path = args.CLIP_model_name
        model, _ = clip.load(str(model_path), device=device)
        model.eval()
        encode_text = model.encode_text
        logit_scale = model.logit_scale
    else:
        raise NotImplementedError(f'Model {args.CLIP_model_name} not found')
    return encode_text, logit_scale 