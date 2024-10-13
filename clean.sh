git commit版本是main


delete:
--adapterCLS_loss
--refine_actionness_loss
--distillation_loss




Thumos14
1. baseline :: 29.6086
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v9_1" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss_coef 3 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic
2. baseline + refine :: 34.9199
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v9_2_refineCat" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss_coef 3 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic --enable_refine --refine_drop_saResidual --refine_cat_type 'sum'
3. baseline + actionness + refine :: 32.8686
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v9_v8_4_refineSum" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --save_result --batch_size 16 --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --actionness_loss_coef 3 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic --enable_refine --refine_drop_saResidual --salient_loss --salient_loss_coef 3 --refine_cat_type 'sum'


test full method:
CUDA_VISIBLE_DEVICES=0 python test.py --model_name "Train_Thumos14_baseline_refine_actionness" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --batch_size 16 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic --enable_refine --refine_drop_saResidual --salient_loss


ActivityNet1.3
1. baseline :: 28.140
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v8_61" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --target_type "prompt" --lr 5e-5 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic --actionness_loss_coef 1 --giou_loss_coef 1
2. baseline + refine
3. baseline + actionness + refine :: 32.051
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v8_38" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --save_result --batch_size 16 --target_type "prompt" --lr 5e-5 --epochs 100 --num_queries 30 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --norm_embed --exp_logit_scale --proposals_weight_type "after_softmax" --enable_classAgnostic --actionness_loss_coef 3 --enable_refine --refine_drop_saResidual --salient_loss


