echo "16个实验后台执行开始, Compute Node: 209 ....."
#【base-17】--cls_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_30" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 3 --exp_logit_scale &
sleep 3
#【base-17】--cls_loss_coef 5 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_31" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 5 --exp_logit_scale &
sleep 3
#【base-17】--set_cost_class 3 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_32" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --set_cost_class 3 &
sleep 3
#【base-17】--set_cost_class 5 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_33" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --set_cost_class 5 &
sleep 3
#【base-17】--giou_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_34" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --giou_loss_coef 3 &
sleep 3
#【base-17】--bbox_loss_coef 6 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_35" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 6 &
sleep 3
#【base-17】--bbox_loss_coef 4 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_36" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 4 &
sleep 3
#【base-17】--bbox_loss_coef 3 --giou_loss_coef 1 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_37" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 3 --giou_loss_coef 1 &
sleep 3
#【base-17】--bbox_loss_coef 2 --giou_loss_coef 4 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_38" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 2 --giou_loss_coef 4 &
sleep 3
#【base-17】--cls_loss_coef 3 --bbox_loss_coef 2 --giou_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_39" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 3 --exp_logit_scale --bbox_loss_coef 2 --giou_loss_coef 3 &
sleep 3
#【base-17】--cls_loss_coef 1 --bbox_loss_coef 4 --giou_loss_coef 1 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_40" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 1 &
sleep 3
#【base-17】--cls_loss_coef 3 --bbox_loss_coef 1 --giou_loss_coef 2 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_41" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 3 --exp_logit_scale --bbox_loss_coef 1 --giou_loss_coef 2 &
sleep 3
#【base-17】--cls_loss_coef 2 --bbox_loss_coef 3 --giou_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_42" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 3 --giou_loss_coef 3 &
sleep 3
#【base-17】--cls_loss_coef 4 --bbox_loss_coef 3 --giou_loss_coef 4 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_43" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 4 --exp_logit_scale --bbox_loss_coef 3 --giou_loss_coef 4 &
sleep 3
#【base-17】--cls_loss_coef 6 --bbox_loss_coef 4 --giou_loss_coef 6 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_44" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 &
sleep 3
#【base-17】--cls_loss_coef 2 --bbox_loss_coef 4 --giou_loss_coef 2 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_45" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 2 &
wait
echo "16个实验后台执行结束, Compute Node: 209 ....."
