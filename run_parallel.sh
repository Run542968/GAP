echo "7个实验后台执行开始, Compute Node: 209 ....."
#【base-88】--set_cost_class 2 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_99" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 2 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--set_cost_class 3 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_100" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 3 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--set_cost_class 1 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_101" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 1 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--set_cost_class 4 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_102" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 2 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 4 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--cls_loss_coef 3 --bbox_loss_coef 5 --giou_loss_coef 2 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_103" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 3 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 2 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--set_cost_class 1 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_104" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 2 --set_cost_bbox 5 --set_cost_giou 2 &
sleep 3
#【base-88】--set_cost_class 4 --set_cost_bbox 5 --set_cost_giou 2 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_105" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 4 --exp_logit_scale --bbox_loss_coef 5 --giou_loss_coef 2 --set_cost_class 2 --set_cost_bbox 5 --set_cost_giou 2 &
wait
echo "7个实验后台执行结束, Compute Node: 209 ....."

