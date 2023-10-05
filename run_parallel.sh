echo "6个实验后台执行开始, Compute Node: 211 ....."
#【base-53】--inference_slice_overlap 0.4 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_127" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.4 &
sleep 3
#【base-53】--inference_slice_overlap 0.5 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_128" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.5 &
sleep 3
#【base-53】--inference_slice_overlap 0.6 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_129" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.6 &
sleep 3
#【base-53】--inference_slice_overlap 0.7 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_130" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.7 &
sleep 3
#【base-53】--inference_slice_overlap 0.8 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_131" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.8 &
sleep 3
#【base-53】--inference_slice_overlap 0.9 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_132" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.9 &
wait
echo "6个实验后台执行结束, Compute Node: 211 ....."

