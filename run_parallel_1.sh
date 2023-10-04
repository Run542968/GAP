echo "11个实验后台执行开始, Compute Node: 209 ....."
#【base-53】--batch_size 32 ::
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_67" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 &
sleep 3
#【base-53】--batch_size 64 ::
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_68" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 &
sleep 3
#【base-53】--slice_size 256 ::
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_69" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --slice_size 256 &
sleep 3
#【base-53】--slice_size 64 ::
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_70" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --slice_size 64 &
sleep 3
#【base-53】--slice_overlap 0.8 ::
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_71" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --slice_overlap 0.8 &
sleep 3
#【base-53】--slice_overlap 0.85 ::
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_72" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --slice_overlap 0.85 &
sleep 3
#【base-53】--slice_overlap 0.9 ::
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_73" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --slice_overlap 0.9 &
sleep 3
#【base-53】--inference_slice_overlap 0.1 ::
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_74" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.1 &
sleep 3
#【base-53】--inference_slice_overlap 0.15 ::
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_75" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.15 &
sleep 3
#【base-53】--inference_slice_overlap 0.2 ::
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_76" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.2 &
sleep 3
#【base-53】--inference_slice_overlap 0.3 ::
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_77" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 6 --exp_logit_scale --bbox_loss_coef 4 --giou_loss_coef 6 --set_cost_class 2 --set_cost_bbox 3 --set_cost_giou 2 --inference_slice_overlap 0.3 &
wait
echo "11个实验后台执行结束, Compute Node: 209 ....."

