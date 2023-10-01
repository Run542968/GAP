echo "11个实验后台执行开始, Compute Node: 211 ....."
#【base-17】--num_queries 20 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_19" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 20 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--num_queries 25 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_20" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 25 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--num_queries 30 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_21" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 30 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--num_queries 35 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_22" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 35 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--num_queries 45 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_23" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 45 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--num_queries 50 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_24" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 50 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale &
sleep 3
#【base-17】--ROIalign_size 4 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_25" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --ROIalign_size 4 &
sleep 3
#【base-17】--ROIalign_size 8 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_26" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --ROIalign_size 8 &
sleep 3
#【base-17】--ROIalign_size 12 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_27" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --ROIalign_size 12 &
sleep 3
#【base-17】--ROIalign_size 20 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_28" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --ROIalign_size 20 &
sleep 3
#【base-17】--ROIalign_size 24 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_29" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 1 --exp_logit_scale --ROIalign_size 24 &
wait
echo "11个实验后台执行结束, Compute Node: 211 ....."

