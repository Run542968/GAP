echo "9个实验后台执行开始, Compute Node: 209 ....."
# 【new change of segmentation_head】 --segmentation_loss :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_5" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss &
sleep 3
#【base-5】--segmentation_head_type "Conv" :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_6" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "Conv" &
sleep 3
#【base-5】--segmentation_head_type "MHA" :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_7" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "MHA" &
sleep 3
#【base-5】--segmentation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_8" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_loss_coef 0.1 &
sleep 3
#【base-5】--segmentation_loss_coef 0.5 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_9" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_loss_coef 0.5 &
sleep 3
#【base-6】--segmentation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_10" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "Conv" --segmentation_loss_coef 0.1 &
sleep 3
#【base-6】--segmentation_loss_coef 0.5 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_11" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "Conv" --segmentation_loss_coef 0.5 &
sleep 3
#【base-7】--segmentation_loss_coef 0.1 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_12" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "MHA" --segmentation_loss_coef 0.1 &
sleep 3
#【base-7】--segmentation_loss_coef 0.5 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v2_13" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --norm_embed --ROIalign_strategy "after_pred" --segmentation_loss --segmentation_head_type "MHA" --segmentation_loss_coef 0.5 &
wait
echo "9个实验后台执行结束, Compute Node: 209 ....."