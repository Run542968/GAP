echo "8个实验后台执行开始, Compute Node: 211 ....."
# 【base-1】--lr_semantic_head 1e-8 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_15" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-8 &
sleep 3
# 【base-1】--lr_semantic_head 1e-9 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_16" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-9 &
sleep 3
# 【base-14】--subaction_version "v2" :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_17" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --subaction_version "v2" &
sleep 3
# 【base-14】--subaction_version "v3" ::
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_18" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --subaction_version "v3" &
sleep 3
# 【base-14】--instance_loss_coef 0.1 ::
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_19" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --instance_loss_coef 0.1 &
sleep 3
# 【base-14】--instance_loss_coef 5 ::
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_20" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --instance_loss_coef 5 &
sleep 3
# 【base-14】--instance_loss_coef 10 ::
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_21" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --instance_loss_coef 10 &
sleep 3
# 【base-14】--instance_loss_coef 20 ::
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v2_22" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "MHA" --lr_semantic_head 1e-7 --instance_loss_coef 20 &
wait
echo "8个实验后台执行结束, Compute Node: 211 ....."

