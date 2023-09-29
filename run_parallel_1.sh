echo "11个实验后台执行开始, Compute Node: 211 ....."
#【base-6】--lr_semantic_head 1e-5 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_23" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-5 &
sleep 3
#【base-6】--lr_semantic_head 1e-6 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_24" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-6 &
sleep 3
#【base-6】--lr_semantic_head 1e-7 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_25" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-7 &
sleep 3
#【base-6】--subaction_version "v2" :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_26" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --subaction_version "v2" &
sleep 3
#【base-6】--subaction_version "v3" :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_27" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --subaction_version "v3" &
sleep 3
#【base-6】--lr_semantic_head 1e-5 --subaction_version "v2" :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_28" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-5 --subaction_version "v2" &
sleep 3
#【base-6】--lr_semantic_head 1e-6 --subaction_version "v2" :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_29" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-6 --subaction_version "v2" &
sleep 3
#【base-6】--lr_semantic_head 1e-7 --subaction_version "v2" :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_30" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-7 --subaction_version "v2" &
sleep 3
#【base-6】--lr_semantic_head 1e-5 --subaction_version "v3" :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_31" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-5 --subaction_version "v3" &
sleep 3
#【base-6】--lr_semantic_head 1e-6 --subaction_version "v3" :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_32" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-6 --subaction_version "v3" &
sleep 3
#【base-6】--lr_semantic_head 1e-7 --subaction_version "v3" :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_description_zs50_8frame_v3_33" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "description" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --cls_loss_coef 1 --instance_loss --instance_head_type "Conv" --lr_semantic_head 1e-7 --subaction_version "v3" &
wait
echo "11个实验后台执行结束, Compute Node: 211 ....."

