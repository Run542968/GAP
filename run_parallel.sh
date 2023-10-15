echo "6个实验后台执行开始, Compute Node: 211 ....."
# DETR
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_1" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale &
sleep 30
# --actionness_loss
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_2" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --actionness_loss &
sleep 3
# --enable_classAgnostic
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_3" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# only DETR 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v7_1" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale &
sleep 30
# --actionness_loss 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v7_2" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --actionness_loss &
sleep 3
# --enable_classAgnostic 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v7_3" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --enable_classAgnostic &
wait
echo "6个实验后台执行结束, Compute Node: 211 ....."

