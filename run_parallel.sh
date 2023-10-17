echo "12个实验后台执行开始, Compute Node: 211 ....."
# 【base-3】--num_queries 5 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_4" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 5 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 10 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_5" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 10 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 15 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_6" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 15 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 20 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_7" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 20 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 30 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_8" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 30 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 50 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs50_8frame_v7_9" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 50 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 5
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_7" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 5 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 10
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_8" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 10 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 15
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_9" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 15 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 20
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_10" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 20 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 30
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_11" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 30 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
sleep 3
# 【base-3】--num_queries 50
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v7_12" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v7" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 50 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --enable_classAgnostic &
wait
echo "12个实验后台执行结束, Compute Node: 211 ....."

