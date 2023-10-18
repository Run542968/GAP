echo "12个实验后台执行开始, Compute Node: 211 ....."
#【base-1】--enable_refine
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_2" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 4 --dec_layers 4
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_3" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 6 --dec_layers 6
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_4" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 6 --dec_layers 6 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 2 --dec_layers 2
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_5" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 4 --dec_layers 1
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_6" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 1 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 1 --dec_layers 4
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_7" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 1 --dec_layers 4 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 3 --dec_layers 3
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_8" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 3 --dec_layers 3 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--enc_layers 1 --dec_layers 3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_9" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 1 --dec_layers 3 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--num_queries 30
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_10" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 30 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--num_queries 50
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_11" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 50 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--num_queries 60
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_12" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 60 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
#【base-2】--num_queries 100
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_13" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 100 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_refine &
sleep 3
# 【base-4】--enc_layers 4 --dec_layers 4
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_5" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 4 --dec_layers 4 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 1 --dec_layers 4
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_6" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 1 --dec_layers 4 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 1 --dec_layers 1
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_7" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 1 --dec_layers 1 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 6 --dec_layers 6
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_8" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 6 --dec_layers 6 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 2 --dec_layers 4
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_9" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 4 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 4 --dec_layers 3
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_10" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 4 --dec_layers 3 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 3 --dec_layers 1
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_11" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 3 --dec_layers 1 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
sleep 3
# 【base-4】--enc_layers 2 --dec_layers 5
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_v7_12" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 5 --eval_proposal --enable_backbone --lr_backbone 1e-2 --enable_refine &
wait
echo "12个实验后台执行结束, Compute Node: 211 ....."

