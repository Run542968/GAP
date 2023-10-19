echo "8个实验后台执行开始, Compute Node: 211 ....."
#【base-17】--dec_layers 4
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_20" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--dec_layers 3
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_21" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 3 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--dec_layers 2
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_22" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--enc_layers 3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_23" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 3 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--enc_layers 4
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_24" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--num_queries 20
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_25" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 20 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--num_queries 30
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_26" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 30 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_injection &
sleep 3
#【base-17】--num_queries 50
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_v7_27" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --prefix "v7" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 50 --enc_layers 2 --dec_layers 5 --eval_proposal --actionness_loss_coef 3 --enable_injection &
wait
echo "8个实验后台执行结束, Compute Node: 211 ....."

