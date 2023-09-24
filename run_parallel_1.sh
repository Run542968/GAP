echo "8个实验后台执行开始, Compute Node: 212 ....."
#【base-1】--eval_proposal :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_8" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal &
sleep 3
#【base-8】--enable_backbone --lr_backbone 1e-3 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_9" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 &
sleep 3
#【base-8】--cls_loss_coef 1 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_10" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal --cls_loss_coef 1 &
sleep 3
#【base-8】--cls_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_11" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-8】--dec_layers 3 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_12" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 3 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-8】--dec_layers 4 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_13" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-8】--enc_layers 1 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_14" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 1 --dec_layers 2 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-8】--enc_layers 3 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_15" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 3 --dec_layers 2 --eval_proposal --cls_loss_coef 3 &
wait
echo "8个实验后台执行结束, Compute Node: 212 ....."

