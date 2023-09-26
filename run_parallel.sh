echo "5个实验后台执行开始, Compute Node: 211 ....."
#【base-13】--dec_layers 3 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_10" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 3 --dec_layers 4 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-13】--dec_layers 4 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_11" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --eval_proposal --cls_loss_coef 3 &
sleep 3
#【base-13】--cls_loss_coef 1 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_12" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 1 &
sleep 3
#【base-13】--cls_loss_coef 4
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_13" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 4 &
sleep 3
#【base-13】--cls_loss_coef 5
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_14" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 5 &
wait
echo "5个实验后台执行结束, Compute Node: 211 ....."

