echo "7个实验后台执行开始, Compute Node: 211 ....."
# 【base-2】--lr_backbone 1e-2 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_12" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-2 &
sleep 3
# 【base-2】--lr_backbone 5e-3 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_13" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 5e-3 &
sleep 3
# 【base-5】--backbone_layers 2 ::
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_14" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 --backbone_layers 2 &
sleep 3
# 【base-5】--backbone_layers 3 ::
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_15" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 --backbone_layers 3 &
sleep 3
# 【base-5】--backbone_layers 4 ::
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_16" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 --backbone_layers 4 &
sleep 3
# 【base-5】--cls_loss_coef 1 ::
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_17" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 --cls_loss_coef 1 &
sleep 3
# 【base-5】--cls_loss_coef 3 ::
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_18" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 --cls_loss_coef 3 &
wait
echo "7个实验后台执行结束, Compute Node: 211 ....."

