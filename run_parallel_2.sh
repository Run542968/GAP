echo "8个实验后台执行开始, Compute Node: 212 ....."
# --enable_backbone --lr_backbone 1e-5 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_4" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-5 &
sleep 3
# --enable_backbone --lr_backbone 1e-3 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_5" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-3 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --backbone_layers 2 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_6" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --backbone_layers 2 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --backbone_layers 3 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_7" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --backbone_layers 3 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --cls_loss_coef 1 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_8" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --cls_loss_coef 1 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --cls_loss_coef 3 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_9" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --cls_loss_coef 3 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --giou_loss_coef 1 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_10" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --giou_loss_coef 1 &
sleep 3
# --enable_backbone --lr_backbone 1e-4 --giou_loss_coef 3 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_zs_75_binary_11" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-4 --giou_loss_coef 3 &
wait
echo "8个实验后台执行结束, Compute Node: 212 ....."
