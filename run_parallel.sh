echo "7个实验后台执行开始, Compute Node: 212 ....."
# 【base-1】--num_queries 5 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs50_v2_3" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed &
sleep 60
# 【base-3】--enable_backbone --lr_backbone 1e-2 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs50_v2_4" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 &
sleep 60
# 【base-1】--num_queries 5 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v2_3" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed &
sleep 60 
# 【base-3】--enable_backbone --lr_backbone 1e-2 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v2_4" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 &
sleep 60
# 【base-2】--lr_backbone 1e-2 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_zs50_binary_4" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-2 &
sleep 60
# 【base-2】--lr_backbone 1e-1 --eval_proposal :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_zs50_binary_5" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --eval_proposal --enable_backbone --lr_backbone 1e-1 &
sleep 60
# 【base-4】 --dec_layers 3 --cls_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_zs50_binary_6" --cfg_path "./config/ActivityNet13_CLIP_zs_50.yaml" --use_mlflow --save_result --batch_size 16 --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 3 --eval_proposal --enable_backbone --lr_backbone 1e-2 --cls_loss_coef 3 &
wait
echo "7个实验后台执行结束, Compute Node: 212 ....."

