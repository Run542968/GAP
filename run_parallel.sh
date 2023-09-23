echo "6个实验后台执行开始, Compute Node: 212 ....."
# --enable_backbone :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_2" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone &
sleep 3
# 【base-2】--backbone_layers 2 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_3" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone --backbone_layers 2 &
sleep 3
# 【base-2】--backbone_layers 3 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_4" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone --backbone_layers 3 &
sleep 3
# 【base-2】--lr_backbone 1e-4 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_5" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone --lr_backbone 1e-4 &
sleep 3
# 【base-5】--backbone_layers 2 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_6" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone --lr_backbone 1e-4 --backbone_layers 2 &
sleep 3
# 【base-5】--backbone_layers 3 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_binary_7" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 2 --binary --enable_backbone --lr_backbone 1e-4 --backbone_layers 3 &
wait
echo "6个实验后台执行结束, Compute Node: 212 ....."

