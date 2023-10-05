echo "4个实验后台执行开始, Compute Node: 211 ....."
#【base-3】--enc_layers 4 --cls_loss_coef 4 --cls_loss_coef 4 :: 
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_15" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 4 --dec_layers 4 --eval_proposal --cls_loss_coef 4 &
sleep 3
#【base-13】 --postprocess_topk 5 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_20" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 5 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 4 &
sleep 3
#【base-15】 --postprocess_topk 5 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_25" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 5 --num_queries 40 --enc_layers 4 --dec_layers 4 --eval_proposal --cls_loss_coef 4 &
sleep 3
#【base-16】 --postprocess_topk 5 :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_zs50_8frame_binary_30" --cfg_path "./config/Thumos14_CLIP_zs_50_8frame.yaml" --use_mlflow --save_result --batch_size 16  --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 5 --num_queries 40 --enc_layers 2 --dec_layers 4 --eval_proposal --cls_loss_coef 2 &
wait
echo "4个实验后台执行结束, Compute Node: 211 ....."

    