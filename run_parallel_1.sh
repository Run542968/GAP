echo "7个实验后台执行开始, Compute Node: 211 ....."
#【base-2】--dec_layers 4 --cls_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v2_14" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --ROIalign_strategy "after_pred" --cls_loss_coef 3 
wait
echo "7个实验后台执行结束, Compute Node: 211 ....."

