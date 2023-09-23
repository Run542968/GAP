echo "3个实验后台执行开始, Compute Node: 212 ....."
# 【base-44】--segmentation_loss --enable_backbone :: 
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_80" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --segmentation_loss --enable_backbone &
sleep 3
# 【base-44】--segmentation_loss --enable_backbone --backbone_layers 2 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_81" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --segmentation_loss --enable_backbone --backbone_layers 2 &
sleep 3
# 【base-44】--segmentation_loss --enable_backbone --backbone_layers 3 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_82" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --segmentation_loss --enable_backbone --backbone_layers 3 &
wait
echo "3个实验后台执行结束, Compute Node: 212 ....."

