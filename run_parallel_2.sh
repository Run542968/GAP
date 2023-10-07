echo "5个实验后台执行开始, Compute Node: 211 ....."
# 【base-v2-7】 --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-7 :: 
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_25" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-7 &
sleep 3
# 【base-v2-7】 --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-6 :: 
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_26" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-6 &
sleep 3
# 【base-v2-7】 --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-6 :: 
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_27" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-6 &
sleep 3
# 【base-v2-7】 --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-5 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_28" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-5 &
sleep 3
# 【base-v2-7】 --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-5 :: 
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_29" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-5 &
wait
echo "5个实验后台执行结束, Compute Node: 211 ....."
