echo "11个实验后台执行开始, Compute Node: 211 ....."
# 【base-12】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_33" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-7】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_34" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss --semantic_vhead_type "Enc" --semantic_thead_type "MHA" --augment_prompt_type "attention" --subaction_version "v3" --lr_semantic_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-8】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_35" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss --semantic_vhead_type "None" --semantic_thead_type "MHA" --augment_prompt_type "attention" --subaction_version "v3" --lr_semantic_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-9】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_36" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss --semantic_vhead_type "Conv" --semantic_thead_type "MHA" --augment_prompt_type "attention" --subaction_version "v3" --lr_semantic_head 1e-6 --instance_loss_type "BCE" &
sleep 3
# 【base-10】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_37" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss --semantic_vhead_type "Enc" --semantic_thead_type "MHA" --augment_prompt_type "attention" --subaction_version "v3" --lr_semantic_head 1e-6 --instance_loss_type "BCE" &
sleep 3
# 【base-24】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_38" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-25】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_39" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-28】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_40" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v2 --semantic_vhead_type "Conv" --augment_prompt_type "single" --lr_semantic_head 1e-5 --instance_loss_type "BCE" &
sleep 3
# 【base-30】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=5 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_41" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v3 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-7 --lr_temporal_head 1e-7 --instance_loss_type "BCE" &
sleep 3
# 【base-31】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_42" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v3 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-6 --lr_temporal_head 1e-6 --instance_loss_type "BCE" &
sleep 3
# 【base-32】--instance_loss_type "BCE"
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v4_43" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --instance_loss_v3 --semantic_vhead_type "Enc" --augment_prompt_type "single" --lr_semantic_head 1e-5 --lr_temporal_head 1e-5 --instance_loss_type "BCE" &
wait
echo "11个实验后台执行结束, Compute Node: 211 ....."

