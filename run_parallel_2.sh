echo "6个实验后台执行开始, Compute Node: 211 ....."
#【base-1】--distillation_loss_coef 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_3" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 0.1 &
sleep 3
#【base-1】--distillation_loss_coef 0.5
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_4" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 0.5 &
sleep 3
#【base-1】--distillation_loss_coef 2
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_5" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 2 &
sleep 3
#【base-1】--distillation_loss_coef 3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_6" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 3 &
sleep 3
#【base-1】--distillation_loss_coef 4
CUDA_VISIBLE_DEVICES=6 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_7" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 4 &
sleep 3
#【base-1】--distillation_loss_coef 5
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v5_8" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --enable_backbone --lr_backbone 1e-2 --exp_logit_scale --segmentation_loss --semantic_vhead_type "None" --augment_prompt_type "single" --distillation_loss --distillation_loss_coef 5 &
wait
echo "6个实验后台执行结束, Compute Node: 211 ....."
