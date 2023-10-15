echo "8个实验后台执行开始, Compute Node: 211 ....."
# 【base-15】--text_distillation_loss_coef 1
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_42" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --text_distillation_loss &
sleep 3
# 【base-15】--text_distillation_loss_coef 0.5
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_43" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --text_distillation_loss --text_distillation_loss_coef 0.5 &
sleep 3
# 【base-15】--text_distillation_loss_coef 3
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_44" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --text_distillation_loss --text_distillation_loss_coef 3 &
sleep 3
# 【base-15】--text_distillation_loss_coef 5
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_45" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --text_distillation_loss --text_distillation_loss_coef 5 &
sleep 3
# 【base-15】--exclusive_loss
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_46" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --exclusive_loss &
sleep 3
# 【base-15】--exclusive_loss --exclusive_loss_coef 0.5
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_47" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --exclusive_loss --exclusive_loss_coef 0.5 &
sleep 3
# 【base-15】--exclusive_loss --exclusive_loss_coef 3
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_48" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --exclusive_loss --exclusive_loss_coef 3 &
sleep 3
# 【base-15】--exclusive_loss --exclusive_loss_coef 5
CUDA_VISIBLE_DEVICES=7 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_49" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --exclusive_loss --exclusive_loss_coef 5 &
wait
echo "8个实验后台执行结束, Compute Node: 211 ....."

