echo "7个实验后台执行开始, Compute Node: 211 ....."
#【base-7】--distillation_loss_coef 0.1 --classification_loss_coef 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_11" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --distillation_loss --distillation_loss_coef 0.1 --classification_loss_coef 0.1 &
sleep 3
#【base-7】--distillation_loss_coef 0.1 --classification_loss_coef 0.01
CUDA_VISIBLE_DEVICES=0 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_12" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --distillation_loss --distillation_loss_coef 0.1 --classification_loss_coef 0.01 &
sleep 3
#【base-7】--distillation_loss_coef 0.1 --classification_loss_coef 2
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_13" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --distillation_loss --distillation_loss_coef 0.1 --classification_loss_coef 2 &
sleep 3
#【base-1】--classification_loss_coef 0.1
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_14" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --classification_loss_coef 0.1 &
sleep 3
#【base-1】--classification_loss_coef 0.01
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_15" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --classification_loss_coef 0.01 &
sleep 3
#【base-1】--classification_loss_coef 2
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_16" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --classification_loss_coef 2 &
sleep 3
#【base-1】--classification_loss_coef 3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v5_17" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v5" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --cls_loss_coef 3 --classification_loss --classification_loss_coef 3 &
wait
echo "7个实验后台执行结束, Compute Node: 211 ....."

    