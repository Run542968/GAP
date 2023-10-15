echo "4个实验后台执行开始, Compute Node: 209 ....."
# 【base-8】--exclusive_loss --train_interval 1 --test_interval 1
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v6_32" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --actionness_loss --exclusive_loss --train_interval 1 --test_interval 1 &
sleep 3
# 【base-8】--exclusive_loss --exclusive_loss_coef 2
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v6_33" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --actionness_loss --exclusive_loss --exclusive_loss_coef 2 &
sleep 3
# 【base-8】--exclusive_loss --exclusive_loss_coef 3
CUDA_VISIBLE_DEVICES=3 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v6_34" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --actionness_loss --exclusive_loss --exclusive_loss_coef 3 &
sleep 3
# 【base-8】--exclusive_loss --exclusive_loss_coef 0.5
CUDA_VISIBLE_DEVICES=4 python main.py --model_name "Thumos14_CLIP_prompt_zs_8frame_v6_35" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 4 --norm_embed --exp_logit_scale --actionness_loss --exclusive_loss --exclusive_loss_coef 0.5 &
wait
echo "4个实验后台执行结束, Compute Node: 209 ....."

