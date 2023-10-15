echo "4个实验后台执行开始, Compute Node: 212 ....."
# 【base-15】--queryRelation_loss --queryRelation_loss_coef 2 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_55" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --queryRelation_loss --queryRelation_loss_coef 2 &
sleep 3
# 【base-15】--queryRelation_loss --queryRelation_loss_coef 3 :: 
CUDA_VISIBLE_DEVICES=1 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_56" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --queryRelation_loss --queryRelation_loss_coef 3 &
sleep 3
# 【base-15】--queryRelation_loss --queryRelation_loss_coef 4
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_57" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --queryRelation_loss --queryRelation_loss_coef 4 &
sleep 3
# 【base-15】--queryRelation_loss --queryRelation_loss_coef 5
CUDA_VISIBLE_DEVICES=2 python main.py --model_name "ActivityNet13_CLIP_prompt_zs_v6_58" --cfg_path "./config/ActivityNet13_CLIP_zs_75.yaml" --use_mlflow --save_result --prefix "v6" --batch_size 16 --target_type "prompt" --lr 1e-4 --epochs 100 --num_queries 5 --postprocess_type "class_agnostic" --postprocess_topk 100 --rescale_length 300 --enc_layers 2 --dec_layers 2 --norm_embed --cls_loss_coef 3 --giou_loss_coef 3 --actionness_loss --queryRelation_loss --queryRelation_loss_coef 5 &
wait
echo "4个实验后台执行结束, Compute Node: 212 ....."

