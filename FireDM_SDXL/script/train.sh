
# CUDA_LAUNCH_BLOCKING=4 
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/train_semantic_voc.py --save_name Train_5_images_t1_attention_transformer_VOC_10layers --config ./config/VOC/VOC_Sematic_Seg_5Images.yaml --image_limitation 5

# Train
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python train.py --save_name Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images --config ./config/VOC/VOC_Sematic_Seg_5Images.yaml --image_limitation 118
