CUDA_VISIBLE_DEVICES=0 python generate.py --grounding_ckpt './checkpoint/Train/latest_checkpoint.pth' --n_each_class 1000 --outdir './Results/image_mask/' --thread_num 1 --H 512 --W 512 --config ./config/VOC/Sematic.yaml

