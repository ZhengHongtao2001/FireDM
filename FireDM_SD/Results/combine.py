import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
coco_category_to_id_v1 = { 'fire':0}
ADE20K_150_CATEGORIES = [
    {"color": [250, 250, 250], "id": 0, "isthing": 0, "name": "background"},
    {"color": [106,90,205], "id": 1, "isthing": 0, "name": "fire"}
    ]



def _get_ade20k_full_meta():
    stuff_ids = [k["id"] for k in ADE20K_150_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"].split(",")[0] for k in ADE20K_150_CATEGORIES]
    
    color = [k["color"] for k in ADE20K_150_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "color":color
    }
    return ret


def register_all_ade20k_full(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    meta = _get_ade20k_full_meta()
    for name, dirname in [("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors = meta["color"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_full(_root)


metadata = MetadataCatalog.get("sem_seg"
        )
# small text
# image_car_93258


# /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former    ./VOC_Multi_Attention_car_GPT3/train_image/
root_image = "./DataDiffusion/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/Image"


# ground_truth
# root_mask = "./DiffMask_VOC/VOC_Multi_Attention_bird_sub_30000_NoClipRetrieval/refine_gt_crf_0.6/"
root_mask = "./DataDiffusion/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/Mask"

vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)


#vis_list = [i for i in vis_list if i.replace("jpg","png") in mask_lsit]
# vis_list = [i for i in vis_list if i.replace("jpg","png") in root_mask_check_list]

    
for image_pa in vis_list[:300]:
    

    print(image_pa)

    mask_path = os.path.join(root_mask,image_pa)
    
    # mask_path = os.path.join(root_mask,image_pa.replace("jpg","png"))
    image_path = os.path.join(root_image,image_pa)
    
    if not os.path.isfile(mask_path):
        continue
        
    print(mask_path)
    mask = cv2.imread(mask_path)[:,:,0]
    image = cv2.imread(image_path)

    #BGR转RGB，方法2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    
    # mask = mask
    print(mask.shape)

    print(np.unique(mask))
#     mask[mask==19]=2
    h,w = image.shape[:2]

    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(
                        mask, alpha=0.5
                    )

    vis_output.save("./DataDiffusion/Vis/{}".format(image_pa))

