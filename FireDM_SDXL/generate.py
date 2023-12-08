import os
import yaml
import random
import argparse
import heapq

import cv2
import numpy as np
from scipy.special import softmax

import torch
import torch.nn.functional as F

from detectron2.structures import Boxes, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.schedulers import EulerDiscreteScheduler 
from tqdm import tqdm

from ptp_utils import text2image, register_attention_control, save_images, encode_prompt
from model.unet import get_feature_dic, clear_feature_dic, ProxyUNet2D
from model.segment.transformer_decoder_semantic import SegDecoderOpenWord
from dataset import *


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

class AttentionStore:
    def __init__(self):
        self.num_att_layers = -1
        self.cur_step = 0 
        self.cur_att_layer = 0
        self.step_store = self._get_empty_store()
        self.attention_store = {}
        self.activate = True
    
    def reset(self):
        self.cur_step = 0 
        self.cur_att_layer = 0
        self.step_store = self._get_empty_store()
        self.attention_store = {}
    
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self._get_empty_store()

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0 # reset
            self.cur_step += 1 if self.activate else 0
            self.between_steps()
        return attn
    
    def _get_empty_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [],  "mid_self": [],  "up_self": []}


def freeze_params(params):
    for param in params:
        param.requires_grad = False
        

def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    for i in range(1,semseg.shape[0]):
        if (semseg[i]*(semseg[i]>0.5)).sum()<5000:
            semseg[i] = 0

    return semseg


def instance_inference(mask_cls, mask_pred,class_n = 2,test_topk_per_image=20,query_n = 100):
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(class_n , device=mask_cls.device).unsqueeze(0).repeat(query_n, 1).flatten(0, 1)
    # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // class_n
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
#     print(topk_indices)
    mask_pred = mask_pred[topk_indices]


    result = Instances(image_size)
    # mask (before sigmoid)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result


classes = {
    0: 'background',
    1: 'fire',
}

name_to_idx = {
    'background':0,
    'fire':1,
}


classes_check = {
    0: [],
    1: ['fire'],
}


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def sub_processor(pid , opt):
    torch.cuda.set_device(pid)
    print('processor %d' % pid)
    seed_everything(opt.seed)
    
    cfg = dict2obj(yaml.load(open(opt.config), Loader=yaml.FullLoader))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # ===================load model=====================
    sd_xl_path = "../models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
    
    tokenizer = CLIPTokenizer.from_pretrained(sd_xl_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sd_xl_path, subfolder="tokenizer_2")
    

    vae = AutoencoderKL.from_pretrained(sd_xl_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(sd_xl_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(sd_xl_path, subfolder="text_encoder_2")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(sd_xl_path, subfolder="scheduler")
    
    unet: UNet2DConditionModel = ProxyUNet2D.from_pretrained(sd_xl_path, subfolder="unet")
    
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    unet = unet.to(device)

    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(text_encoder_2.parameters())
    freeze_params(unet.parameters())

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    unet.eval()
    
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    num_diffusion_steps = 50
    guidance_scale = 5
    # ===================load model=====================
    
    # ===================load seg model=====================
    seg_model = SegDecoderOpenWord(num_classes=cfg.SEG_Decoder.num_classes, num_queries=cfg.SEG_Decoder.num_queries).to(device)
    base_weights = torch.load(opt.grounding_ckpt, map_location="cpu")
    print('load weight:', opt.grounding_ckpt)
    seg_model.load_state_dict(base_weights, strict=True)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    Image_path = os.path.join(outpath, "Image")
    os.makedirs(Image_path, exist_ok=True)
        
    Mask_path = os.path.join(outpath, "Mask")
    os.makedirs(Mask_path, exist_ok=True)
    
    number_per_thread_num = int(int(opt.n_each_class)/opt.thread_num)
    seed = pid * (number_per_thread_num *2)

    controller = AttentionStore()
    register_attention_control(unet, controller)
    
    test_prompts = [
        "There was a fire on boat (flame :1.33). 8k, Details, press photos"
        "There was a fire on mine (flame :1.33). 8k, Details, press photos"
        "A (fierce fire: 1.33) is raging in the building. 8k, finely detail, news photo",
         "A (fierce fire: 1.33) is raging in the forest, turning trees into ashes. 8k, finely detail, news photo",
         "The (fire: 1.33) is spreading under the push of the wind, with smoke filling the air. 8k, finely detail, news photo",
         "The (firelight: 1.33) illuminates the night sky, reflecting the tragic scene of the forest. 8k, finely detail, news photo",
         "A car suddenly caught (fire: 1.33) on the highway, with flames bursting out of the hood. 8k, finely detail, news photo",
         "The (fire: 1.33) spread rapidly, engulfing the entire vehicle. 8k, finely detail, news photo",
         "The (firelight: 1.33) illuminates the surrounding environment, with smoke diffusing in the air. 8k, finely detail, news photo",
         "Firefighters are fighting a (big fire: 1.33), with (flames: 1.33) raging in the building. 8k, finely detail, news photo",
         "Firefighters aim their water guns at the source of the (fire: 1.33), with water mist and flames intertwined. 8k, finely detail, news photo",
         "Despite the (fierce fire: 1.33), the firefighters do not flinch and are determined to extinguish the fire source. 8k, finely detail, news photo",
    ]

    for idx in classes:
        if idx==0:
            continue
                
        class_target = classes[idx]
    
        with torch.no_grad():
            for step in tqdm(range(100)):
                
                torch.cuda.empty_cache()
                unet = unet.to(device)
                
                # clear all features and attention maps
                clear_feature_dic()
                controller.reset()
            
                g_cpu = torch.Generator()
                g_cpu.manual_seed(seed + step)
                
                prompts = [random.choice(test_prompts)]

                prompts = list(prompts) if isinstance(prompts, tuple) else prompts
                print("prompts:", prompts)
                
                generated_images, x_t = text2image(unet, vae, tokenizer, text_encoder, tokenizer_2, text_encoder_2, noise_scheduler, prompts, controller, num_inference_steps=num_diffusion_steps, guidance_scale=guidance_scale, generator=g_cpu, vae_scale_factor=vae_scale_factor, is_train=False)
                save_images(generated_images, out_put="{}/{}_{}.png".format(Image_path, class_target, step))
                
                # free cuda memory 
                unet = unet.to("cpu")
                torch.cuda.empty_cache()
            
                full_arr = np.zeros((21, 1024, 1024), np.float32)
                full_arr[0] = 0.5
                
                for idxx in classes:
                    if idxx==0:
                        continue

                    class_name = classes[idxx]
                    if class_name not in classes_check[idx]:
                        continue   
                        
                     # train segmentation
                    query_text = class_name
                    # text_input = tokenizer(query_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    # text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                    
                    text_embeddings = encode_prompt(query_text, tokenizer, tokenizer_2, text_encoder, text_encoder_2)['prompt_embeds']
                    text_embeddings = text_embeddings.to(device)

                    if text_embeddings.size()[1] > 1:
                        text_embeddings = torch.unsqueeze(text_embeddings.mean(1),1)

                    diffusion_features = get_feature_dic()
                    outputs = seg_model(diffusion_features, controller, prompts, tokenizer, text_embeddings)
                    
                    mask_cls_results, mask_pred_results = outputs["pred_logits"], outputs["pred_masks"]
                    mask_pred_results = F.interpolate(mask_pred_results, size=(1024, 1024), mode="bilinear", align_corners=False)
                    
                    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                        instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result,class_n = cfg.SEG_Decoder.num_classes,test_topk_per_image=3,query_n = cfg.SEG_Decoder.num_queries)
                            
                        pred_masks = instance_r.pred_masks.cpu().numpy().astype(np.uint8)
                        scores = instance_r.scores 

                        topk_idx = heapq.nlargest(1, range(len(scores)), scores.__getitem__)
                        mask_instance = (pred_masks[topk_idx[0]]>0.5 * 1).astype(np.uint8) 
                        full_arr[idxx] = np.array(mask_instance)

                full_arr = softmax(full_arr, axis=0)
                mask = np.argmax(full_arr, axis=0)
                
                cv2.imwrite("{}/{}_{}.png".format(Mask_path, class_target, step), mask)

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="the prompt to render"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="?",
        default="lion",
        help="the category to ground"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./DataDiffusion/VOC/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help="number of threads",
    )
    
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--prompt_root",
        action='store_true',
        help="uses prompt",
        default="./dataset/Prompts_From_GPT/voc2012"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_each_class",
        type=int,
        default=20,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=1024,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--grounding_ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    
    import multiprocessing as mp
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []
    print('Start Generation')
    for i in range(opt.thread_num):
        p = mp.Process(target=sub_processor, args=(i, opt))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)

if __name__ == "__main__":
    main()
