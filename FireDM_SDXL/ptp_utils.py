# The codes was modified from 
#   https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py
#   https://github.com/showlab/DatasetDM/blob/main/ptp_utils.py

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image
import cv2
from typing import Tuple
from tqdm import tqdm
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils import USE_PEFT_BACKEND

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h,:,:] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def save_images(images, num_rows=1, offset_ratio=0.02, out_put="./test_1.jpg"):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    
    pil_img = Image.fromarray(image_)
    pil_img.save(out_put)


def latent2image(vae, latents):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def init_latent(latent, batch_size, num_channels_latents, height, width, vae_scale_factor, scheduler_init_noise_sigma, generator, device, is_train=True):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latent = torch.randn(shape, generator=generator, layout=torch.strided) if latent is None else latent
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latent.expand(*shape).to(device)
    latents = latents * scheduler_init_noise_sigma if not is_train else latents
    return latent, latents

def init_time_ids(height, width, dtype):
    add_time_ids = [height, width, 0, 0, height, width] # original_size (H, W) + crops_coords_top_left (x: 0, y: 0) + target_size (H, W)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids

def diffusion_step(unet, scheduler, latents, prompt_embeds, add_text_embeds, add_time_ids, t, guidance_scale):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        return_dict=False,
    )[0]
    
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompts, tokenizer, tokenizer_2, text_encoder, text_encoder_2):
    tokenizers = [tokenizer, tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2]
    prompt_embeds_list = []
    
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}

@torch.no_grad()
def text2image(
    unet,
    vae,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    scheduler,
    prompt,
    controller,
    num_inference_steps,
    guidance_scale,
    generator,
    latent=None,
    vae_scale_factor=None,
    is_train=True,
):
    height = width = 1024
    vae_scale_factor = vae_scale_factor or 2 ** (len(vae.config.block_out_channels) - 1)
    batch_size = len(prompt)
    uncond_prompt = [""] * batch_size
    
    cond_embeds = encode_prompt(prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    text_embeddings, pooled_text_embeddings = cond_embeds["prompt_embeds"], cond_embeds["pooled_prompt_embeds"]
    
    uncond_embeds = encode_prompt(uncond_prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2)
    negative_embeddings, pooled_negative_embeddings = uncond_embeds["prompt_embeds"], uncond_embeds["pooled_prompt_embeds"]

    negative_add_time_ids = add_time_ids = init_time_ids(height, width, text_embeddings.dtype)

    prompt_embeds = torch.cat([negative_embeddings, text_embeddings], dim=0)
    add_text_embeds = torch.cat([pooled_negative_embeddings, pooled_text_embeddings], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    
    prompt_embeds = prompt_embeds.to(unet.device)
    add_text_embeds = add_text_embeds.to(unet.device)
    add_time_ids = add_time_ids.to(unet.device).repeat(batch_size, 1)

    scheduler.set_timesteps(num_inference_steps)

    latent, latents = init_latent(latent, batch_size, unet.config.in_channels, height, width, vae_scale_factor, scheduler.init_noise_sigma, generator, unet.device, is_train)
    
    time_range = scheduler.timesteps
    
    time_range = time_range[-1:] if is_train else time_range
    filter_list = time_range[-1:]
    
    for t in tqdm(time_range):
        controller.activate = True if t in filter_list else False
        latents = diffusion_step(unet, scheduler, latents, prompt_embeds, add_text_embeds, add_time_ids, t, guidance_scale)
    
    image = latent2image(vae, latents)
    
    return image, latent

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        
        def attn_processor_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            temb = cross_attention_kwargs.pop("temb", None) if cross_attention_kwargs else None
            scale = cross_attention_kwargs.pop("scale", 1.0) if cross_attention_kwargs else 1.0
            is_cross = encoder_hidden_states is not None
            
            args = () if USE_PEFT_BACKEND else (scale,)
            
            residual = hidden_states
            
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states, *args)
            value = self.to_v(encoder_hidden_states, *args)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            _ = controller(attention_probs, is_cross, place_in_unet=place_in_unet)
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            if isinstance(self.processor, AttnProcessor) or isinstance(self.processor, AttnProcessor2_0):
                return attn_processor_forward(self, hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
            else:
                return self.processor(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
            
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
            
    controller.num_att_layers = cross_att_count
