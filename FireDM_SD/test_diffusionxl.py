#from diffusers import DiffusionPipeline
import torch
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "/zhenghongtao/FireDM/models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]
# image
# pipe = DiffusionPipeline.from_pretrained("/zhenghongtao/FireDM/models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.to("cuda")

# # if using torch < 2.0
# # pipe.enable_xformers_memory_efficient_attention()

# prompt = "Flames are burning in the forest."

# images = pipe(prompt=prompt).images[0]
image.save("output_image.jpg")