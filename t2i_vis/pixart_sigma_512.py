from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler, StableDiffusionPipeline, DDIMScheduler
import torch

# # 加载预训练的stable diffusion模型
# model_id = "/data/songtianhui.sth/models/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

model_path = "/data/songtianhui.sth/models/PixArt-XL-2-512x512/"
prompt_json_path = "/home/shuaiou.ws/flowdcn-t2i/data/coco/coco_val_captions.json"

import torch
import json
import re
from PIL import Image
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from vp_scheduling import NeuralSolver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16
def remove_invalid_chars(text):
    pattern = r'[^\w\s]'    # 只保留中文、数字、字母、空格
    return re.sub(pattern, '', text)


pipe = PixArtSigmaPipeline.from_pretrained(
    model_path,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
dpmsolver = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
unipcsolver = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# dssolver = NeuralSolver(num_train_timesteps=1000)
# pipe.enable_model_cpu_offload()
prompts_list:list = json.load(open(prompt_json_path))
prompts_list = prompts_list[2:150]
for prompts in prompts_list:
    prompt = prompts[0]
    dpmimages = []
    unipcimages = []
    dssimages = []
    for step in range(5, 11):
        pipe.scheduler = dpmsolver
        dpmimage:Image = pipe(prompt, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=2).images[0]
        pipe.scheduler = unipcsolver
        unipcimage:Image = pipe(prompt, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=2).images[0]
        pipe.scheduler = NeuralSolver(num_train_timesteps=1000)
        dssimage:Image = pipe(prompt, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=2).images[0]
        dpmimages.append(dpmimage)
        unipcimages.append(unipcimage)
        dssimages.append(dssimage)
    dpm_horizontal_concat = Image.new("RGB", (dpmimages[0].width*len(dpmimages), dpmimages[0].height))
    unipc_horizontal_concat = Image.new("RGB", (unipcimages[0].width*len(unipcimages), unipcimages[0].height))
    dss_horizontal_concat = Image.new("RGB", (dssimages[0].width*len(dssimages), dssimages[0].height))
    for i, image in enumerate(dpmimages):
        dpm_horizontal_concat.paste(image, (i*dpmimages[0].width, 0))
        unipc_horizontal_concat.paste(unipcimages[i], (i*unipcimages[0].width, 0))
        dss_horizontal_concat.paste(dssimages[i], (i*dssimages[0].width, 0))
    vertical_concat = Image.new("RGB", (dpm_horizontal_concat.width, dpm_horizontal_concat.height+unipc_horizontal_concat.height+dss_horizontal_concat.height))
    vertical_concat.paste(dpm_horizontal_concat, (0, 0))
    vertical_concat.paste(unipc_horizontal_concat, (0, dpm_horizontal_concat.height))
    vertical_concat.paste(dss_horizontal_concat, (0, dpm_horizontal_concat.height + unipc_horizontal_concat.height))
    # vertical_concat.save(f"./pixart-sigma-512/{remove_invalid_chars(prompt)}.jpg")
    break


