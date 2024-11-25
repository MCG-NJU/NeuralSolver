from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler, StableDiffusionPipeline, DDIMScheduler
import torch
import os

# # 加载预训练的stable diffusion模型
# model_id = "/data/songtianhui.sth/models/stable-diffusion-v1-5/"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
step = 9
model_path = "/home/shuaiou.ws/fasterSolver/t2i_vis/PixArt-XL-2-256x256/"
prompt_json_path = "/home/shuaiou.ws/flowdcn-t2i/data/coco/coco_val_captions.json"
save_dir = f"/home/shuaiou.ws/fasterSolver/t2i_vis/pixart-sigma-256-{step}steps"

import torch
import json
import re
from PIL import Image
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from vp_scheduling import NeuralSolver
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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
prompts_list5000 = prompts_list[:5000]
prompt_list = []
for prompts in prompts_list5000:
    prompt = prompts[0]
    prompt_list.append(prompt)

# mkdir for images
for save_type in ["dpm", "unipc", "dss"]:
    if not os.path.exists(f"{save_dir}/{save_type}"):
        os.makedirs(f"{save_dir}/{save_type}")

bsz = 16
bnum = len(prompts_list)//bsz + 1

for bid in range(bnum):
    prompts = prompt_list[bid*bsz:(bid+1)*bsz]
    pipe.scheduler = dpmsolver
    dpmimages:Image = pipe(prompts, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=1.5).images
    pipe.scheduler = unipcsolver
    unipcimages:Image = pipe(prompts, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=1.5).images
    pipe.scheduler = NeuralSolver(num_train_timesteps=1000)
    dssimages:Image = pipe(prompts, num_inference_steps=step, generator=torch.Generator(0), guidance_scale=1.5).images
    for prompt, dpmimage, unipcimage, dssimage in zip(prompts, dpmimages, unipcimages, dssimages):
        # save images
        dpmimage.save(f"{save_dir}/dpm/{remove_invalid_chars(prompt)}.png")
        unipcimage.save(f"{save_dir}/unipc/{remove_invalid_chars(prompt)}.png")
        dssimage.save(f"{save_dir}/dss/{remove_invalid_chars(prompt)}.png")
    print(bid*bsz)







