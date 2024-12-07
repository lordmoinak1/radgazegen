import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler#, ControlNetCondition

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy import ndimage


def preprocess_control_image(image_temp):
    image_temp = np.array(image_temp)
    image_temp = cv2.resize(image_temp, (512, 512))
    image_temp = Image.fromarray(image_temp)
    return image_temp

def test_radgazegen():
    base_model_path = "path/to/stable_diffusion_v1_5_mimic_50k/"

    controlnet_path_canny = "path/to/controlnet_egc2_canny/"
    controlnet_path_sobel = "path/to/controlnet_egc2_sobel/"
    controlnet_path_gl = "path/to/controlnet_egc2_gl/"
    controlnet_path_segmentation = "path/to/controlnet_egc2_segmentation/"
    controlnet_path_hva = "path/to/gazecontrolnet/"

    controlnet_canny = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")
    controlnet_sobel = ControlNetModel.from_pretrained(controlnet_path_sobel, torch_dtype=torch.float16).to("cuda")
    controlnet_gl = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16).to("cuda")
    controlnet_segmentation = ControlNetModel.from_pretrained(controlnet_path_segmentation, torch_dtype=torch.float16).to("cuda")
    controlnet_hva = ControlNetModel.from_pretrained(controlnet_path_hva, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, 
        controlnet=[
            controlnet_canny, 
            controlnet_sobel, 
            controlnet_gl, 
            controlnet_segmentation, 
            # controlnet_hva
            ], 
        safety_checker=None,  
        torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe = pipe.to("cuda:0")

    df = pd.read_csv('path/to/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        control_image_canny = load_image("path/to/eval_reflacx_canny/{}".format(name))
        control_image_sobel = load_image("path/to/eval_reflacx_sobel/{}".format(name))
        control_image_gl = load_image("path/to/eval_reflacx_gl/{}".format(name))
        control_image_segmentation = load_image("path/to/eval_reflacx_segmentation/{}".format(name))
        try:
            control_image_hva = load_image("path/to/eval_reflacx/{}".format(name))
        except:
            control_image_hva = load_image("path/to/eval_reflacx/{}".format(name_x))
        name_x = name
        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel),
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),
            # preprocess_control_image(control_image_hva),
            ]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, 
            controlnet_conditioning_scale=0.01,#[0.1, 0.1, 0.1, 0.1, 0.001],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("path/to/ablation_radcn/{}".format(name))

if __name__ == '__main__':
    flag = 0    

    test_gazecontrolnet()


