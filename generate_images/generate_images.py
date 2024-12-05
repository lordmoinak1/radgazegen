import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def test_gazecontrolnet():
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_50k/"
    controlnet_path_canny = "/path/to/controlnet_egc2_canny/"
    controlnet_path_sobel = "/path/to/controlnet_egc2_sobel/"
    controlnet_path_gl = "/path/to/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/path/to/controlnet_egc2_segmentation/"
    controlnet_path_hva = "/path/to/radgazegen/"

    controlnet_canny = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")
    controlnet_sobel = ControlNetModel.from_pretrained(controlnet_path_sobel, torch_dtype=torch.float16).to("cuda")
    controlnet_gl = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16).to("cuda")
    controlnet_segmentation = ControlNetModel.from_pretrained(controlnet_path_segmentation, torch_dtype=torch.float16).to("cuda")
    controlnet_hva = ControlNetModel.from_pretrained(controlnet_path_hva, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_canny, controlnet_sobel, controlnet_gl, controlnet_segmentation, controlnet_hva], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/temp/reports/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        name = name.split('/')[1]

        control_image_canny = load_image("/path/to/chexpert_multicontrolnet/canny/{}.png".format(name))
        control_image_sobel = load_image("/path/to/chexpert_multicontrolnet/sobel/{}.png".format(name))
        control_image_gl = load_image("/path/to/chexpert_multicontrolnet/gl/{}.png".format(name))
        control_image_segmentation = load_image("/path/to/chexpert_multicontrolnet/segmentation/{}.png".format(name))
        control_image_hva = load_image("/path/to/chexpert/hypotheses/{}.png".format(name))

        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel), 
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),
            preprocess_control_image(control_image_hva),]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.01, #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/path/to/chexpert/radgazegen/{}.png".format(name))

def test_sd():
    model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df = pd.read_csv('/path/to/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        name = name.split('/')[1]

        generator = torch.manual_seed(3407)
        image = pipe(text, num_inference_steps=50, generator=generator).images[0]

        image.save("/path/to/sd/{}.png".format(name))

def test_roentgen():
    model_path = "/path/to/roentgen/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df = pd.read_csv('/path/to/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        name = name.split('/')[1]

        generator = torch.manual_seed(3407)
        image = pipe(text, num_inference_steps=50, generator=generator).images[0]

        image.save("/path/to/roentgen/{}.png".format(name))

def test_t2i_adapter():
    adapter = T2IAdapter.from_pretrained("/path/to/t2i_xl_base1.0_egc1_15k", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    df = pd.read_csv('/path/to/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        image = load_image("/path/to/CheXpert/{}".format(name))
        
        name = name.split('/')[1]

        resized_image = np.array(image)
        resized_image = cv2.resize(resized_image, (512, 512))

        control_image = np.array(image)
        control_image = cv2.Canny(control_image, 50, 100)
        control_image = control_image[:, :, None]
        control_image = np.concatenate([control_image, control_image, control_image], axis=2)
        control_image = cv2.resize(control_image, (512, 512))
        control_image = Image.fromarray(control_image)

        generator = torch.manual_seed(3407)

        image = pipe(text, image=control_image, generator=generator, num_inference_steps=10).images[0]
        image.save("/path/to/t2i/{}.png".format(name))

def test_controlnet():
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/path/to/controlnet_egc1_sdv1-5_canny/"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/path/to/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        image = load_image("/path/to/CheXpert/{}".format(name))
        
        name = name.split('/')[1]

        resized_image = np.array(image)
        resized_image = cv2.resize(resized_image, (512, 512))

        control_image = np.array(image)
        control_image = cv2.Canny(control_image, 50, 100)
        control_image = control_image[:, :, None]
        control_image = np.concatenate([control_image, control_image, control_image], axis=2)
        control_image = cv2.resize(control_image, (512, 512))
        control_image = Image.fromarray(control_image)

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/path/to/controlnet/{}.png".format(name))

def preprocess_control_image(image_temp):
        image_temp = np.array(image_temp)
        image_temp = cv2.resize(image_temp, (512, 512))
        image_temp = Image.fromarray(image_temp)
        return image_temp

def test_multicontrolnet():
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path_canny = "/path/to/controlnet_egc2_canny/"
    controlnet_path_sobel = "/path/to/controlnet_egc2_sobel/"
    controlnet_path_gl = "/path/to/controlnet_egc2_gl/"
    controlnet_path_segmentation = /path/to/controlnet_egc2_segmentation/"

    controlnet_canny = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")
    controlnet_sobel = ControlNetModel.from_pretrained(controlnet_path_sobel, torch_dtype=torch.float16).to("cuda")
    controlnet_gl = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16).to("cuda")
    controlnet_segmentation = ControlNetModel.from_pretrained(controlnet_path_segmentation, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_canny, controlnet_sobel, controlnet_gl, controlnet_segmentation], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/path/to/chexpert_reports.csv')
    for name, text in tqdm(zip(df['name'], df['report'])):
        name = name.split('/')[1]

        control_image_canny = load_image("/path/to/canny/{}.png".format(name))
        control_image_sobel = load_image("/path/to/sobel/{}.png".format(name))
        control_image_gl = load_image("/path/to/gl/{}.png".format(name))
        control_image_segmentation = load_image("/path/to/segmentation/{}.png".format(name))

        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel), 
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=0.01, #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/path/to/multicontrolnet/{}.png".format(name))
        

if __name__ == '__main__':
    flag = 0

    # test_sd()
    # test_roentgen()
    # test_t2i_adapter()
    # test_controlnet()
    # test_multicontrolnet()
    # test_gazecontrolnet()
