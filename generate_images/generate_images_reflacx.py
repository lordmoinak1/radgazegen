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

def test_gazecontrolnet():
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_mimic_15k/"
    controlnet_path_canny = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_canny/"
    controlnet_path_sobel = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_sobel/"
    controlnet_path_gl = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_segmentation/"
    controlnet_path_hva = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/gazecontrolnet/"

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
            controlnet_hva
            ], 
        safety_checker=None,  
        torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        control_image_canny = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_canny/{}".format(name))
        control_image_sobel = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_sobel/{}".format(name))
        control_image_gl = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_gl/{}".format(name))
        control_image_segmentation = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_segmentation/{}".format(name))
        try:
            control_image_hva = load_image("/data06/shared/moibhattacha/gazecontrolnet/generated_images/eval_reflacx/{}".format(name))
        except:
            control_image_hva = load_image("/data06/shared/moibhattacha/gazecontrolnet/generated_images/eval_reflacx/{}".format(name_x))
        name_x = name
        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel),
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),
            preprocess_control_image(control_image_hva),
            ]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, 
            controlnet_conditioning_scale=[0.1, 0.1, 0.1, 0.1, 0.001],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/1_gazecontrolnet_hva/{}".format(name))

def test_sd():
    model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        generator = torch.manual_seed(3407)
        image = pipe(text, num_inference_steps=50, generator=generator).images[0]

        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/sd/{}".format(name))

def test_roentgen():
    model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/roentgen/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        generator = torch.manual_seed(3407)
        image = pipe(text, num_inference_steps=30, generator=generator).images[0]

        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/roentgen/{}".format(name))

def test_t2i_adapter():
    adapter = T2IAdapter.from_pretrained("/data04/shared/moibhattacha/model_weights/diffusers_finetuning/t2i_xl_base1.0_egc1_15k", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        image = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_canny/{}".format(name))

        generator = torch.manual_seed(3407)

        image = pipe(text, image=preprocess_control_image(image), generator=generator, num_inference_steps=10).images[0]
        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/t2i/{}".format(name))

def preprocess_control_image(image_temp):
        image_temp = np.array(image_temp)
        image_temp = cv2.resize(image_temp, (512, 512))
        image_temp = Image.fromarray(image_temp)
        return image_temp

def test_controlnet():
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"

    controlnet_path_canny = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_canny/"
    controlnet_path_sobel = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_sobel/"
    controlnet_path_gl = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_segmentation/"

    controlnet = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        control_image_canny = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_canny/{}".format(name))
        control_image_sobel = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_sobel/{}".format(name))
        control_image_gl = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_gl/{}".format(name))
        control_image_segmentation = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_segmentation/{}".format(name))

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=preprocess_control_image(control_image_gl), controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/controlnet_gl/{}".format(name))

def test_multicontrolnet():
    base_model_path = "/data04/shared/moibhattacha/model_weights/diffusers_finetuning/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path_canny = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_canny/"
    controlnet_path_sobel = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_sobel/"
    controlnet_path_gl = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/data05/shared/moibhattacha/model_weights/eccv_gazecontrolnet/controlnet_egc2_segmentation/"

    controlnet_canny = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")
    controlnet_sobel = ControlNetModel.from_pretrained(controlnet_path_sobel, torch_dtype=torch.float16).to("cuda")
    controlnet_gl = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16).to("cuda")
    controlnet_segmentation = ControlNetModel.from_pretrained(controlnet_path_segmentation, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, 
        controlnet=[
            controlnet_canny, 
            controlnet_sobel, 
            controlnet_gl, 
            controlnet_segmentation
            ], 
        safety_checker=None,  
        torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/home/moibhattacha/gazecontrolnet/metadata_reflacx_multiple.csv')
    for name, text in tqdm(zip(df['file_name'], df['text'])):
        control_image_canny = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_canny/{}".format(name))
        control_image_sobel = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_sobel/{}".format(name))
        control_image_gl = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_gl/{}".format(name))
        control_image_segmentation = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_segmentation/{}".format(name))

        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel),
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, controlnet_conditioning_scale=[0.01, 0.01, 0.01, 0.1], #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/multicontrolnet_segmentation/{}".format(name))

if __name__ == '__main__':
    flag = 0

    # test_sd()
    # test_roentgen()
    # test_t2i_adapter()
    # test_controlnet()
    # test_multicontrolnet()
    test_gazecontrolnet()

    # image_name = 'ffebc425-86614d95-5bb96eaa-da4060e0-1136f220.png'
    # controls = ['gl', 'canny']
    # text = "right port catheter tip projects over the mid svc. no acute cardiopulmonary findings. no acute osseous abnormality."

    # test_gazecontrolnet(image_name, text, controls)

