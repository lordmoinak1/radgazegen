import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def test_radgazegen(class_name):
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path_canny = "/path/to/controlnet_egc2_canny/"
    controlnet_path_sobel = "/path/to/controlnet_egc2_sobel/"
    controlnet_path_gl = "/path/to/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/path/to/controlnet_egc2_segmentation/"
    controlnet_path_hva = "/path/to/gazecontrolnet/"

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

    df = pd.read_csv('/path/to/gazecontrolnet_mimic_{}_reports.csv'.format(class_name))
    count = 0
    for canny, sobel, gl, segmentation, hva, text in tqdm(zip(df['canny'], df['sobel'], df['gl'], df['segmentation'], df['hva'], df['text'])):
        # image = load_image("/home/moibhattacha/gazecontrolnet/CheXpert/{}".format(name))
        print(hva)
        control_image_canny = load_image(canny)
        control_image_sobel = load_image(sobel)
        control_image_gl = load_image(gl)
        control_image_segmentation = load_image(segmentation)
        control_image_hva = load_image(hva)

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
        # image.save("/data05/shared/moibhattacha/gazecontrolnet/generated_images/reflacx/multicontrolnet_all/{}".format(name))
        count += 1
        image.save("/path/to/radgazegen/{}/sample_{}.png".format(class_name, count))

def test_controlnet(class_name):
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path = "/path/to/controlnet_egc1_sdv1-5_canny/"

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/path/to/mimic_{}_reports.csv'.format(class_name))
    count = 0
    for name, text in tqdm(zip(df['canny'], df['text'])):
        control_image = load_image(name)
        control_image = np.array(control_image)
        control_image = cv2.resize(control_image, (512, 512))
        control_image = Image.fromarray(control_image)

        image = pipe(
            text, num_inference_steps=50, image=control_image, controlnet_conditioning_scale=0.2, #image=resized_image, control_image=control_image
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        count += 1
        image.save("/path/to/controlnet/{}/sample_{}.png".format(class_name, count))

def preprocess_control_image(image_temp):
        image_temp = np.array(image_temp)
        image_temp = cv2.resize(image_temp, (512, 512))
        image_temp = Image.fromarray(image_temp)
        return image_temp

def test_multicontrolnet(class_name):
    base_model_path = "/path/to/stable_diffusion_v1_5_egc1_15k/"
    controlnet_path_canny = "/path/to/controlnet_egc2_canny/"
    controlnet_path_sobel = "/path/to/controlnet_egc2_sobel/"
    controlnet_path_gl = "/path/to/controlnet_egc2_gl/"
    controlnet_path_segmentation = "/path/to/controlnet_egc2_segmentation/"

    controlnet_canny = ControlNetModel.from_pretrained(controlnet_path_canny, torch_dtype=torch.float16).to("cuda")
    controlnet_sobel = ControlNetModel.from_pretrained(controlnet_path_sobel, torch_dtype=torch.float16).to("cuda")
    controlnet_gl = ControlNetModel.from_pretrained(controlnet_path_gl, torch_dtype=torch.float16).to("cuda")
    controlnet_segmentation = ControlNetModel.from_pretrained(controlnet_path_segmentation, torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=[controlnet_canny, controlnet_segmentation], safety_checker=None,  torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda:0")

    df = pd.read_csv('/path/to/mimic_{}_reports.csv'.format(class_name))
    count = 0
    for canny, segmentation, text in tqdm(zip(df['canny'], df['segmentation'], df['text'])):
        control_image_canny = load_image(canny)
        control_image_sobel = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_sobel/{}".format(name))
        control_image_gl = load_image("/data05/shared/moibhattacha/multicontrolnet/eval_reflacx_gl/{}".format(name))
        control_image_segmentation = load_image(segmentation)

        control_image = [
            preprocess_control_image(control_image_canny), 
            preprocess_control_image(control_image_sobel), 
            preprocess_control_image(control_image_gl), 
            preprocess_control_image(control_image_segmentation),]

        generator = torch.manual_seed(3407)
        image = pipe(
            text, num_inference_steps=50, generator=generator, image=control_image, 
            #controlnet_conditioning_scale=0.5, #image=resized_image, control_image=control_image #### use based on preference 
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        ).images[0]
        count += 1
        image.save("/path/to/multicontrolnet/{}/sample_{}.png".format(class_name, count))

if __name__ == '__main__':
    flag = 0

    # test_controlnet('tortuousaorta')
    # test_controlnet('calcificationoftheaorta')
    # test_controlnet('pleuralother')
    # test_controlnet('consolidation')
    # test_controlnet('enlargedcardiomediastinum')
    # test_controlnet('fracture')

    # test_multicontrolnet('tortuousaorta')
    # test_multicontrolnet('calcificationoftheaorta')
    # test_multicontrolnet('pleuralother')
    # test_multicontrolnet('consolidation')
    # test_multicontrolnet('enlargedcardiomediastinum')
    # test_multicontrolnet('fracture')

    # test_multicontrolnet('pneumomediastinum')
    # test_multicontrolnet('subcutaneousemphysema')
    # test_multicontrolnet('pneumoperitoneum')

    # test_radgazegen('tortuousaorta')
    # test_radgazegen('calcificationoftheaorta')
    # test_radgazegen('pleuralother')
    # test_radgazegen('consolidation')
    # test_radgazegen('enlargedcardiomediastinum')
    # test_radgazegen('fracture')

    # test_radgazegen('pneumomediastinum')
    # test_radgazegen('subcutaneousemphysema')
    # test_radgazegen('pneumoperitoneum')



