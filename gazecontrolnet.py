import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchxrayvision as xrv

from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



def generate_hva(model, input_tensor, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(281)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # model_outputs = cam.outputs

def score(img0, img1):
    mse = nn.MSELoss()
    score = mse(img0, img1)
    return score

def generate_hypotheses(path, label, name):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # img = cv2.resize(img, (512, 512))

    device = torch.device('cuda:0')

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    model.eval()

    voi_lut=True
    fix_monochrome=True

    count = 0
    best_score = 0.
    for i in os.listdir('/path/to/vindr_processed/{}/'.format(label)):
        ds = dcmread('/path/to/vindr_processed/{}/{}'.format(label, i))

        if voi_lut:
            img_x = apply_voi_lut(ds.pixel_array, ds)
        else:
            img_x = ds.pixel_array
        if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
            img_x = np.amax(img_x) - img_x
        img_x = img_x - np.min(img_x)
        img_x = img_x / np.max(img_x)
        img_x = (img_x * 255).astype(np.uint8)

        _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 

        kernel = np.ones((100, 100), np.uint8) 
        thresh = cv2.dilate(thresh, kernel, iterations=3) 

        height, width = thresh.shape[:2]
        left = (width - min(width, height)) // 2
        top = (height - min(width, height)) // 2
        right = left + min(width, height)
        bottom = top + min(width, height)
        center_img = thresh[top:bottom, left:right]
        black_background = np.zeros_like(thresh, dtype=np.uint8)
        black_background[top:bottom, left:right] = center_img
        black_background = cv2.resize(black_background, (img_x.shape[1], img_x.shape[0]))
        overlay_masked = cv2.bitwise_and(img_x, img_x, mask=black_background)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=(0.4), std=(0.2) )
        ])
        img0 = transform(img)
        img1 = transform(torch.from_numpy(overlay_masked.astype(float)))

        with torch.no_grad():
            outputs0 = model.features2(img0[None,...].to(device)) 
            outputs1 = model.features2(img1[None,...].to(device)) 

        match_score = score(outputs0, outputs1)
        if match_score > best_score:
            best_score = match_score
            best_hypothesis = black_background#overlay_masked

        count += 1
        if count == 5:
            break

    return best_hypothesis


if __name__ == '__main__':
    flag = 0

    df_0 = pd.read_csv('/path/to/label0.csv')
    df_1 = pd.read_csv('/path/to/label1.csv')

    vindr_labels = [
        'ate',  
        'cmg',  
        'cns',  
        'edm',  
        'eff',  
        'enl',  
        'fra',  
        'les',  
        'opa',  
        'pne',  
        'pth',  
        'ptx',
    ]

    count_x = 0
    for i in tqdm(os.listdir('/path/to/gaze_heatmaps/')):
        flag = 0
        name = i.split('.png')[0]
        name = name.split('focal_')[-1]
        if name == '1a9b3b82-402d9e19-445679da-a0accb94-ebc21d66':
            if name in df_0['dicom_id'].values.tolist():
                element = df_0.loc[df_0['dicom_id'] == name]
                element = element.drop(columns=['Unnamed: 0', 'dicom_id'])
                index = np.where(np.array(element.values.tolist()) == 1)
                columns = element.columns.values.tolist()
                
                if len(index[1]) > 0:
                    label = columns[np.array(index[1])[np.random.randint(low=0, high=len(np.array(index[1])), size=1)[0]]]
                else:
                    label = vindr_labels[np.random.randint(low=0, high=len(vindr_labels), size=1)[0]]
            elif name in df_1['dicom_id'].values.tolist():
                element = df_1.loc[df_1['dicom_id'] == name]
                element = element.drop(columns=['Unnamed: 0', 'dicom_id'])
                index = np.where(np.array(element.values.tolist()) == 1)
                columns = element.columns.values.tolist()
                
                if len(index[1]) > 0:
                    label = columns[np.array(index[1])[np.random.randint(low=0, high=len(np.array(index[1])), size=1)[0]]]
                else:
                    label = vindr_labels[np.random.randint(low=0, high=len(vindr_labels), size=1)[0]]

            best_hypothesis = generate_hypotheses('/path/to/gaze_heatmaps/'+i, label, name)
            cv2.imwrite('/path/to/'+name+'.png', best_hypothesis)