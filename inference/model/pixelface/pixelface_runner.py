
import os
import sys
import cv2
import torch
from glob import glob

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer

from inference.model.pixelface.facerestore import Pixel2Face

if not os.path.exists('experiments/pretrained_models'):
    os.makedirs('experiments/pretrained_models')
realesr_model_path = 'experiments/pretrained_models/RealESRGAN_x4plus.pth'
if not os.path.exists(realesr_model_path):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -O experiments/pretrained_models/RealESRGAN_x4plus.pth")

if not os.path.exists('experiments/RestoreFormerPlusPlus/'):
    os.makedirs('experiments/RestoreFormerPlusPlus/')
restoreformerplusplus_model_path = 'experiments/RestoreFormerPlusPlus/last.ckpt'
if not os.path.exists(restoreformerplusplus_model_path):
    os.system("wget https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt -O experiments/RestoreFormerPlusPlus/last.ckpt")

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=realesr_model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

os.makedirs('output', exist_ok=True)

def inference(img, aligned, scale):
    if scale > 4:
        scale = 4
    try:
        extension = os.path.splitext(os.path.basename(str(img)))[1]
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        elif len(img.shape) == 2:
            img_mode = None
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            print('too large size')
            return None, None
        
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        face_enhancer = Pixel2Face(model_path=restoreformerplusplus_model_path, upscale=2, arch='RestoreFormer++', bg_upsampler=upsampler)

        try:
            has_aligned = True if aligned == 'aligned' else False
            _, restored_aligned, restored_img = face_enhancer.enhance(img, has_aligned=has_aligned, only_center_face=False, paste_back=True)
            if has_aligned:
                output = restored_aligned[0]
            else:
                output = restored_img
        except RuntimeError as error:
            print('Error', error)

        try:
            if scale != 2:
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                h, w = img.shape[0:2]
                output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
        except Exception as error:
            print('Wrong scale input.', error)
        if img_mode == 'RGBA':
            extension = 'png'
        else:
            extension = 'jpg'
        save_path = f'output/out_{extension}'
        cv2.imwrite(save_path, output)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output, save_path
    except Exception as error:
        print('Exception :', error)
        return None, None
    

