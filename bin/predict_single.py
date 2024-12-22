#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from LaMa_copy.saicinpainting.evaluation.utils import move_to_device
from LaMa_copy.saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image

from LaMa_copy.saicinpainting.training.trainers import load_checkpoint

def resize(image, max_size, interpolation=Image.BICUBIC):
    w, h = image.size
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
    return image

class LaMa_Inpainter():
    def __init__(self, path = "./assets/lama_checkpoint", checkpoint = "best.ckpt"):
        self.device = torch.device("cpu")
        train_config_path = os.path.join(path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            self.train_config = OmegaConf.create(yaml.safe_load(f))
        self.train_config.training_model.predict_only = True
        self.train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(path, checkpoint)

        self.model = load_checkpoint(self.train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze() #May move to function calls

        self.model.to(self.device)
        print("LaMa Loaded")

    def inpaint(self, image, mask, size = 512):
        
        og_size = image.size
        ## What if we keep aspect ratio??

        # What happens if we don't resize it?? (still ensure mask is same size as image)

        image = np.array(image.convert("RGB").resize((size, size)))
        mask = np.array(mask.convert("L").resize((size, size)))[None, ...]

        image = np.transpose(image, (2, 0, 1))
        
        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        batch = {"image" : image, "mask" : mask}

        batch = default_collate([batch])

        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)
            # print(batch)             
            cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        
        return Image.fromarray(cur_res).resize(og_size)

def main():
    lama = LaMa_Inpainter()

    image = Image.open("./test_images2/test2.png")
    mask = Image.open("./test_images2/test2_mask001.png")

    lama.inpaint(image, mask).save("inpainted_image.jpeg")

if __name__ == '__main__':
    main()
