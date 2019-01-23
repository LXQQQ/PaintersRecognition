#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from os import mkdir, listdir, makedirs
from os.path import join, abspath, basename, splitext, dirname

from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img

DATA='./pic_dataset'
IMGS_DIM_2D=(224,224)

def _save_scaled_cropped_img(src, dest):
    image = load_img(src)
    image = fit(image, IMGS_DIM_2D, method=LANCZOS)
    image.save(dest,False)
    return image

def main():
    for dir in listdir(DATA):
        for file in listdir(join(DATA,dir)):
            _save_scaled_cropped_img(join(DATA,dir,file),join(DATA,dir,file))


if __name__ == '__main__':
    main()