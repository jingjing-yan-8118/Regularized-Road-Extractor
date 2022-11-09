# -*- coding: UTF-8 -*-

from skimage import morphology, data, color
import matplotlib.pyplot as plt
import cv2 as cv
import os
import argparse



# step1: skeleton

def skeleton(filename, writename):
    image = cv.imread(filename, 2).astype('float32') / 255.0
    image[image > 0.4] = 1.0
    image[image <= 0.4] = 0
    skeleton = morphology.skeletonize(image)
    skeleton = (skeleton * 255).astype('uint8')
    cv.imwrite(writename, skeleton)

def clip(filename, writename, width, height):
    image = cv.imread(filename)
    image_clip = image[:height, :width, :]
    cv.imwrite(writename, image_clip)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask_root', type=str, default='/media/yanjingjing/TOSHIBA EXT/yanjingjing/dataset/Wuhan-0.5/mask/')
    parser.add_argument('--skeleton_root', type=str, default='/home/yanjingjing/data/temp/cen/')
    args = parser.parse_args()

    mask_root = args.mask_root
    skeleton_root = args.skeleton_root
    # os.makedirs(skeleton_root)
    image_list = list(os.listdir(mask_root))
    name_list = list(map(lambda x: x[:-4], image_list))
    print (name_list)
    for name in name_list:
        clipname = mask_root + name + '.png'
        skeletonname = skeleton_root + name + '.png'
        # clip(filename,clipname,8192,8192)
        skeleton(clipname, skeletonname)
        print(name + ' OK!')
