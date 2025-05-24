import os
import shutil

import cv2 as cv
import numpy as np

if __name__ == '__main__':
    rootDir = "/media/data2/myName/ImageData/imagenet-c/snow" # root directory of the particular perturbation.
    trashJPG = np.zeros((32, 32, 3), dtype=np.uint8)
    for root, dirs, files in os.walk(rootDir):
        if root[-2] == '/':
            if root[-1] == '1' or root[-1] == '2' or root[-1] == '3' or root[-1] == '4' or root[-1] == '5':
                trainDir = os.path.join(root, 'train')
                testDir = os.path.join(root, 'val')
                os.mkdir(trainDir)
                os.mkdir(testDir)
                for dir in dirs:
                    src = os.path.join(root, dir)
                    dst = os.path.join(testDir, dir)
                    print('move {} to {}'.format(src, dst))
                    shutil.move(src, dst)
                    os.mkdir(os.path.join(trainDir, dir))
                    print('create {}'.format(os.path.join(trainDir, dir)))
                    cv.imwrite(os.path.join(os.path.join(trainDir, dir), 'trash.jpg'), trashJPG)

