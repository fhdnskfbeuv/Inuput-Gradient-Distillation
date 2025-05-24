import cv2 as cv
import torch
import numpy as np


def preprocessAttrMap(attrMap):  # [B, C, H, W] -> [B, H, W]
    B, C, H, W = attrMap.shape
    grayAttrMap = torch.mean(attrMap, 1)
    attrMapMean = torch.mean(grayAttrMap.view(-1, H * W), dim=1)
    attrMapStd = torch.std(grayAttrMap.view(-1, H * W), dim=1)
    minTensor = torch.min(grayAttrMap.view(-1, H * W), dim=1)[0]
    maxTensor = torch.max(grayAttrMap.view(-1, H * W), dim=1)[0]
    for i in range(len(grayAttrMap)):
        grayAttrMap[i, :] = torch.clip(grayAttrMap[i, :],
                                       attrMapMean[i] - 3 * attrMapStd[i],
                                       attrMapMean[i] + 3 * attrMapStd[i])
        grayAttrMap[i, :] = (grayAttrMap[i, :] - minTensor[i]) / (maxTensor[i] - minTensor[i])
    return grayAttrMap


def combineInputandAttr(input, attrMap, ratio=0.3):  # [B, C, H, W], [B, H, W]
    output = torch.permute(torch.clip(input, 0, 1), (0, 2, 3, 1)).detach().cpu().numpy() * 255
    attrMap = attrMap.detach().cpu().numpy()
    for i in range(len(output)):
        attrImg = cv.normalize(attrMap[i, :], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        output[i, :] = cv.cvtColor(output[i, :], cv.COLOR_RGB2BGR)
        output[i, :] = (1 - ratio) * output[i, :] + ratio * cv.applyColorMap(attrImg, cv.COLORMAP_JET)
    return np.uint8(output)


def batch2cvImg(input):
    output = torch.permute(torch.clip(input, 0, 1), (0, 2, 3, 1)).detach().cpu().numpy() * 255
    for i in range(len(output)):
        output[i, :] = cv.cvtColor(output[i, :], cv.COLOR_RGB2BGR)
    return np.uint8(output)


def saveImgs(input, paths: list, isSave=None):  # [B, H, W, C]
    for i in range(len(input)):
        if isSave is None or isSave[i]:
            cv.imwrite(paths[i], input[i, :])
