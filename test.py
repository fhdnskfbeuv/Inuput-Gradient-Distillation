import copy
import csv
import math
import os
import random
import shutil
import time

import captum.attr
import numpy as np
import torch
import torchvision.models
import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.models.resnet import BasicBlock

import advattack
import autoattack as aa
import imagenet100
import metric
import myGrad
import visualize
from models import densenet, resnet


def loadWeight(model, weight):
    tempDict = dict()
    for k, v in weight.items():
        if k in model.state_dict().keys():
            tempDict[k] = v
    model.load_state_dict(tempDict)
    return model


def getResnet18(classNum, weight=None):
    retModel = torchvision.models.ResNet(BasicBlock, [2, 2, 2, 2], classNum)
    if weight is not None:
        retModel = loadWeight(retModel, weight)
    return retModel, 'resnet18'


def getResnet18SmallKernel(classNum, weight=None):
    retModel = resnet.ResNet18(classNum)
    if weight is not None:
        retModel = loadWeight(retModel, weight)
    return retModel, 'resnet18SmallKernel'


def getResnet50(classNum, weight=None):
    retModel = resnet.ResNet50(classNum)
    if weight is not None:
        # print(weight)
        retModel = loadWeight(retModel, weight)
    return retModel, 'resnet50'


def getResnet34(classNum, weight=None):
    retModel = torchvision.models.ResNet(BasicBlock, [3, 4, 6, 3], classNum)
    resnet.ResNet34()
    if weight is not None:
        retModel.load_state_dict(weight)
    return retModel, 'resnet34'


def getDensenet(classNum, weight=None):
    retModel = densenet.densenet_cifar(classNum)
    if weight is not None:
        retModel.load_state_dict(weight)
    return retModel, 'densenet'


def getDenseNet161(classNum, weight=None):
    retModel = torchvision.models.DenseNet(48, (6, 12, 36, 24), 96, num_classes=classNum)
    if weight is not None:
        retModel.load_state_dict(weight)
    return retModel, 'densenet161'


def loadPretrainedWeight(path, device):
    checkpoint = torch.load(path, map_location=device)
    # print(checkpoint)
    # Makes us able to load models saved with legacy versions
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]
    sd = {k[len('module.model.'):]: v for k, v in sd.items()}
    return sd


cifar10Arg = {'datasetName': 'cifar10',
              'classNum': 10,
              'modelC': getResnet18SmallKernel,
              'weightLoadF': torch.load,
              'imageFolder': r'/media/data2/chenjx353/ImageData/',
              'modelWeight': [
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10Others/resnet50Cifar10Nat",
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10Others/resnet50Cifar10Adv8-255",
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10Others/resnet18smallkernelNAT",
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10Others/resnet18smallkernelPGD"
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_16)",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                  "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4",
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff5",
                  # "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff6",
              ]
              }

cifar100Arg = {'datasetName': 'cifar100',
               'classNum': 100,
               'modelC': getResnet18SmallKernel,
               'weightLoadF': torch.load,
               'imageFolder': r'/media/data2/chenjx353/ImageData/',
               'modelWeight': [
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_16)",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4",
               ]
               }

imagenetArg = {'datasetName': 'imagenet',
               'classNum': 1000,
               'modelC': getResnet18,
               'imageFolder': r'/media/dataX/lizheng/imagenet-pretrain/',
               'weightLoadF': torch.load,
               'modelWeight': [
                   "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet/resnet18_l2_eps0.ckpt",
                   "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet/resnet18_linf_eps8.0.ckpt",
               ]
               }

imagenet100Arg = {'datasetName': 'imagenet100',
                  'classNum': 100,
                  'modelC': getResnet18,
                  'imageFolder': r"/media/dataX/dongjunh/ImageNet-CLS/",
                  'weightLoadF': torch.load,
                  'modelWeight': [
                     #"/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                      #pgdat
                      "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                      #"/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_64)",
                      "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                      "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                      "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                      "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4",
                      # tradesIGD
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADES_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff6",
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADES_ReduceLROnPlateau_wd0.0005_cutout(1_64)_eps8-255_reg6",
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADESIGD_ReduceLROnPlateau_wd0.0005_eps8-255_reg(1, 6.0)",
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADESIGD_ReduceLROnPlateau_wd0.0005_eps8-255_reg(2, 6.0)",
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADESIGD_ReduceLROnPlateau_wd0.0005_eps8-255_reg(3, 6.0)",
                      #"/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_TRADESIGD_ReduceLROnPlateau_wd0.0005_eps8-255_reg(4, 6.0)",
                      # l2gd
                      "/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_L2WD_ReduceLROnPlateau_wd0.0005_eps8-255_reg1000.0",
                      "/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_L2WD_ReduceLROnPlateau_wd0.0005_eps8-255_reg2500.0",
                      "/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_L2WD_ReduceLROnPlateau_wd0.0005_eps8-255_reg5000.0",
                      "/media/data2/chenjx353/IGD/ckpt/imagenet100/resnet18_L2WD_ReduceLROnPlateau_wd0.0005_eps8-255_reg7500.0",
                  ]
                  }

tinyImageNetArg = {'datasetName': 'tinyImageNet',
                   'classNum': 200,
                   'modelC': getResnet18,
                   'imageFolder': "/media/data2/chenjx353/ImageData/tiny-imagenet-200/val",
                   'weightLoadF': torch.load,
                   'modelWeight': [
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_16)",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4"
                   ]
                   }

imagenetC100Arg = {'datasetName': 'imagenetC100',
                   'classNum': 100,
                   'modelC': getResnet18,
                   'type': 'gaussian_noise',
                   'strength': '3',
                   'imageFolder': "/media/data2/chenjx353/ImageData/imagenet-c/",
                   'weightLoadF': torch.load,
                   'modelWeight': [
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/IGD/doublecheckckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_64)",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/IGD/doublecheckckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_PGDAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1_cutout(1_64)",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff1",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff2",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff3",
                       "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_gradRegularAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff4",

                   ]
                   }

if __name__ == '__main__':
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    doOutputSample = False  # we use this to get clean images. set False and ignore
    doNat = True  # evaluate nat. acc
    doAdv = True  # evaluate adv. acc with AA
    doGini = True  # calculate gini value
    doVisual = True  # generate attribution heat map. Only work when doGini = True
    doInductiveNoise = True  # evaluate model's robustness INA
    doRandomNoise = True  # evaluate model's robustness RN
    doOcc = True  # evaluate model's robustness IOA
    doOccSNR = False  # code for table 6
    doNoiseSNR = False  # code for fig 4
    batchSize = 128
    # ------------------------------------#
    arg = imagenet100Arg  # evaluate dataset argument
    visualDir = './visual'
    resultDir = './result'

    os.makedirs(visualDir, exist_ok=True)
    os.makedirs(resultDir, exist_ok=True)
    # ------------------------define device-------------------------#
    gpus = [4]  # gpus that is available. In this paper, we only use the first gpu
    GPUIndex = gpus[0]
    torch.cuda.set_device(GPUIndex)
    print('use cuda', gpus, 'main card', GPUIndex)
    device = torch.device("cuda:{}".format(GPUIndex) if torch.cuda.is_available() else "cpu")
    # -----------------------define test param--------------#
    advParams = [
        # (1 / 255, 20, 0.5 / 255, '1/255', '20', '05/255')
        # ,
        #          (2 / 255, 20, 0.5 / 255, '2/255', '20', '05/255')
        # ,
        #          (4 / 255, 20, 1 / 255, '4/255', '20', '1/255')
        # ,
        (8 / 255, 20, 2 / 255, '8/255', '20', '2/255')
        # ,
        #          (16 / 255, 20, 4 / 255, '16/255', '20', '4/255')
        # ,
        #          (32 / 255, 20, 4 / 255, '32/255', '20', '4/255')
        # ,
    ]
    rows = []
    headerRow = []
    gradFs = [
        #'ig',
        #'inputX',
        #'gradShap',
        #'smoothGrad',
        'saliency'
    ]

    # ---------------define dataset------------------#
    datasetName = arg['datasetName']
    visualDir = os.path.join(visualDir, datasetName)
    if not os.path.exists(visualDir):
        os.makedirs(visualDir)
    datasetResultDir = os.path.join(resultDir, datasetName)
    os.makedirs(datasetResultDir, exist_ok=True)
    csvW = csv.writer(open('./{}_test_result.csv'.format(datasetName), 'a+', newline=''))
    trainDataset = None
    testDataset = None
    datasetFolder = arg['imageFolder']
    classNum = arg['classNum']
    if not os.path.exists(datasetFolder):
        os.makedirs(datasetFolder)
    if datasetName == 'cifar10':
        testImgTrans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        testDataset = CIFAR10(datasetFolder, False, testImgTrans, None, True)
    elif datasetName == 'cifar100':
        testImgTrans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        testDataset = CIFAR100(datasetFolder, False, testImgTrans, None, True)
    elif datasetName == 'imagenet':
        testImgTrans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        testDataset = torchvision.datasets.ImageFolder(os.path.join(datasetFolder, 'val'), testImgTrans)
    elif datasetName == 'imagenet100':
        trainDataset, testDataset = imagenet100.load_imagenet100(datasetFolder, True)
        classNum = len(trainDataset.class_to_idx.keys())
    elif datasetName == 'tinyImageNet':
        testImgTrans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        testDataset = torchvision.datasets.ImageFolder(datasetFolder, testImgTrans)
        classNum = len(testDataset.class_to_idx.keys())
    elif datasetName == 'imagenetC100':
        datasetFolder = os.path.join(datasetFolder, arg['type'])
        datasetFolder = os.path.join(datasetFolder, arg['strength'])
        trainDataset, testDataset = imagenet100.load_imagenet100(datasetFolder, True)
        classNum = len(trainDataset.class_to_idx.keys())
    else:
        print('no such dataset:{}'.format(datasetName))
        exit(0)
    testDataLoader = DataLoader(testDataset, batchSize, num_workers=8)
    if doOutputSample:
        startIndex = 0
        currentDir = os.path.join(visualDir, 'clean')
        if not os.path.exists(currentDir):
            os.makedirs(currentDir)
        for batchX, batchY in tqdm.tqdm(testDataLoader):
            paths = []
            for i in range(len(batchX)):
                paths.append(os.path.join(currentDir, '{}.jpg'.format(startIndex + i)))
            visualize.saveImgs(visualize.batch2cvImg(batchX), paths)
            startIndex += len(batchX)
    modelC = arg['modelC']
    weightLoadF = arg['weightLoadF']
    allCorrectMask = np.ones((len(testDataset),), dtype=np.int64)
    if doOcc or doRandomNoise or doInductiveNoise or datasetName == 'imagenetC100' or doOccSNR or doNoiseSNR or (
            doVisual and doGini):
        tempTestLoader = testDataLoader
        if datasetName == 'imagenetC100':
            print('testing imagenetC100, using clean imagenet100')
            _, tempTestDataset = imagenet100.load_imagenet100(imagenet100Arg['imageFolder'], True)
            tempTestLoader = DataLoader(tempTestDataset, batchSize, num_workers=8)
        print('Calculating the number of samples that correctly classified by all models')
        for weightPath in arg['modelWeight']:
            print('model:{}'.format(weightPath))
            natAccRow = []
            advAccRow = []
            model, modelName = modelC(classNum, weightLoadF(weightPath, device))
            model = model.to(device)
            model.eval()
            left = 0
            for batchX, batchY in tqdm.tqdm(tempTestLoader):
                batchX, batchY = batchX.to(device), batchY.to(device)
                with torch.no_grad():
                    pred = model(batchX)
                natPredictClass = torch.argmax(pred, 1).cpu().numpy()
                allCorrectMask[left:left + len(batchX)] *= (natPredictClass == batchY.cpu().numpy())
                left += len(batchX)
    allCorrectNum = allCorrectMask.sum()
    print('all correct classified sample num:{}'.format(allCorrectNum))
    writenHeader = []
    for weightPath in arg['modelWeight']:
        print('model:{}'.format(weightPath))
        natAccRow = []
        advAccRow = []
        model, modelName = modelC(classNum, weightLoadF(weightPath, device))
        model = model.to(device)
        model.eval()
        folderName = weightPath
        folderName = folderName.replace('/', '_')
        currentVisDir = os.path.join(visualDir, folderName)
        os.makedirs(currentVisDir, exist_ok=True)
        modelResultDir = os.path.join(datasetResultDir, folderName)
        os.makedirs(modelResultDir, exist_ok=True)
        # -----------------------validating-----------------#
        valLoss = 0
        natCorrectNum = 0
        totalNum = testDataset.__len__()
        correctArr = np.zeros((totalNum,), dtype=bool)
        row = []
        row.append(weightPath)
        headerRow.append('model')
        if doNat:
            if datasetName == 'imagenetC100':
                headerRow.append('nat acc/error rate({}_{})'.format(arg['type'], arg['strength']))
            else:
                headerRow.append('nat acc/error rate/confidence')
            # ----------------------nat acc---------------------#
            left = 0
            successAttackNum = 0
            totalConfidence = 0
            for batchX, batchY in tqdm.tqdm(testDataLoader):
                batchX, batchY = batchX.to(device), batchY.to(device)
                with torch.no_grad():
                    pred = model(batchX)
                maxRes = torch.max(torch.softmax(pred, 1), 1)
                natPredictClass = maxRes[1].cpu().numpy()
                natPredictConf = maxRes[0].cpu().detach().numpy()
                correctArr[left:left + len(batchX)] = (natPredictClass == batchY.cpu().numpy())[:]
                natCorrectNum += (correctArr[left:left + len(batchX)]).sum()
                totalConfidence += ((natPredictClass == batchY.cpu().numpy()) * natPredictConf).sum()
                successAttackNum += (allCorrectMask[left:left + len(batchX)] * (
                        1 - (natPredictClass == batchY.cpu().numpy()))).sum()
                left += len(batchX)
            natAcc = natCorrectNum / totalNum
            errRate = successAttackNum / allCorrectNum
            row.append('{}/{}/{}'.format(natAcc, errRate, totalConfidence / natCorrectNum))
            np.save(os.path.join(modelResultDir, 'nat.npy'), correctArr)
        if doAdv:
            # --------------------adv acc------------------------#
            for p in advParams:
                aat = aa.AutoAttack(model, eps=p[0], verbose=False)
                advCorrectNum = 0
                headerRow.append('adv acc{}'.format(p[3]))
                left = 0
                for batchX, batchY in tqdm.tqdm(testDataLoader):
                    batchX, batchY = batchX.to(device), batchY.to(device)
                    batchX = aat.run_standard_evaluation(batchX, batchY)
                    with torch.no_grad():
                        pred = model(batchX)
                    advPredictClass = torch.argmax(pred, 1).cpu().numpy()
                    correctArr[left:left + len(batchX)] = (advPredictClass == batchY.cpu().numpy())[:]
                    advCorrectNum += correctArr[left:left + len(batchX)].sum()
                    left += len(batchX)
                advAcc = advCorrectNum / totalNum
                row.append(advAcc)
                np.save(os.path.join(modelResultDir, 'adv_{}.npy'.format(p[3].replace('/', '_'))), correctArr)
        if doGini:
            scales = [1, 16]
            for scale in scales:
                for gradF in gradFs:
                    giniNP = np.zeros((totalNum,), dtype=float)
                    headerRow.append('{}_gini_{}*{}'.format(gradF, scale, scale))
                    storeDir = os.path.join(currentVisDir, gradF)
                    if doVisual and scale == 1:
                        if not os.path.exists(storeDir):
                            os.makedirs(storeDir)
                        else:
                            shutil.rmtree(storeDir)
                            os.makedirs(storeDir)
                    # -----------------------------gini----------------------#
                    totalGini = 0
                    gradMapSum = 0
                    startIndex = 0
                    for batchX, batchY in tqdm.tqdm(testDataLoader):
                        batchX, batchY = batchX.to(device), batchY.to(device)
                        gradMap = None
                        if gradF == 'grad':
                            gradMap = myGrad.saliency(model, batchX, batchY, False)[0]
                        elif gradF == 'ig':
                            baselines = torch.randn(batchX.shape).to(device)
                            gradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines, batchY,
                                                                                       n_steps=20).detach()
                        elif gradF == 'inputX':
                            gradMap = captum.attr.InputXGradient(model).attribute(batchX, batchY).detach()
                        elif gradF == 'gradShap':
                            baselines = torch.zeros(batchX.shape).to(device)
                            gradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                                target=batchY).detach()
                        elif gradF == 'smoothGrad':
                            gradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                     nt_type='smoothgrad_sq',
                                                                                                     stdevs=0.2,
                                                                                                     nt_samples=20,
                                                                                                     target=batchY).detach()
                        elif gradF == 'saliency':
                            gradMap = captum.attr.Saliency(model).attribute(batchX, batchY, False).detach()
                        else:
                            print('no such method:{}'.format(gradF))
                            exit(1)
                        gradMap = gradMap.data.abs()
                        if scale > 1:
                            H = gradMap.shape[-2]
                            W = gradMap.shape[-1]
                            hPadSize = math.ceil(H / scale) * scale - H
                            wPadSize = math.ceil(W / scale) * scale - W
                            gradMap = torch.nn.functional.avg_pool2d(
                                torch.nn.functional.pad(gradMap, (0, wPadSize, 0, hPadSize), 'reflect'),
                                scale,
                                stride=scale).detach()
                            # gradMap *= scale * scale
                        gradMapSum += gradMap.cpu().numpy().sum()
                        giniArr = metric.tensorGini(gradMap.cpu())
                        giniNP[startIndex:startIndex + len(batchX)] = giniArr[:]
                        if doVisual and scale == 1:
                            with torch.no_grad():
                                pred = model(batchX)
                            isSave = (torch.argmax(pred,
                                                   1).detach().cpu().numpy() == batchY.detach().cpu().numpy()) * allCorrectMask[
                                                                                                                 startIndex:startIndex + len(
                                                                                                                     batchX)]
                            grayGradMap = visualize.preprocessAttrMap(gradMap)
                            imgs = visualize.combineInputandAttr(batchX, grayGradMap, 0.5)
                            paths = []
                            for index in range(len(giniArr)):
                                fileName = '{}_{}_gini{}.jpg'.format(startIndex + index, gradF, giniArr[index])
                                paths.append(os.path.join(storeDir, fileName))
                            visualize.saveImgs(imgs, paths, isSave)
                        startIndex += len(batchX)
                        totalGini += np.mean(giniArr) * len(batchX)
                    row.append('{}/{}'.format((totalGini / totalNum).item(), gradMapSum))
                    np.save(os.path.join(modelResultDir, 'gini_scale{}_gradf{}.npy'.format(scale, gradF)), giniNP)
        if doOccSNR:
            n = 10
            r = 10
            for gradF in gradFs:
                headerRow.append('occSNR_{}(n={}, r={}) classScore/sumSquare/squareSum'.format(gradF, n, r))
                totalConf = 0
                sumSquare = 0
                squareSum = 0
                SNR = 0
                left = 0
                for batchX, batchY in tqdm.tqdm(testDataLoader):
                    batchX, batchY = batchX.to(device), batchY.to(device)
                    gradMap = None
                    with torch.no_grad():
                        pred = model(batchX)
                    sortRes = torch.sort(torch.softmax(pred, 1), 1, descending=True)
                    maxRes = (sortRes[0][:, 0], sortRes[1][:, 0])
                    secRes = (sortRes[0][:, 1], sortRes[1][:, 1])
                    natPredictClass = maxRes[1].cpu().numpy()
                    natPredictConf = maxRes[0].cpu().detach().numpy()
                    basePredictConf = secRes[0].cpu().detach().numpy()
                    totalConf += (allCorrectMask[left:left + len(batchX)] * natPredictConf).sum()
                    if gradF == 'grad':
                        gradMap = myGrad.saliency(model, batchX, batchY, False)[0]
                    elif gradF == 'ig':
                        baselines = torch.randn(batchX.shape).to(device)
                        gradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines, batchY,
                                                                                   n_steps=20).detach()
                    elif gradF == 'inputX':
                        gradMap = captum.attr.InputXGradient(model).attribute(batchX, batchY)
                    elif gradF == 'gradShap':
                        baselines = torch.zeros(batchX.shape).to(device)
                        gradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                            target=batchY).detach()
                    elif gradF == 'smoothGrad':
                        gradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                 nt_type='smoothgrad_sq',
                                                                                                 stdevs=0.2,
                                                                                                 nt_samples=20,
                                                                                                 target=batchY).detach()
                    elif gradF == 'saliency':
                        gradMap = captum.attr.Saliency(model).attribute(batchX, batchY, False).detach()
                    else:
                        print('no such method:{}'.format(gradF))
                        exit(1)
                    gradMap = gradMap.data
                    gradMap = torch.sum(gradMap, 1, keepdim=False)
                    W, H = batchX.shape[-1], batchX.shape[-2]
                    rowLen = gradMap.shape[-1]
                    regionalAttrMap = gradMap.view(-1, gradMap.shape[-1] * gradMap.shape[-2])
                    maxRegion, maxRegionIndex = torch.sort(regionalAttrMap, 1, descending=True)
                    gradMap = gradMap.view(-1, H, W)
                    for index in range(len(gradMap)):
                        mask = torch.zeros(gradMap[index].shape).cuda()
                        for regionIndex in maxRegionIndex[index, :n]:
                            selectedI = torch.div(regionIndex, rowLen, rounding_mode='floor')
                            selectedJ = regionIndex % rowLen
                            leftupX = min(H, max(selectedI - r, 0))
                            leftupY = min(W, max(selectedJ - r, 0))
                            rightdownX = min(H, max(leftupX + r, 0))
                            rightdownY = min(W, max(leftupY + r, 0))
                            mask[leftupX:rightdownX, leftupY:rightdownY] = 1
                        sumSquare += np.square((mask * gradMap[index, :, :]).detach().cpu().numpy().sum()) * \
                                     allCorrectMask[left + index]
                        squareSum += np.square((mask * gradMap[index, :, :]).detach().cpu().numpy()).sum() * \
                                     allCorrectMask[left + index]
                    left += len(batchX)
                totalConf /= allCorrectNum
                sumSquare /= allCorrectNum
                squareSum /= allCorrectNum
                SNR_B = totalConf ** 2 / (0 * sumSquare + squareSum)
                SNR_G = totalConf ** 2 / (0.25 * sumSquare + squareSum)
                SNR_W = totalConf ** 2 / (sumSquare + squareSum)
                row.append('{}/{}/{}'.format(totalConf, sumSquare, squareSum))
        if doNoiseSNR:
            thress = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
            isInductive = True
            randomSampleNum = 1
            for gradF in gradFs:
                for thres in thress:
                    if isInductive:
                        headerRow.append(
                            'InductivenoiseSNR_{}_thres{} (classScore, LML)/sumSquare(mean, min, max, std)/squareSum(mean, min, max, std)'.format(
                                gradF, thres))
                    else:
                        headerRow.append(
                            'noiseSNR_{}_thres{} (classScore, LML)/sumSquare(mean, min, max, std)/squareSum(mean, min, max, std)'.format(
                                gradF,
                                thres))
                    totalConf = 0
                    sumSquare = 0
                    squareSum = 0
                    secSumSquare = 0
                    secSquareSum = 0
                    SNR = 0
                    left = 0
                    mu = 0
                    LML = 0  # logit margin loss
                    sumSquareArr = np.zeros((allCorrectNum,))
                    squareSumArr = np.zeros((allCorrectNum,))
                    secSumSquareArr = np.zeros((allCorrectNum,))
                    secSquareSumArr = np.zeros((allCorrectNum,))
                    flag = 0
                    for batchX, batchY in tqdm.tqdm(testDataLoader):
                        batchX, batchY = batchX.to(device), batchY.to(device)
                        gradMap = None
                        secGradMap = None
                        with torch.no_grad():
                            pred = model(batchX)
                        sortRes = torch.sort(torch.softmax(pred, 1), 1, descending=True)
                        maxRes = (sortRes[0][:, 0], sortRes[1][:, 0])
                        secRes = (sortRes[0][:, 1], sortRes[1][:, 1])
                        logitSortRes = torch.sort(pred, 1, descending=True)
                        logitMaxRes = (logitSortRes[0][:, 0], logitSortRes[1][:, 0])
                        logitSecRes = (logitSortRes[0][:, 1], logitSortRes[1][:, 1])
                        natPredictClass = maxRes[1].cpu().numpy()
                        basePredictClass = secRes[1].cpu().numpy()
                        natPredictConf = maxRes[0].cpu().detach().numpy()
                        basePredictConf = secRes[0].cpu().detach().numpy()
                        totalConf += (allCorrectMask[left:left + len(batchX)] * natPredictConf).sum()
                        mu += (allCorrectMask[left:left + len(batchX)] * (natPredictConf - basePredictConf)).sum()
                        LML += (allCorrectMask[left:left + len(batchX)] * (
                                logitMaxRes[0].cpu().detach().numpy() - logitSecRes[
                            0].cpu().detach().numpy())).sum()
                        natPredictClass = torch.tensor(natPredictClass, dtype=torch.long).to(device)
                        basePredictClass = torch.tensor(basePredictClass, dtype=torch.long).to(device)
                        if gradF == 'grad':
                            gradMap = myGrad.saliency(model, batchX, natPredictClass, False)[0]
                            secGradMap = myGrad.saliency(model, batchX, basePredictClass, False)[0]
                        elif gradF == 'ig':
                            baselines = torch.randn(batchX.shape).to(device)
                            gradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines,
                                                                                       natPredictClass,
                                                                                       n_steps=20).detach()
                            secGradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines,
                                                                                          basePredictClass,
                                                                                          n_steps=20).detach()
                        elif gradF == 'inputX':
                            gradMap = captum.attr.InputXGradient(model).attribute(batchX, natPredictClass)
                            secGradMap = captum.attr.InputXGradient(model).attribute(batchX, basePredictClass)
                        elif gradF == 'gradShap':
                            baselines = torch.zeros(batchX.shape).to(device)
                            gradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                                target=natPredictClass).detach()
                            secGradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                                   target=basePredictClass).detach()
                        elif gradF == 'smoothGrad':
                            gradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                     nt_type='smoothgrad_sq',
                                                                                                     stdevs=0.2,
                                                                                                     nt_samples=20,
                                                                                                     target=batchY).detach()
                            secGradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                        nt_type='smoothgrad_sq',
                                                                                                        stdevs=0.2,
                                                                                                        nt_samples=20,
                                                                                                        target=batchY).detach()
                        elif gradF == 'saliency':
                            gradMap = captum.attr.Saliency(model).attribute(batchX, natPredictClass, False).detach()
                            secGradMap = captum.attr.Saliency(model).attribute(batchX, basePredictClass, False).detach()
                        else:
                            print('no such method:{}'.format(gradF))
                            exit(1)
                        gradMap = gradMap.data
                        secGradMap = secGradMap.data
                        W, H, C = batchX.shape[-1], batchX.shape[-2], batchX.shape[-3]
                        totalPixelNum = W * H * C
                        thresPixel = int(max(0, min(totalPixelNum * thres, totalPixelNum)))
                        gradMap = gradMap.view(-1, totalPixelNum)
                        secGradMap = secGradMap.view(-1, totalPixelNum)
                        indeices = np.arange(0, gradMap.shape[-1], dtype=np.int32)
                        for i in range(len(batchX)):
                            if allCorrectMask[left + i]:
                                if isInductive:
                                    sortRes = torch.sort(gradMap[i], descending=True)
                                    secSortRes = torch.sort(secGradMap[i], descending=True)
                                    maxKAttr, _ = sortRes[0][:thresPixel], sortRes[1][:thresPixel]
                                    secMaxKAttr, _ = secSortRes[0][:thresPixel], secSortRes[1][:thresPixel]
                                    sumSquareArr[flag] = np.square(maxKAttr.detach().cpu().numpy().sum())
                                    squareSumArr[flag] = np.square(maxKAttr.detach().cpu().numpy()).sum()
                                    secSumSquareArr[flag] = np.square(secMaxKAttr.detach().cpu().numpy().sum())
                                    secSquareSumArr[flag] = np.square(secMaxKAttr.detach().cpu().numpy()).sum()
                                else:
                                    for sampleI in range(randomSampleNum):
                                        maxThresIndex = np.random.choice(indeices, thresPixel, replace=False)
                                        # print(maxThresIndex.shape)
                                        maxKAttr = gradMap[i][maxThresIndex]
                                        secMaxKAttr = secGradMap[i][maxThresIndex]
                                        sumSquareArr[flag] += np.square(maxKAttr.detach().cpu().numpy().sum())
                                        squareSumArr[flag] += np.square(maxKAttr.detach().cpu().numpy()).sum()
                                        secSumSquareArr[flag] += np.square(secMaxKAttr.detach().cpu().numpy().sum())
                                        secSquareSumArr[flag] += np.square(secMaxKAttr.detach().cpu().numpy()).sum()
                                    sumSquareArr[flag] /= randomSampleNum
                                    squareSumArr[flag] /= randomSampleNum
                                    secSumSquareArr[flag] /= randomSampleNum
                                    secSquareSumArr[flag] /= randomSampleNum
                                flag += 1
                        left += len(batchX)
                    assert flag == allCorrectNum
                    totalConf /= allCorrectNum
                    mu /= allCorrectNum
                    LML /= allCorrectNum
                    sumSquare, sumSquareMin, sumSquareMax, sumSquareStd = sumSquareArr.mean(), sumSquareArr.min(), sumSquareArr.max(), sumSquareArr.std()
                    squareSum, squareSumMin, squareSumMax, squareSumStd = squareSumArr.mean(), squareSumArr.min(), squareSumArr.max(), squareSumArr.std()
                    secSquareSum, secSquareSumStddev = secSquareSumArr.mean(), secSquareSumArr.std()
                    secSumSquare, secSumSquareStddev = secSumSquareArr.mean(), secSumSquareArr.std()
                    sigma = np.sqrt(squareSum + secSquareSum)
                    SNR = totalConf ** 2 / squareSum
                    row.append('({}, {})/{},{},{},{}/{},{},{},{}'.format(totalConf, LML,
                                                                         sumSquare,
                                                                         sumSquareMin,
                                                                         sumSquareMax,
                                                                         sumSquareStd,
                                                                         squareSum,
                                                                         squareSumMin,
                                                                         squareSumMax,
                                                                         squareSumStd))
        if doInductiveNoise:
            thress = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
            for gradF in gradFs:
                for thres in thress:
                    addCorrectArr = np.zeros((totalNum,), dtype=np.float)
                    replaceCorrectArr = np.zeros((totalNum,), dtype=np.float)
                    headerRow.append('{}_addNoise{} acc/rate'.format(gradF, thres))
                    headerRow.append('{}_replaceNoise{} acc/rate'.format(gradF, thres))
                    # -------------------------noise-------------------#
                    addNoiseCorrectNum = 0
                    replaceNoiseCorrectNum = 0
                    natCorrectNum = 0
                    successAttackNumAdd = 0
                    successAttackNumReplace = 0
                    left = 0
                    for batchX, batchY in tqdm.tqdm(testDataLoader):
                        batchX, batchY = batchX.to(device), batchY.to(device)
                        gradMap = None
                        if gradF == 'grad':
                            gradMap = myGrad.saliency(model, batchX, batchY, False)[0]
                        elif gradF == 'ig':
                            baselines = torch.randn(batchX.shape).to(device)
                            gradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines, batchY,
                                                                                       n_steps=20).detach()
                        elif gradF == 'inputX':
                            gradMap = captum.attr.InputXGradient(model).attribute(batchX, batchY)
                        elif gradF == 'gradShap':
                            baselines = torch.zeros(batchX.shape).to(device)
                            gradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                                target=batchY).detach()
                        elif gradF == 'smoothGrad':
                            gradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                     nt_type='smoothgrad_sq',
                                                                                                     stdevs=0.2,
                                                                                                     nt_samples=20,
                                                                                                     target=batchY).detach()
                        elif gradF == 'saliency':
                            gradMap = captum.attr.Saliency(model).attribute(batchX, batchY, False).detach()
                        else:
                            print('no such method:{}'.format(gradF))
                            exit(1)
                        gradMap = gradMap.data
                        # gradMap = torch.mean(gradMap, 1, keepdim=True)
                        # gradMap = torch.cat([gradMap, gradMap, gradMap], 1)
                        with torch.no_grad():
                            pred = model(batchX)
                        addNoiseBatchX = advattack.inductiveNoiseAttack(copy.deepcopy(batchX).cpu(),
                                                                        copy.deepcopy(gradMap).cpu(),
                                                                        int(batchX.shape[-1] *
                                                                            batchX.shape[-2] *
                                                                            batchX.shape[-3] * thres),
                                                                        'add')
                        addNoiseBatchX = addNoiseBatchX.to(device)
                        replaceNoiseBatchX = advattack.inductiveNoiseAttack(
                            copy.deepcopy(batchX).cpu(),
                            copy.deepcopy(gradMap).cpu(),
                            int(batchX.shape[-1] *
                                batchX.shape[-2] *
                                batchX.shape[-3] * thres),
                            'replace')
                        replaceNoiseBatchX = replaceNoiseBatchX.to(device)
                        with torch.no_grad():
                            addNoisePredClass = torch.argmax(model(addNoiseBatchX), 1).cpu().numpy()
                            replaceNoisePredClass = torch.argmax(model(replaceNoiseBatchX), 1).cpu().numpy()
                        addCorrectArr[left:left + len(batchX)] = (addNoisePredClass == batchY.cpu().numpy())[:]
                        replaceCorrectArr[left:left + len(batchX)] = (replaceNoisePredClass == batchY.cpu().numpy())[:]
                        addNoiseCorrectNum += addCorrectArr[left:left + len(batchX)].sum()
                        replaceNoiseCorrectNum += replaceCorrectArr[left:left + len(batchX)].sum()
                        successAttackNumAdd += (allCorrectMask[left:left + len(batchX)] * (
                                1 - (addNoisePredClass == batchY.cpu().numpy()))).sum()
                        successAttackNumReplace += (allCorrectMask[left:left + len(batchX)] * (
                                1 - (replaceNoisePredClass == batchY.cpu().numpy()))).sum()
                        left += len(batchX)
                    row.append('{}/{}'.format(addNoiseCorrectNum / totalNum, successAttackNumAdd / allCorrectNum))
                    row.append(
                        '{}/{}'.format(replaceNoiseCorrectNum / totalNum, successAttackNumReplace / allCorrectNum))
                    np.save(os.path.join(modelResultDir, 'INA1_gradf{}_thres{}.npy'.format(gradF, thres)),
                            addCorrectArr)
                    np.save(os.path.join(modelResultDir, 'INA2_gradf{}_thres{}.npy'.format(gradF, thres)),
                            replaceCorrectArr)
        if doRandomNoise:
            sampleNum = 1
            thress = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
            # thress = [0.1]
            # -------------------------noise-------------------#
            for thres in thress:
                headerRow.append('random additive noise({}) acc/rate'.format(thres))
                addNoiseCorrectNum = 0
                natCorrectNum = 0
                successAttackNumAdd = 0
                left = 0
                for batchX, batchY in tqdm.tqdm(testDataLoader):
                    batchX, batchY = batchX.to(device), batchY.to(device)
                    for s in range(sampleNum):
                        addNoiseBatchX = advattack.randomNoise(copy.deepcopy(batchX), int(batchX.shape[-1] *
                                                                                          batchX.shape[-2] *
                                                                                          batchX.shape[-3] * thres))
                        with torch.no_grad():
                            addNoisePredClass = torch.argmax(model(addNoiseBatchX), 1).cpu().numpy()
                        correctArr[left:left + len(batchX)] = (addNoisePredClass == batchY.cpu().numpy())[:]
                        addNoiseCorrectNum += correctArr[left:left + len(batchX)].sum()
                        successAttackNumAdd += (allCorrectMask[left:left + len(batchX)] * (
                                1 - (addNoisePredClass == batchY.cpu().numpy()))).sum()
                    left += len(batchX)
                row.append('{}/{}'.format(addNoiseCorrectNum / (totalNum * sampleNum),
                                          successAttackNumAdd / (allCorrectNum * sampleNum)))
                np.save(os.path.join(modelResultDir, 'RN_thres{}.npy'.format(thres)), correctArr)
        if doOcc:
            # ---------------------occ-----------------------#
            NRList = [
                (10, 10, 0),
                (10, 10, 0.5),
                (10, 10, 1)
            ]
            for (N, R, color) in NRList:
                for gradF in gradFs:
                    headerRow.append('{}_occ({})_N{}_R{} acc/rate'.format(gradF, color, N, R))
                    occCorrectNum = 0
                    natCorrectNum = 0
                    successAttackNum = 0
                    left = 0
                    for batchX, batchY in tqdm.tqdm(testDataLoader):
                        batchX, batchY = batchX.to(device), batchY.to(device)
                        gradMap = None
                        if gradF == 'grad':
                            gradMap = myGrad.saliency(model, batchX, batchY, False)[0]
                        elif gradF == 'ig':
                            baselines = torch.randn(batchX.shape).to(device)
                            gradMap = captum.attr.IntegratedGradients(model).attribute(batchX, baselines, batchY,
                                                                                       n_steps=20).detach()
                        elif gradF == 'inputX':
                            gradMap = captum.attr.InputXGradient(model).attribute(batchX, batchY).detach()
                        elif gradF == 'gradShap':
                            baselines = torch.zeros(batchX.shape).to(device)
                            gradMap = captum.attr.GradientShap(model).attribute(batchX, baselines, n_samples=10,
                                                                                target=batchY).detach()
                        elif gradF == 'smoothGrad':
                            gradMap = captum.attr.NoiseTunnel(captum.attr.Saliency(model)).attribute(batchX,
                                                                                                     nt_type='smoothgrad_sq',
                                                                                                     stdevs=0.2,
                                                                                                     nt_samples=20,
                                                                                                     target=batchY).detach()
                        elif gradF == 'saliency':
                            gradMap = captum.attr.Saliency(model).attribute(batchX, batchY, False).detach()
                        else:
                            print('no such method:{}'.format(gradF))
                            exit(1)
                        gradMap = gradMap.data
                        gradMap = torch.mean(gradMap, 1, keepdim=True)
                        gradMap = torch.cat([gradMap, gradMap, gradMap], 1)
                        # with torch.no_grad():
                        #     pred = model(batchX)
                        # natCorrectNum += (torch.argmax(pred, 1) == batchY).cpu().numpy().sum()
                        occBatchX = advattack.inductiveOcclusionAttack(model, batchX, batchY, gradMap, N, R,
                                                                       color)
                        with torch.no_grad():
                            occPredClass = torch.argmax(model(occBatchX), 1).cpu().numpy()
                        correctArr[left:left + len(batchX)] = (occPredClass == batchY.cpu().numpy())[:]
                        occCorrectNum += correctArr[left:left + len(batchX)].sum()
                        successAttackNum += (allCorrectMask[left:left + len(batchX)] * (
                                1 - (occPredClass == batchY.cpu().numpy()))).sum()
                        left += len(batchX)
                    row.append('{}/{}'.format(occCorrectNum / totalNum, successAttackNum / allCorrectNum))
                    print('{}/{}'.format(occCorrectNum / totalNum, successAttackNum / allCorrectNum))
                    np.save(os.path.join(modelResultDir, '{}_occ({})_N{}_R{}.npy'.format(gradF, color, N, R)),
                            correctArr)
        rows.append(row)
        writenHeader = headerRow
        headerRow = []
    rows.insert(0, writenHeader)
    csvW.writerows(rows)
