import csv
import os

import torch
import torchvision.models
import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.models.resnet import BasicBlock, Bottleneck

import CutOut
import advattack
import imagenet100
import myGrad
import myLoss
from models import resnet, densenet


def getVGG11(classNum, weight=None):
	retModel = torchvision.models.VGG(torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs["A"]),
									  num_classes=classNum)
	if weight is not None:
		retModel.load_state_dict(weight)
	return retModel, 'vgg11'


def getResnet18(classNum, weight=None):
	retModel = torchvision.models.ResNet(BasicBlock, [2, 2, 2, 2], classNum)
	if weight is not None:
		retModel.load_state_dict(weight)
	return retModel, 'resnet18'


def getResnet50(classNum, weight=None):
	retModel = torchvision.models.ResNet(Bottleneck, [3, 4, 6, 3], classNum)
	if weight is not None:
		retModel.load_state_dict(weight)
	return retModel, 'resnet50'


def getResnet18SmallKernel(classNum, weight=None):
	retModel = resnet.ResNet18(classNum)
	if weight is not None:
		retModel.load_state_dict(weight)
	return retModel, 'resnet18SmallKernel'


def getResnet34(classNum, weight=None):
	retModel = torchvision.models.ResNet(BasicBlock, [3, 4, 6, 3], classNum)
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


cifar10Arg = {'datasetName': 'cifar10',
			  'classNum': 10,
			  'trainDatasetFolder': "/media/data2/chenjx353/ImageData/",
			  'testDatasetFolder': "/media/data2/chenjx353/ImageData/",
			  'modelC': getResnet18SmallKernel,
			  'guideModelC': getResnet18SmallKernel,
			  'guideModelPath': "/media/data2/chenjx353/gradRegularAT/ckpt/cifar10/resnet18SmallKernel_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1"
			  }

cifar100Arg = {'datasetName': 'cifar100',
			   'classNum': 100,
			   'trainDatasetFolder': "/media/data2/chenjx353/ImageData/",
			   'testDatasetFolder': "/media/data2/chenjx353/ImageData/",
			   'modelC': getResnet18SmallKernel,
			   'guideModelC': getResnet18SmallKernel,
			   'guideModelPath': "/media/data2/chenjx353/gradRegularAT/ckpt/cifar100/resnet18SmallKernel_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1"
			   }

imagenetArg = {'datasetName': 'imagenet',
			   'classNum': 1000,
			   'trainDatasetFolder': "/media/dataX/lizheng/imagenet-pretrain/train/",
			   'testDatasetFolder': "/media/dataX/lizheng/imagenet-pretrain/val/",
			   'modelC': getResnet18,
			   'guideModelC': getResnet18,
			   'guideModelPath': "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet/resnet18_l2_eps0.ckpt"
			   }

imagenet100Arg = {'datasetName': 'imagenet100',
				  'classNum': 100,
				  'trainDatasetFolder': "/media/dataX/dongjunh/ImageNet-CLS/",
				  'testDatasetFolder': "/media/dataX/dongjunh/ImageNet-CLS/",
				  'modelC': getResnet18,
				  'guideModelC': getResnet18,
				  'guideModelPath': "/media/data2/chenjx353/gradRegularAT/ckpt/imagenet100/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1"
				  }

tinyImageNetArg = {'datasetName': 'tinyImageNet',
				   'classNum': 200,
				   'trainDatasetFolder': "/media/data2/chenjx353/ImageData/tiny-imagenet-200/train",
				   'testDatasetFolder': "/media/data2/chenjx353/ImageData/tiny-imagenet-200/val",
				   'modelC': getResnet18,
				   'guideModelC': getResnet18,
				   'guideModelPath': "/media/data2/chenjx353/gradRegularAT/ckpt/tinyImageNet/resnet18_NAT_ReduceLROnPlateau_wd0.0005_eps8-255_regCoeff-1"
				   }

if __name__ == '__main__':
	torch.set_num_threads(16)
	modelDir = './ckpt'
	arg = imagenet100Arg
	verbose = True
	cutout = False  # use CutOut augmentation
	cutoutHoles = 1  # Follow Duan et al.
	cutoutSize = 64  # 64 for 224*224 like imagenet, 16 for 32*32 like CIFAR10
	# -------------------train param----------------#
	batchSize = 128
	epochNum = 150
	lr = 0.1
	wd = 5e-4
	linfEps = 8 / 255
	linfEpsStr = '8-255'
	linfStepNum = 10
	linfStepSize = 2.5 * linfEps / linfStepNum
	methods = {
		# 'NAT': [-1], # Standard Training
		# 'PGDAT': [-1],
		'L2WD': [1000.0],  # Section VI-C5
		# 'TRADES': [6], # Section VI-C3
		# 'IGD': [1, 2, 3, 4], # OUR METHOD
		# 'TRADESIGD': [(4, 6.0), (3, 6.0), (2, 6.0), (1, 6.0)] # Section VI-C2
		# 'GINI': [0.1, 0.5, 1.0]  # Section VI-C5. Just address Reviewer's "brilliant" suggestion. Don't waste your time to train a trash
		# 'IGD(L2)': [1.0] # Section VI-C4
	}
	# ------------------------define device-------------------------#
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# ---------------define dataset------------------#
	datasetName = arg['datasetName']
	csvW = csv.writer(open('./{}_result.csv'.format(datasetName), 'a+', newline=''))
	trainDataset = None
	testDataset = None
	trainDatasetFolder = arg['trainDatasetFolder']
	testDatasetFolder = arg['testDatasetFolder']
	classNum = arg['classNum']
	if not os.path.exists(trainDatasetFolder):
		os.mkdir(trainDatasetFolder)
	if not os.path.exists(testDatasetFolder):
		os.mkdir(testDatasetFolder)
	if datasetName == 'cifar10':
		trainImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor()
		])
		testImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		])
		trainDataset = CIFAR10(trainDatasetFolder, True, trainImgTrans, None, True)
		testDataset = CIFAR10(testDatasetFolder, False, testImgTrans, None, True)
	elif datasetName == 'cifar100':
		trainImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor()
		])
		testImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		])
		trainDataset = CIFAR100(trainDatasetFolder, True, trainImgTrans, None, True)
		testDataset = CIFAR100(testDatasetFolder, False, testImgTrans, None, True)
	elif datasetName == 'imagenet':
		trainImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.Resize(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		testImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.Resize(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		trainDataset = torchvision.datasets.ImageFolder(trainDatasetFolder, trainImgTrans)
		testDataset = torchvision.datasets.ImageFolder(testDatasetFolder, testImgTrans)
	elif datasetName == 'imagenet100':
		trainDataset, testDataset = imagenet100.load_imagenet100(trainDatasetFolder, True)
		classNum = len(trainDataset.class_to_idx.keys())
	elif datasetName == 'tinyImageNet':
		trainImgTrans = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(64, padding=4),
														torchvision.transforms.RandomHorizontalFlip(),
														torchvision.transforms.ToTensor()])
		testImgTrans = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		])
		trainDataset = torchvision.datasets.ImageFolder(trainDatasetFolder, trainImgTrans)
		testDataset = torchvision.datasets.ImageFolder(testDatasetFolder, testImgTrans)
		classNum = len(trainDataset.class_to_idx.keys())
		assert classNum == len(testDataset.class_to_idx.keys())
	else:
		print('no such dataset:{}'.format(datasetName))
		exit(0)
	if cutout:
		print('Using CutOut')
		trainDataset.transform.transforms.append(CutOut.Cutout(cutoutHoles, cutoutSize))
	trainDataLoader = DataLoader(trainDataset, batchSize, shuffle=True, num_workers=16)
	testDataLoader = DataLoader(testDataset, batchSize, num_workers=24)

	if not os.path.exists(modelDir):
		os.mkdir(modelDir)
	modelDir = os.path.join(modelDir, datasetName)
	if not os.path.exists(modelDir):
		os.mkdir(modelDir)
	guideModel = None
	guideModelPath = arg['guideModelPath']
	for method, regularCoeffs in methods.items():
		if method in ['IGD', 'IGD(L2)']:
			guideModel, guideModelName = arg['guideModelC'](classNum)
			guideModel.load_state_dict(torch.load(guideModelPath, map_location=device))
			print('guide model: {}'.format(guideModelPath))
			guideModel = guideModel.to(device)
		for regCoefficient in regularCoeffs:
			print('{}_reg{}'.format(method, regCoefficient))
			natAccRow = []
			advAccRow = []
			model, modelName = arg['modelC'](classNum)
			model = model.to(device)
			opt = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=wd)
			sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', verbose=True, threshold=1e-2, patience=10)
			lossF = None
			modelFileName = '{}_{}_{}_wd{}'.format(modelName, method, sche.__class__.__name__, wd)
			if cutout:
				modelFileName += '_cutout({}_{})'.format(cutoutHoles, cutoutSize)
			modelFileName += '_eps{}'.format(linfEpsStr)
			modelFileName += '_reg{}'.format(regCoefficient)
			modelPath = os.path.join(modelDir, modelFileName)
			if method == 'NAT':
				lossF = torch.nn.CrossEntropyLoss()
			elif method in ['PGDAT', 'l2PGDAT']:
				lossF = torch.nn.CrossEntropyLoss()
			elif method == 'IGD':
				lossF = myLoss.IGDLoss()
			elif method == 'IGD(L2)':
				lossF = myLoss.IGDLossL2()
			elif method == 'TRADES':
				lossF = myLoss.trades_loss
			elif method == 'TRADESIGD':
				lossF = myLoss.TRADESIGDLoss()
			elif method == 'L2WD':
				lossF = myLoss.L2GradientDecay()
			elif method == 'GINI':
				lossF = myLoss.GiniLoss()
			else:
				print('no such method:{}'.format(method))
				exit(1)
			bestAdvAcc = 0
			bestNatAcc = 0
			for epoch in range(1, epochNum + 1):
				print('Epoch {}'.format(epoch))
				trainLoss = 0
				trainCELoss = 0
				trainCOSLoss = 0
				valLoss = 0
				sampleNum = 0
				# -----------------training-------------------#
				tempDataLoader = trainDataLoader
				if verbose:
					tempDataLoader = tqdm.tqdm(trainDataLoader, total=len(trainDataLoader))
				for batchX, batchY in tempDataLoader:
					batchX, batchY = batchX.to(device), batchY.to(device)
					loss = None
					opt.zero_grad()
					if method == 'NAT':
						model.train()
						pred = model(batchX)
						loss = lossF(pred, batchY)
					elif method == 'PGDAT':
						model.eval()
						batchX = advattack._pgd_whitebox(model,
														 torch.nn.CrossEntropyLoss,
														 batchX,
														 batchY,
														 linfEps,
														 linfStepNum,
														 linfStepSize)
						model.train()
						pred = model(batchX)
						loss = lossF(pred, batchY)
					elif method == 'TRADES':
						loss = lossF(model,
									 batchX,
									 batchY,
									 opt,
									 linfStepSize,
									 linfEps,
									 linfStepNum,
									 regCoefficient,
									 'l_inf')
					elif method == 'TRADESIGD':
						model.eval()
						guideGrad = myGrad.saliency(guideModel, batchX, batchY, False)[0]
						model.train()
						loss, ceLoss, cosLoss = lossF(model, batchX, batchY, guideGrad, opt, linfEps, linfStepSize,
													  linfStepNum,
													  regCoefficient[0], regCoefficient[1], 'l_inf')
						trainCELoss += ceLoss * len(batchX)
						trainCOSLoss += cosLoss * len(batchX)
					elif method in ['IGD', 'absIGD', 'IGD(L1)', 'IGD(L2)']:
						model.eval()
						advBatchX = advattack._pgd_whitebox(model,
															torch.nn.CrossEntropyLoss,
															batchX,
															batchY,
															linfEps,
															linfStepNum,
															linfStepSize)
						guideGrad = myGrad.saliency(guideModel, batchX, batchY, False)[0]
						model.train()
						loss, ceLoss, cosLoss = lossF(model, advBatchX, batchX, batchY, guideGrad, regCoefficient)
						trainCELoss += ceLoss * len(batchX)
						trainCOSLoss += cosLoss * len(batchX)
					elif method in ['GINI']:
						model.eval()
						advBatchX = advattack._pgd_whitebox(model,
															torch.nn.CrossEntropyLoss,
															batchX,
															batchY,
															linfEps,
															linfStepNum,
															linfStepSize)
						model.train()
						loss, ceLoss, cosLoss = lossF(model, advBatchX, batchX, batchY, regCoefficient)
						trainCELoss += ceLoss * len(batchX)
						trainCOSLoss += cosLoss * len(batchX)
					elif method == 'L2WD':
						loss, ceLoss, wdLoss = lossF(model, batchX, batchY, (linfEps, linfStepNum, linfStepSize),
													 regCoefficient)
						trainCELoss += ceLoss * len(batchX)
						trainCOSLoss += wdLoss * len(batchX)
					else:
						print('no such method:{}'.format(method))
					opt.zero_grad()
					loss.backward()
					opt.step()
					trainLoss += loss.cpu().item() * len(batchX)
					sampleNum += len(batchX)
				sche.step(trainLoss / sampleNum)
				print('train loss {} with ce {} cos {}'.format(trainLoss / sampleNum,
															   trainCELoss / sampleNum,
															   trainCOSLoss / sampleNum))
				# -----------------------validating-----------------#
				valLoss = 0
				advCorrectNum = 0
				natCorrectNum = 0
				totalNum = 0
				for batchX, batchY in testDataLoader:
					model.eval()
					batchX, batchY = batchX.to(device), batchY.to(device)
					totalNum += len(batchX)
					if method == 'NAT':
						natPred = model(batchX)
						loss = lossF(natPred, batchY)
						natPredictClass = torch.argmax(natPred, 1).cpu().numpy()
						natCorrectNum += (natPredictClass == batchY.cpu().numpy()).sum()
					elif method in ['PGDAT', 'TRADES', 'IGD', 'IGD(L2)', 'L2WD', 'TRADESIGD']:
						advBatchX = advattack._pgd_whitebox(model,
															torch.nn.CrossEntropyLoss,
															batchX,
															batchY,
															linfEps,
															linfStepNum,
															linfStepSize)
						advPred = model(advBatchX)
						natPred = model(batchX)
						advPredictClass = torch.argmax(advPred, 1).cpu().numpy()
						natPredictClass = torch.argmax(natPred, 1).cpu().numpy()
						advCorrectNum += (advPredictClass == batchY.cpu().numpy()).sum()
						natCorrectNum += (natPredictClass == batchY.cpu().numpy()).sum()
					else:
						print('no such method:{}'.format(method))
						exit(1)
				advAcc = advCorrectNum / totalNum
				natAcc = natCorrectNum / totalNum
				natAccRow.append(natAcc)
				advAccRow.append(advAcc)
				storeNow = False
				if method == 'NAT':
					if natAcc > bestNatAcc:
						bestNatAcc = natAcc
						storeNow = True
				else:
					if advAcc > bestAdvAcc:
						bestAdvAcc = advAcc
						storeNow = True
				if storeNow:
					torch.save(model.state_dict(), modelPath)
					print('store current weight to {}'.format(modelPath))
				print('val nat acc {}, val adv acc{}'.format(natAcc, advAcc))
			natAccRow.insert(0, '{}_natAcc'.format(modelPath))
			advAccRow.insert(0, '{}_advAcc'.format(modelPath))
			csvW.writerow(natAccRow)
			csvW.writerow(advAccRow)
