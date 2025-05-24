import copy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


def _pgd_whitebox(model,
                  lossC,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X_pgd.device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = lossC()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def _pgd_whitebox_l2(model,
                  lossC,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
    batch_size = len(X)
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = 0.001 * torch.randn(X.shape).cuda().detach()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = lossC()(model(X_pgd), y)
        loss.backward()
        # renorming gradient
        grad_norms = X_pgd.grad.view(batch_size, -1).norm(p=2, dim=1)
        X_pgd.grad.div_(grad_norms.view(-1, 1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            X_pgd.grad[grad_norms == 0] = torch.randn_like(X_pgd.grad[grad_norms == 0])
        eta = step_size * X_pgd.grad.data
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = X_pgd.data.clamp_(0, 1).sub_(X)
        eta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd


def msd_v0(model, X, y, epsilon_l_inf=0.03, epsilon_l_2=0.5,
           alpha_l_inf=0.003, alpha_l_2=0.05, num_iter=50, device="cuda:0"):
    delta = torch.zeros_like(X, requires_grad=True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device).half()
    max_max_loss = torch.zeros(y.shape[0]).to(y.device).half()

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():
            # For L_2
            delta_l_2 = delta.data + alpha_l_2 * delta.grad / norms(delta.grad)
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2 = torch.min(torch.max(delta_l_2, -X), 1 - X)  # clip X+delta to [0,1]

            # For L_inf
            delta_l_inf = (delta.data + alpha_l_inf * delta.grad.sign()).clamp(-epsilon_l_inf, epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1 - X)  # clip X+delta to [0,1]

            # # For L1
            # k = random.randint(5, 20)
            # alpha_l_1 = (alpha_l_1_default / k) * 20
            # delta_l_1 = delta.data + alpha_l_1 * l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k=k)
            # delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
            # delta_l_1 = torch.min(torch.max(delta_l_1, -X), 1 - X)  # clip X+delta to [0,1]

            # Compare
            delta_tup = (delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device).half()
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction='none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp).half()
            delta.data = max_delta.data
            max_max_delta[max_loss > max_max_loss] = max_delta[max_loss > max_max_loss]
            max_max_loss[max_loss > max_max_loss] = max_loss[max_loss > max_max_loss]
        delta.grad.zero_()

    return max_max_delta


def _fgsm_whitebox(model,
                   X,
                   y,
                   epsilon):
    X_pgd = Variable(X.data, requires_grad=True)

    # random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    # X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    loss.backward()
    eta = epsilon * X_pgd.grad.data.sign()
    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def _cw_whitebox(model,
                 X,
                 y,
                 epsilon,
                 num_steps,
                 step_size,
                 classNum):  # 8/255: 0.003    4/255: 0.003     2/255: 0.0015   1/255: 0.0015
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = cwloss(model(X_pgd), y, num_classes=classNum)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def cwloss(output, target, confidence=50, num_classes=10):
    # compute the probability of the label class versus the maximum other
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def inductiveNoiseAttack(batchX, attrMap, thres, noiseType):
    assert batchX.shape == attrMap.shape
    oriShape = batchX.shape
    totalPixelNum = batchX.shape[-1] * batchX.shape[-2] * batchX.shape[-3]
    thres = max(0, min(thres, totalPixelNum))
    batchX = batchX.view(-1, totalPixelNum)
    attrMap = attrMap.view(-1, totalPixelNum)
    for i in range(len(batchX)):
        sortRes = torch.sort(attrMap[i], descending=True)
        maxKAttr, maxThresIndex = sortRes[0][:thres], sortRes[1][:thres]
        mask = torch.zeros_like(batchX[i], dtype=torch.int64)
        mask[maxThresIndex] = 1
        if noiseType == 'add':
            batchX[i] += torch.randn(batchX[i].shape) * mask
        elif noiseType == 'replace':
            batchX[i] = (1 - mask) * batchX[i] + torch.randn(batchX[i].shape) * mask
        else:
            print('no such type')
            return None
    batchX = batchX.reshape(oriShape)
    batchX = torch.clamp(batchX, 0, 1)
    attrMap = attrMap.reshape(oriShape)
    return batchX


def randomNoise(batchX, thres):
    oriShape = batchX.shape
    totalPixelNum = batchX.shape[-1] * batchX.shape[-2] * batchX.shape[-3]
    thres = max(0, min(thres, totalPixelNum))
    batchX = batchX.view(-1, totalPixelNum)
    indeices = np.arange(0, batchX.shape[-1], dtype=np.int)
    for i in range(len(batchX)):
        maxThresIndex = np.random.choice(indeices, thres, replace=False)
        mask = torch.zeros_like(batchX[i], dtype=torch.int64).to(batchX.device)
        mask[maxThresIndex] = 1
        batchX[i] += mask * torch.randn(batchX[i].shape).to(batchX.device)
    batchX = batchX.reshape(oriShape)
    batchX = torch.clamp(batchX, 0, 1)
    return batchX


def inductiveOcclusionAttack(model, batchX, batchY, attrMap, N, R, c):
    #stride = 1
    assert batchX.shape == attrMap.shape
    oriShape = batchX.shape
    sampleNum = len(batchX)
    W, H = batchX.shape[-1], batchX.shape[-2]
    batchX.requires_grad = False
    attrMap.requires_grad = False
    with torch.no_grad():
        pred = model(batchX)
    predClass = torch.argmax(pred, 1)
    sample2perturb = (predClass.cpu() == batchY.cpu()).numpy()
    batchX = batchX.view(-1, H, W)  # [bacthSize * cNum, H, W]
    occBatchX = copy.deepcopy(batchX)
    channelNum = len(batchX) // sampleNum
    rowLen = attrMap.shape[-1]
    regionalAttrMap = attrMap.view(-1, attrMap.shape[-1] * attrMap.shape[-2])
    maxRegion, maxRegionIndex = torch.sort(regionalAttrMap, 1, descending=True)
    nr2mask = {}
    with torch.no_grad():
        for n in range(1, N + 1):
            for r in range(1, R + 1):
                nr2mask['{}_{}'.format(n, r)] = torch.zeros_like(batchX)
        for index in range(len(batchX)):
            for r in range(1, R + 1):
                for i, regionIndex in enumerate(maxRegionIndex[index, :N]):
                    selectedI = torch.div(regionIndex, rowLen, rounding_mode='floor')
                    selectedJ = regionIndex % rowLen
                    leftupX = min(H, max(selectedI - r, 0))
                    leftupY = min(W, max(selectedJ - r, 0))
                    rightdownX = min(H, max(selectedI + r, 0))
                    rightdownY = min(W, max(selectedJ + r, 0))
                    nr2mask['{}_{}'.format(i+1, r)][index, leftupX:rightdownX, leftupY:rightdownY] = 1
                    if i > 0:
                        nr2mask['{}_{}'.format(i + 1, r)] += nr2mask['{}_{}'.format(i, r)]
                        nr2mask['{}_{}'.format(i + 1, r)] = torch.clip(nr2mask['{}_{}'.format(i + 1, r)], 0, 1)
    for n in range(1, N + 1):
        if sample2perturb.sum() == 0:
            break
        for r in range(1, R + 1):
            if sample2perturb.sum() == 0:
                break
            # mask = torch.zeros(batchX.shape, dtype=torch.int64)
            perturbedNum = 0
            mask = nr2mask['{}_{}'.format(n, r)]
            noModifyMask = torch.zeros(batchX.shape).cuda()
            for index in range(len(batchX)):
                if not sample2perturb[index // channelNum]:
                    noModifyMask[index, :, :] = 1
                    continue
                perturbedNum += 1
            with torch.no_grad():
                occBatchX = noModifyMask * occBatchX + (1 - noModifyMask) * ((1 - mask) * batchX + mask * c)
                occBatchX = torch.reshape(occBatchX, oriShape)
                pred = model(occBatchX)
            predClass = torch.argmax(pred, 1)
            oldSample2perturb = sample2perturb
            sample2perturb = (predClass.cpu() == batchY.cpu()).numpy()
            occBatchX = occBatchX.view(-1, H, W)
    occBatchX = torch.reshape(occBatchX, oriShape)
    batchX = torch.reshape(batchX, oriShape)
    return occBatchX


if __name__ == '__main__':
    batchX = torch.zeros((1, 3, 8, 8)) + 0.5
    attrMap = torch.zeros((1, 3, 8, 8))
    attrMap[:, :, 2:5, 2:5] = 1
    attacked = inductiveNoiseAttack(batchX, attrMap, 26, 'replace')
    attacked = attacked.view(-1, 8, 8)
    for a in attacked:
        print(a)
