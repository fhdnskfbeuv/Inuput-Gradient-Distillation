import numpy as np
import torch


def binaryMetric(pred, target, thres):  # pred is [0,1], target ∈ {0, 1}, thres is [0, 1]
    assert pred.shape == target.shape
    assert 0 <= thres <= 1
    predictClass = pred.__ge__(thres).astype(int)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = target.shape[0]
    for i in range(predictClass.shape[0]):
        if predictClass[i] == target[i]:
            if predictClass[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predictClass[i] == 1:
                fp += 1
            else:
                fn += 1
    acc = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    return acc, recall, prec


def multiMetric(pred, target, axis):
    predictClass = np.argmax(pred, axis)
    classNum = pred.shape[axis]
    assert predictClass.shape == target.shape
    tpArr = np.zeros((classNum,))
    predArr = np.zeros((classNum,))
    targetArr = np.zeros((classNum,))
    acc = 0
    recall = np.zeros((classNum,))
    prec = np.zeros((classNum,))
    f1 = np.zeros((classNum,))
    for i in range(predictClass.shape[0]):
        if predictClass[i] == target[i]:
            tpArr[predictClass[i]] += 1
            acc += 1
        predArr[predictClass[i]] += 1
        targetArr[target[i]] += 1
    for i in range(classNum):
        recall[i] = tpArr[i] / targetArr[i] if targetArr[i] > 0 else 0
        prec[i] = tpArr[i] / predArr[i] if predArr[i] > 0 else 0
        f1[i] = 0
        if recall[i] + prec[i] > 0:
            f1[i] = 2 * recall[i] * prec[i] / (recall[i] + prec[i])
    return acc / target.shape[0], recall, prec, f1


def gini(arr):  # arr:[L,]
    assert len(arr.shape) == 1
    n = len(arr)
    sortedArr = torch.sort(torch.abs(arr))[0]
    coeff = torch.flip(torch.arange(start=1, end=n + 1), [0])
    sumOfArr = torch.sum(sortedArr)
    sumOfMultArr = torch.sum(coeff * sortedArr)
    if sumOfArr == 0:
        return 0
    return (n + 1 - 2 * (sumOfMultArr / sumOfArr)) / n


def giniLoss(mat):  #  [B, L]
    n = mat.shape[-1]
    sortedArr = torch.sort(torch.abs(mat), dim=-1)[0]
    coeff = torch.flip(torch.arange(start=1, end=n + 1, device=mat.device, dtype=mat.dtype), [-1]).unsqueeze(0).repeat(mat.shape[0], 1)
    sumOfArr = torch.sum(sortedArr, dim=-1, keepdim=True)  # [B, 1]
    sumOfMultArr = torch.sum(coeff * sortedArr, dim=-1, keepdim=True)  # [B, 1]
    sumOfMultArr[sumOfArr == 0] = 0
    sumOfArr[sumOfArr == 0] = 1
    return torch.mean((n + 1 - 2 * (sumOfMultArr / sumOfArr)) / n)


def tensorGini(tensor):  # mat:[..., C, H, W]
    assert len(tensor.shape) >= 4
    matSize = tensor.shape[-1] * tensor.shape[-2] * tensor.shape[-3]
    sampleSize = 1
    outShape = tensor.shape[:len(tensor.shape) - 3]
    for i in range(len(tensor.shape) - 3):
        sampleSize *= tensor.shape[i]
    flattenBatch = tensor.reshape(sampleSize, matSize)
    outTensor = np.zeros((sampleSize,))
    for i in range(len(flattenBatch)):
        outTensor[i] = gini(flattenBatch[i])
    outTensor = np.reshape(outTensor, outShape)
    return outTensor


if __name__ == '__main__':
    np.random.seed()
    testBatch = torch.rand((128, 3, 32, 32))
    print(tensorGini(testBatch))
