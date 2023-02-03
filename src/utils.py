"""
Utils.py
"""
import torch
import numpy as np
import random


def enPrint(text: str):
    """
    Enhanced Print Function
    :param text: Text to Be Printed
    :return: None
    """
    print(f"\033[1;37;41m{text}\033[0m")


def setSeed(seed):
    """
    Initialize Global Random Seed
    :param seed: Seed Value Set by User
    :return: None
    """
    # PyTorch Random Module
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Numpy Random Module
    np.random.seed(seed)
    # Python Random Module
    random.seed(seed)


def miniBatch(*tensors, batchSize: int = 1024):
    """
    Fetch Mini-Batch Data
    :param tensors: input Tensor
    :param batchSize: Batch Size
    :return: Batch Data
    """
    if len(tensors) == 1:
        for startIdx in range(0, len(tensors[0]), batchSize):
            yield tensors[0][startIdx: startIdx + batchSize]
    else:
        for startIdx in range(0, len(tensors[0]), batchSize):
            yield tuple(tensor[startIdx: startIdx + batchSize] for tensor in tensors)


def shuffle(*arrays):
    """
    Used for Shuffle Train Dataset
    :param arrays: input Tensor
    :return: shuffled Tensor
    """
    TensorSize = len(arrays[0])
    tensorIdx = np.arange(len(arrays[0]))
    np.random.shuffle(tensorIdx)

    if len(arrays) == 1:
        return arrays[0][tensorIdx]
    else:
        return tuple(array[tensorIdx] for array in arrays)


def isPredOccurrence(predList: np.ndarray, groundTrueList: np.ndarray):
    """
    Check if Predicted Results Occured in the Ground-true Result List
    :param predList: Predicted Result
    :param groundTrueList: Ground Result
    :return: np.ndarray
    """
    batchResults = []
    assert len(predList) == len(groundTrueList)
    for i in range(len(predList)):
        predResult = predList[i]
        groundTrueResult = groundTrueList[i]
        res = [pred in groundTrueResult for pred in predResult]
        res = np.array(res, dtype=np.float32)
        batchResults.append(res)
    return np.array(batchResults, dtype=np.float32)


def metricAtK(groundTrue, occurrenceResult, k):
    """
    Metric Calculation
    :param groundTrue: Ground True Data
    :param occurrenceResult: if Predicted Results Occurred in Ground-true Results
    :param k: Value of K in TOP@K
    :return: Recall@K, Precision@K, NDCG@K
    """
    rec = recallAtK(groundTrue, occurrenceResult, k)
    pre = precisionAtK(groundTrue, occurrenceResult, k)
    ndcg = NdcgAtK(groundTrue, occurrenceResult, k)
    return rec, pre, ndcg


def recallAtK(groundTrue, occurrenceResult, k):
    """
    Recall Calculation
    :param groundTrue: Ground True Data
    :param occurrenceResult: if Predicted Results Occurred in Ground-true Results
    :param k: Value of K in TOP@K
    :return: Recall
    """
    right_pred = occurrenceResult[:, :k].sum(1)
    recall_n = np.array([len(groundTrue[i]) for i in range(len(groundTrue))])
    recall = np.sum(right_pred / recall_n)
    return recall


def precisionAtK(groundTrue, occurrenceResult, k):
    """
    Precision Calculation
    :param groundTrue: Ground True Data
    :param occurrenceResult: if Predicted Results Occurred in Ground-true Results
    :param k: Value of K in TOP@K
    :return: Precision
    """
    right_pred = occurrenceResult[:, :k].sum(1)
    precis_n = k
    precision = np.sum(right_pred) / precis_n
    return precision


def NdcgAtK(groundTrue, occurrenceResult, k):
    """
    Normalized Discounted Cumulative Gain Calculation
    :math:`rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0`
    :param groundTrue: Ground True Data
    :param occurrenceResult: if Predicted Results Occurred in Ground-true Results
    :param k: Value of K in TOP@K
    :return: NDCG
    """
    assert len(occurrenceResult) == len(groundTrue)
    predData = occurrenceResult[:, :k]
    test_matrix = np.zeros((len(predData), k))
    for i, items in enumerate(groundTrue):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = predData * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)
