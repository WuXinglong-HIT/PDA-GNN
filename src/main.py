import numpy as np
import torch
import parse
import os
import time
import greeting

from torch.optim import Adam
from torch.utils.data import DataLoader
from warnings import simplefilter
from tensorboardX import SummaryWriter
from parse import argParser
from utils import enPrint, setSeed

from model import PDAGNN
from dataProcessor import GraphDataset
from utils import shuffle, miniBatch
from utils import isPredOccurrence
from utils import metricAtK

# Model Parameter Configuration
args = argParser()
BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.test_batch_size
LR = args.lr
EPOCHS = args.epochs
SEED = args.seed
COMMENT = args.comment
DATASET = args.dataset
TOPK = args.topK
WEIGHT_DECAY1 = args.weight_decay_embed
WEIGHT_DECAY2 = args.weight_decay_behavior
FINAL_INTEGRATION = args.final_integration
IF_REG_BEHAV = args.ifRegBehav
IF_REG_EMBEDDING = args.ifRegEmbedding
IF_DROPOUT = args.ifDropOut
IF_LOAD = args.ifLoad
LOAD_MODEL_NAME = args.load_model_name
CUDA_AVAILABLE = args.CUDA_AVAILABLE
DEVICE = args.DEVICE

# Preparation
setSeed(SEED)
simplefilter(action="ignore", category=FutureWarning)

# Environment Configuration
ROOT_PATH = "/".join(os.path.abspath(__file__).split("/")[:-2])  # for Linux Environment
# ROOT_PATH = "\\".join(os.path.abspath(__file__).split("\\")[:-2])  # for Dos Environment
LOG_PATH = os.path.join(ROOT_PATH, "log")
DATA_PATH = os.path.join(ROOT_PATH, "data")
BOARD_PATH = os.path.join(LOG_PATH, "runs")
CHECKPOINT_PATH = os.path.join(LOG_PATH, "checkpoints")
MODEL_DUMP_FILENAME = os.path.join(CHECKPOINT_PATH, LOAD_MODEL_NAME)
DUMP_FILE_PREFIX = MODEL_DUMP_FILENAME.split("epoch")[0]
DUMP_FILE_SUFFIX = ".pth.tar"

# TensorBoard
writer = SummaryWriter(logdir=os.path.join(BOARD_PATH,
                                           argParser.comment + time.strftime("%Y%m%d-%Hh%Mm%Ss",
                                                                             time.localtime(time.time()))),
                       comment=COMMENT)


# Train
def train(userIDs, posItemIDs, negItemIDs, epoch):
    """
    1 Epoch Training Process
    :param userIDs: User ID list
    :param posItemIDs: Positive Item ID list
    :param negItemIDs: Negative Item ID list
    :param epoch: Epoch
    :return: None
    """
    # :param trainStep: global mini-batch step，
    global globalTrainStep

    model.train()
    userIDs, posItemIDs, negItemIDs = shuffle(userIDs, posItemIDs, negItemIDs)
    batchIterNum = len(userID) // BATCH_SIZE + 1
    averageLoss = 0.
    for (batchIter, (batchUser, batchPos, batchNeg)) in enumerate(
            miniBatch(userIDs, posItemIDs, negItemIDs, batchSize=BATCH_SIZE)):
        loss, regEmbedTerm, regBehavTerm = model.bprLoss(userIDs=batchUser, posItemIDs=batchPos, negItemIDs=batchNeg)
        print(f"\tLOSS:{loss:.4f}\tREG1:{regEmbedTerm:.4f}\tREG2:{regBehavTerm:.4f}", end='\t')
        # loss += regEmbedTerm * WEIGHT_DECAY1 + regBehavTerm * WEIGHT_DECAY2
        loss += regEmbedTerm * WEIGHT_DECAY1 + regBehavTerm * WEIGHT_DECAY2
        print(f"Total LOSS:{loss:.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        averageLoss += loss.item()
        globalTrainStep += 1
        writer.add_scalar("Train/BPR_LOSS", loss.item(), global_step=globalTrainStep, walltime=time.time())

    averageLoss /= batchIterNum
    print(f"[EPOCH {epoch:4d}] - [LOSS]: {averageLoss: .3f}")


# Test
def test(testSampleData: dict, globalStep: int):
    """
    Test and Metric Calculation
    :param testSampleData: dict-type variable，key - userID， value - according item list
    :param globalStep: global mini-batch step，used to record metric changes
    :return: None
    """
    testUserIDs = list(testSampleData.keys())
    maxTopK = max(TOPK)
    metrics = np.zeros((3, len(TOPK)))
    model.eval()

    with torch.no_grad():
        batchRecalls = []
        batchPrecisions = []
        batchNDCGs = []
        for batchUserIDs in miniBatch(testUserIDs, batchSize=TEST_BATCH_SIZE):
            groundTruePosItems = [testSampleData[userID] for userID in batchUserIDs]
            testRatings = model.getRatings(batchUserIDs)
            trainPosItems4Users = np.array(graph.getUserPosItems)[batchUserIDs]
            for rowIdx, userPosList in enumerate(trainPosItems4Users):
                testRatings[[rowIdx] * len(userPosList), userPosList] = -np.inf
            _, colIndices = torch.topk(testRatings, k=maxTopK)
            testRatings = testRatings.cpu().numpy()
            del testRatings
            colIndices = colIndices.cpu().numpy()
            groundTruePosItems = np.array(groundTruePosItems)
            res = isPredOccurrence(predList=colIndices, groundTrueList=groundTruePosItems)
            recalls = []
            precisions = []
            ndcgs = []
            for k in argParser.topK:
                recall, precision, ndcg = metricAtK(groundTruePosItems, res, k)
                recalls.append(recall)
                precisions.append(precision)
                ndcgs.append(ndcg)
            recalls = np.array(recalls)
            precisions = np.array(precisions)
            ndcgs = np.array(ndcgs)
            batchRecalls.append(recalls)
            batchPrecisions.append(precisions)
            batchNDCGs.append(ndcgs)
            print(f".", end="")
        print()
        metrics[0] = np.sum(batchRecalls, axis=0)
        metrics[1] = np.sum(batchPrecisions, axis=0)
        metrics[2] = np.sum(batchNDCGs, axis=0)
        metrics /= len(testUserIDs)
        writer.add_scalars(f'Test/Recall',
                           {'Recall@' + str(TOPK[i]): metrics[0][i] for i in range(len(TOPK))}, globalStep)
        writer.add_scalars(f'Test/Precision',
                           {'Precision@' + str(TOPK[i]): metrics[1][i] for i in range(len(TOPK))}, globalStep)
        writer.add_scalars(f'Test/NDCG',
                           {'NDCG@' + str(TOPK[i]): metrics[2][i] for i in range(len(TOPK))}, globalStep)
        enPrint(f"[TEST]")
        for k in range(len(TOPK)):
            print(f"Recall@{TOPK[k]:2d}: {metrics[0][k]: .6f}", end='\t')
            print(f"Precision@{TOPK[k]:2d}: {metrics[1][k]: .6f}", end='\t')
            print(f"NDCG@{TOPK[k]:2d}: {metrics[2][k]: .6f}", end='\t')
            print()


def trainTripleSampling(graph: GraphDataset):
    """
    Training Triplet Initialization
    :param graph: Graph Dataset
    :return: Randomly Sampled Triplets
    """
    trainTripleData = graph.getBPRSamples()
    userID = trainTripleData[:, 0]
    posItemID = trainTripleData[:, 1]
    negItemID = trainTripleData[:, 2]

    userID = torch.from_numpy(userID).long()
    posItemID = torch.from_numpy(posItemID).long()
    negItemID = torch.from_numpy(negItemID).long()

    if CUDA_AVAILABLE:
        userID = userID.to(DEVICE)
        posItemID = posItemID.to(DEVICE)
        negItemID = negItemID.to(DEVICE)
    return userID, posItemID, negItemID


if __name__ == '__main__':
    graph = GraphDataset(DATASET)
    testSampleData = graph.testSampleDict
    model = PDAGNN(graph=graph)
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)

    if IF_LOAD:
        try:
            model.load_state_dict(torch.load(MODEL_DUMP_FILENAME), strict=False)
            enPrint("Model Loaded from Dump File...")
            test(testSampleData, int(LOAD_MODEL_NAME.split(".")[0].split("-")[-1]))
            exit(0)
        except FileNotFoundError as exp:
            print(MODEL_DUMP_FILENAME + " NOT FOUND!")
        finally:
            writer.close()

    optimizer = Adam(model.parameters(), lr=LR)
    averageLoss = 0.
    globalTrainStep = 0

    try:
        for epoch in range(EPOCHS):
            if epoch % 1 == 0:
                test(testSampleData, epoch)
                # torch.save(model.state_dict(), DUMP_FILE_PREFIX + "epoch-" + str(epoch) + DUMP_FILE_SUFFIX)
            userID, posItemID, negItemID = trainTripleSampling(graph)
            train(userID, posItemID, negItemID, epoch)
    finally:
        writer.close()
