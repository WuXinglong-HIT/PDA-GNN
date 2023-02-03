import numpy as np
import scipy.sparse as sp
from os.path import join

import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, coo_matrix

from utils import enPrint
from parse import argParser


class GraphDataset(Dataset):
    """
        Data Loader
        IN THE FORM OF **GRAPH**
    """

    def __init__(self, datasetName="Gowalla"):
        super(GraphDataset, self).__init__()
        self.datasetName = datasetName
        self.filePath = None
        if self.datasetName == 'Gowalla':
            self.filePath = '../data/gowalla'
        elif self.datasetName == 'Amazon-Book':
            self.filePath = '../data/amazon-book'
        elif self.datasetName == 'LastFM':
            self.filePath = '../data/lastfm'
        elif self.datasetName == 'Yelp2018':
            self.filePath = '../data/yelp2018'

        self.numUsers = 0
        self.numItems = 0
        self.trainInteractions = 0
        self.testInteractions = 0
        self.normGraph = None
        self.testSampleDict = {}

        with open(join(self.filePath, "train.txt"), "r") as trainFile:
            enPrint("Training Dataset Loading...")
            # trainRowIndices: userID list
            trainRowIndices = []
            # trainColIndices: itemID list
            trainColIndices = []
            for line in trainFile.readlines():
                if len(line) >= 0:
                    userItems = line.strip().split(" ")
                    userID = int(userItems[0])
                    itemIDs = [int(itemIDStr) for itemIDStr in userItems[1:]]
                    trainRowIndices.extend([userID] * len(itemIDs))
                    trainColIndices.extend(itemIDs)
                    # Number of Users, denoted as M
                    self.numUsers = max(userID, self.numUsers)
                    # Number of Items, denoted as N
                    self.numItems = max(max(itemIDs), self.numItems)
                    self.trainInteractions += len(itemIDs)

        with open(join(self.filePath, "test.txt"), "r") as testFile:
            enPrint("Testing Dataset Loading...")
            testRowIndices = []
            testColIndices = []
            for line in testFile.readlines():
                if len(line) >= 0:
                    userItems = line.strip().split(" ")
                    userID = int(userItems[0])
                    itemIDs = [int(itemIDStr) for itemIDStr in userItems[1:]]
                    testRowIndices.extend([userID] * len(itemIDs))
                    testColIndices.extend(itemIDs)
                    self.numUsers = max(userID, self.numUsers)
                    self.numItems = max(max(itemIDs), self.numItems)
                    self.testInteractions += len(itemIDs)
                    self.testSampleDict[userID] = itemIDs

            # userID、itemID start from 0
            self.numUsers += 1
            self.numItems += 1
            print(f"{len(trainRowIndices)} Interactions in Training Dataset")
            print(f"{len(testRowIndices)} Interactions in Testing Dataset")
            print(f"{self.datasetName} Sparsity:"
                  f"{(len(trainRowIndices) + len(testRowIndices)) / self.numUsers / self.numItems}")
            data = np.ones_like(trainRowIndices)
            # self.adjMatrix: adjacency matrix on traning set
            self.adjMatrix = csr_matrix((data, (trainRowIndices, trainColIndices)),
                                        shape=(self.numUsers, self.numItems))
            self.userDArray = np.array(self.adjMatrix.sum(axis=0)).squeeze()
            self.userDArray[self.userDArray == 0.] = 1
            self.itemDArray = np.array(self.adjMatrix.sum(axis=1)).squeeze()
            self.itemDArray[self.itemDArray == 0.] = 1

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    @property
    def getUserPosItems(self):
        """
        Access positive item ID list of each user in traning set
        :return: list of lists of positive items
        """
        posItemIDsList = []
        for userID in range(self.numUsers):
            _, posItemIDs = self.adjMatrix[userID].nonzero()
            posItemIDsList.append(posItemIDs)
        return posItemIDsList

    @property
    def getNormAdj(self):
        """
        Build the Normalized Adjacency Matrix.
            A =
            :math:`|0,     R|`\n
            :math:`|R^T,   0|`
        :return: csr_matrix
        """
        # enPrint(f"Normalized Adjacency Matrix Loading...")
        # npzFileName = "s_pre_adj_mat.npz"
        npzFileName = "normAdjMatrix.npz"

        def transCsrMatrix2SparseTensor(csrMatrix: csr_matrix) -> torch.sparse.FloatTensor:
            """
            Convert CSR_Matrix to Torch Sparse Float Tensor
            :param csrMatrix: CSR_Matrix
            :return: Torch Sparse Float Tensor
            """
            cooMatrix: coo_matrix = csrMatrix.tocoo()
            matrixTensor = torch.sparse.FloatTensor(torch.LongTensor([cooMatrix.row.tolist(), cooMatrix.col.tolist()]),
                                                    torch.FloatTensor(cooMatrix.data.astype(np.float32)))
            return matrixTensor

        if self.normGraph is None:
            try:
                normG = sp.load_npz(join(self.filePath, npzFileName))
                enPrint("NPZ Matrix Loaded Successfully...")
            except FileNotFoundError:
                enPrint("Generating Normalized Adjacency Matrix from Scratch...")
                normG = sp.dok_matrix((self.numUsers + self.numItems, self.numItems + self.numUsers))
                normG = normG.tolil()
                R = self.adjMatrix.tolil()
                normG[:self.numUsers, self.numUsers:] = R
                normG[self.numUsers:, :self.numUsers] = R.T
                nodeDegrees = np.array(normG.sum(axis=1)).squeeze()
                nodeDegreeSqrts = np.power(nodeDegrees, -0.5)
                nodeDegreeSqrts[np.isinf(nodeDegreeSqrts)] = 0.
                diagMatrix = sp.diags(nodeDegreeSqrts)
                normG = diagMatrix.dot(normG).dot(diagMatrix)
                normG = normG.tocsr()
                sp.save_npz(join(self.filePath, npzFileName), normG)

            self.normGraph = transCsrMatrix2SparseTensor(normG)
            if argParser.CUDA_AVAILABLE:
                self.normGraph = self.normGraph.coalesce().to(argParser.DEVICE)

        return self.normGraph

    def getBPRSamples(self):
        """
        Sampling of triplets - User, Positive Sample, Negtive Sample
        :math:`[[sampleUserID_0, posItemID_0, negItemID_0]`,
        :math:`\cdots,`
        :math:`[sampleUserID_n, posItemID_n, negItemID_n]]`
        :return: samples
        """
        samples = []
        posItemIDsList = self.getUserPosItems
        sampleUserIDs = np.random.randint(0, self.numUsers, self.trainInteractions)
        for sampleUserID in sampleUserIDs:
            posItemIDs4User = posItemIDsList[sampleUserID]
            posItemID = posItemIDs4User[np.random.randint(0, len(posItemIDs4User))]
            # Negative Item Sampling for **sampleUserID**
            while True:
                negItemID = np.random.randint(0, self.numItems)
                if negItemID not in posItemIDs4User:
                    break
            samples.append([sampleUserID, posItemID, negItemID])
        return np.array(samples)


if __name__ == '__main__':
    data = GraphDataset("Yelp2018")
    norm = data.getNormAdj
