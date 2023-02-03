import torch
import collections

from torch import nn
import torch.nn.functional as F
from dataProcessor import GraphDataset
from parse import argParser
from utils import enPrint


class PDAGNN(nn.Module):
    """
    Graph Neural Network Model - PDA-GNN
    """

    def __init__(self, graph: GraphDataset):
        super(PDAGNN, self).__init__()
        args = argParser()
        self.graph = graph

        # Parameter Loaded
        self.embeddingDim = args.embedding_dim
        self.hiddenDims = args.att_hidden_dims
        self.keepProb = args.keep_prob
        self.device = args.DEVICE
        self.finalIntegration = args.final_integration
        self.attNorm = args.att_norm
        # numLayers stand for GCN propagation depth, identifying with number of attributes
        self.numLayers = args.num_layers
        self.ifRegBehav = args.ifRegBehav
        self.ifRegEmbedding = args.ifRegEmbedding
        self.ifDropOut = args.ifDropOut

        self.layerUserEmbeddings = nn.ModuleList()
        self.layerItemEmbeddings = nn.ModuleList()
        for layer in range(self.numLayers + 1):
            userEmbedding = nn.Embedding(num_embeddings=self.graph.numUsers, embedding_dim=self.embeddingDim)
            itemEmbedding = nn.Embedding(num_embeddings=self.graph.numItems, embedding_dim=self.embeddingDim)
            nn.init.normal_(userEmbedding.weight, mean=0., std=0.1)
            nn.init.normal_(itemEmbedding.weight, mean=0., std=0.1)
            userEmbedding.to(device=self.device)
            itemEmbedding.to(device=self.device)
            self.layerUserEmbeddings.append(userEmbedding)
            self.layerItemEmbeddings.append(itemEmbedding)
        self.activeLayer = nn.Sigmoid()

        self.attMlp = nn.Sequential(
            collections.OrderedDict([('attMLP-layer0',
                                      torch.nn.Linear(in_features=self.embeddingDim, out_features=self.hiddenDims[0]))])
        )
        for layerIdx in range(len(self.hiddenDims) - 1):
            linear = torch.nn.Linear(in_features=self.hiddenDims[layerIdx], out_features=self.hiddenDims[layerIdx + 1])
            self.attMlp.add_module('attMLP-layer{0}'.format(layerIdx + 1), linear)
        self.softMaxLayer = nn.Softmax(dim=1)

        dimInput, dimOutput = self.embeddingDim, self.embeddingDim
        self.attW = nn.Parameter(torch.zeros(size=(dimInput, dimOutput)))
        self.attA = nn.Parameter(torch.zeros(size=(2 * dimOutput, 1)))
        nn.init.xavier_uniform_(self.attW.data, gain=1.414)
        nn.init.xavier_uniform_(self.attA, gain=1.414)
        self.attW.to(device=self.device)
        self.attA.to(device=self.device)
        self.leakyReLU = nn.LeakyReLU(negative_slope=2e-1)

        enPrint("MBAGCN Model Loaded...")
        enPrint("Embedding Initialization Loaded...")

    def attLayer(self, inputTensor: torch.Tensor):
        """
        :param inputTensor  The input final tensor[B, n_b, D]
        :return             attention weighted tensor matrix
        """
        # batchSize: B
        batchSize = inputTensor.size()[0]
        # numBehavs: n_b
        numBehavs = inputTensor.size()[1]
        # dimInput: D, dimOutput: D'
        dimInput = inputTensor.size()[2]
        dimOutput = self.attW.size()[1]

        # [B, n_b, D'] -> [B, n_b, D']
        h = torch.matmul(inputTensor, self.attW)
        # [B, n_b * n_b, 2 * D'] -> [B, n_b, n_b, 2 * D']
        aInput = torch.cat([h.repeat(1, 1, numBehavs).view(batchSize, numBehavs * numBehavs, -1),
                            h.repeat(1, numBehavs, 1)], dim=2).view(batchSize, numBehavs, -1, 2 * dimOutput)
        # [B, n_b, n_b, 1] -> [B, n_b, n_b]
        attMat = self.leakyReLU(torch.matmul(aInput, self.attA)).squeeze(3)
        attMat = F.softmax(attMat, dim=2)
        # if self.ifDropOut:
        #     attMat = F.dropout(attMat, self.keepProb, training=self.training)
        # [B, nb, nb] * [B, nb, D'] -> [B, nb, D']

        h_prime = torch.matmul(attMat, h)

        h = h.cpu()
        attMat = attMat.cpu()
        aInput = aInput.cpu()
        inputTensor = inputTensor.cpu()
        del h
        del attMat
        del aInput
        del inputTensor

        return h_prime

    def propagation(self):
        """
        Multi-attribute GCN propagation
        :return: [ userFinalEmbedding, itemFinalEmbedding ]
        """
        finalEmbeds = []
        for behaviorDepth in range(self.numLayers + 1):
            singleBehaivorEmbed = self.singleBehaviorPropagation(behaviorDepth)
            finalEmbeds.append(singleBehaivorEmbed)
            singleBehaivorEmbed = singleBehaivorEmbed.cpu().detach().numpy()
            del singleBehaivorEmbed
        finalEmbedding = None
        if self.finalIntegration == 'MEAN':
            finalEmbedding = torch.stack(finalEmbeds, dim=0)
            finalEmbedding = torch.mean(finalEmbedding, dim=0)
        elif self.finalIntegration == 'NONE':
            finalEmbedding = finalEmbeds[-1]
        elif self.finalIntegration == 'ATT':
            finalEmbedding = torch.stack(finalEmbeds, dim=1)
            if self.attNorm == 'GAT-like':
                finalEmbedding = self.attLayer(finalEmbedding)
                finalEmbedding = torch.sum(finalEmbedding, dim=1)
            else:
                attWeight = self.attMlp(finalEmbedding)
                if self.attNorm == 'SOFTMAX':
                    attWeight = self.softMaxLayer(attWeight)
                elif self.attNorm == 'SUM-RATIO':
                    attSum = torch.sum(attWeight, dim=1, keepdim=True)
                    attWeight = attWeight / attSum
                finalEmbedding = finalEmbedding.permute(0, 2, 1)
                finalEmbedding = torch.matmul(finalEmbedding, attWeight).squeeze()
        userFinalEmbedding, itemFinalEmbedding = torch.split(finalEmbedding, [self.graph.numUsers, self.graph.numItems])
        return userFinalEmbedding, itemFinalEmbedding

    def singleBehaviorPropagation(self, maxDepth):
        """
        Single attribute propagation in GCN.
        :param maxDepth: maximum propagation depth
        :return: propagation tensor of users and items[finalUserEmbedding, finalItemEmbedding]
        """
        layerEmbeddings = []
        userEmbedding = self.layerUserEmbeddings[maxDepth]
        itemEmbedding = self.layerItemEmbeddings[maxDepth]
        layerEmbedding = torch.cat([userEmbedding.weight, itemEmbedding.weight])
        layerEmbeddings.append(layerEmbedding)

        def dropoutLayer(inputTensor, dropOutRatio):
            """
            Dropout Layer
            :param inputTensor: input tensors
            :param dropOutRatio: dropout ratio
            :return: output tensor
            """
            tensorIndices = inputTensor.indices().t()
            tensorValues = inputTensor.values()
            tensorSize = inputTensor.size()
            tensorMask = torch.rand(len(tensorValues)) + dropOutRatio
            tensorMask = tensorMask.int().bool()
            tensorIndices = tensorIndices[tensorMask].t()
            tensorValues = tensorValues[tensorMask] / dropOutRatio
            outputTensor = torch.sparse_coo_tensor(tensorIndices, tensorValues, tensorSize)
            return outputTensor

        graphDropOut = self.graph.getNormAdj
        if self.training:
            if self.ifDropOut:
                # Dropout part of nodes and scale remained parameter up
                graphDropOut = dropoutLayer(graphDropOut, self.keepProb)

        for layerNum in range(maxDepth):
            layerEmbedding = torch.sparse.mm(graphDropOut, layerEmbedding)
            layerEmbeddings.append(layerEmbedding)
        LayerEmbeddingStack = torch.stack(layerEmbeddings, dim=0)
        layerEmbeddingMean = torch.mean(LayerEmbeddingStack, dim=0)
        return layerEmbeddingMean

    def bprLoss(self, userIDs, posItemIDs, negItemIDs):
        allUserEmbeds, allItemEmbeds = self.propagation()
        userEmbeds = allUserEmbeds[userIDs]
        posItemEmbeds = allItemEmbeds[posItemIDs]
        negItemEmbeds = allItemEmbeds[negItemIDs]

        # Regularization Term of Final Embedding
        regEmbedTerm = userEmbeds.norm(2).pow(2) + posItemEmbeds.norm(2).pow(2) + negItemEmbeds.norm(2).pow(2)
        regEmbedTerm = regEmbedTerm / float(len(userIDs)) / 2.
        if self.attNorm == 'GAT-like':
            regEmbedTerm += self.attA.norm(2).pow(2)
            regEmbedTerm += self.attW.norm(2).pow(2) / self.attW.size()[0] / self.attW.size()[1]

        # Regularization Term of User Behavior Similarity Distance
        regBehavTerm = 0.
        for behavIterI in range(self.numLayers):
            for behavIterJ in range(behavIterI + 1, self.numLayers):
                # User Behavior Regularizaiton Term
                userBehavEmbedI = self.layerUserEmbeddings[behavIterI].weight[userIDs]
                userBehavEmbedJ = self.layerUserEmbeddings[behavIterJ].weight[userIDs]
                cosDistanceUser = torch.mul(userBehavEmbedI, userBehavEmbedJ).sum(dim=1)
                cosDistanceUser = cosDistanceUser / torch.mul(userBehavEmbedI.norm(p=2, dim=1),
                                                              userBehavEmbedJ.norm(p=2, dim=1))
                regBehavTerm += cosDistanceUser.sum() / 3.

                # Positive Item Behavior Regularization Term
                posBehavEmbedI = self.layerItemEmbeddings[behavIterI].weight[posItemIDs]
                posBehavEmbedJ = self.layerItemEmbeddings[behavIterJ].weight[posItemIDs]
                cosDistancePos = torch.mul(posBehavEmbedI, posBehavEmbedJ).sum(dim=1)
                cosDistancePos = cosDistancePos / torch.mul(posBehavEmbedI.norm(p=2, dim=1),
                                                            posBehavEmbedJ.norm(p=2, dim=1))
                regBehavTerm += cosDistancePos.sum() / 3.

                # Negative Item Behavior Regularization Term
                negBehavEmbedI = self.layerItemEmbeddings[behavIterI].weight[negItemIDs]
                negBehavEmbedJ = self.layerItemEmbeddings[behavIterJ].weight[negItemIDs]
                cosDistanceNeg = torch.mul(negBehavEmbedI, negBehavEmbedJ).sum(dim=1)
                cosDistanceNeg = cosDistanceNeg / torch.mul(negBehavEmbedI.norm(p=2, dim=1),
                                                            negBehavEmbedJ.norm(p=2, dim=1))
                regBehavTerm += cosDistanceNeg.sum() / 3.
        regBehavTerm = regBehavTerm / float(len(userIDs)) / 2.

        # BPR Loss Term
        posPreds = torch.mul(userEmbeds, posItemEmbeds)
        negPreds = torch.mul(userEmbeds, negItemEmbeds)
        softPlus = torch.nn.Softplus()
        activeDiff = softPlus(torch.sum(negPreds, dim=1) - torch.sum(posPreds, dim=1))
        loss = torch.mean(activeDiff)

        return loss, regEmbedTerm, regBehavTerm

    def getRatings(self, userIDs: list):
        """
        Accessing all rating matrix for userIDs
        :param userIDs: user ID list
        :return: according user ratings for all items
        """
        userIDs = torch.LongTensor(userIDs)
        finalUsers, finalItems = self.propagation()
        userEmbeds = finalUsers[userIDs]
        userRatings = self.activeLayer(torch.matmul(userEmbeds, finalItems.T))
        return userRatings

    def forward(self, userIDs, itemIDs):
        allUserEmbeddings, allItemEmbeddings = self.propagation()
        userEmbeds = allUserEmbeddings[userIDs]
        itemEmbeds = allItemEmbeddings[itemIDs]
        ratingPreds = torch.mul(userEmbeds, itemEmbeds)
        return ratingPreds
