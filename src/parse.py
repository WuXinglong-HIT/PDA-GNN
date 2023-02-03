import argparse
import torch


class argParser:
    # BATCH SIZE
    def __init__(self):
        pass

    batch_size = 10240
    # TEST USER BATCH SIZE
    test_batch_size = 10240
    # Embedding Dimension
    embedding_dim = 128
    # Final Embedding Integration: "MEAN" or "ATT" or "NONE"
    # "MEAN" means that final embedding is integrated equally with different behavior embeddings
    # "ATT" means that final embedding is integrated with attented behavior embeddings
    # "NONE" means that final embedding is replaced with the last behavior embedding
    final_integration = "ATT"
    # final_integration = "MEAN"
    # final_integration = "NONE"
    # Attention MLP Hidden Layer Dimensions
    att_hidden_dims = [16, 1]
    # Attention Normalization Method: "SUM-RATIO" or "SOFTMAX" or "GAT-like"
    att_norm = "GAT-like"
    # att_norm = "SUM-RATIO"
    # att_norm = "SOFTMAX"
    # GCN Layer Depth
    num_layers = 3
    # Learning Rate
    lr = 1e-3
    # Epochs
    epochs = 500
    # Global Random Seed
    seed = 2022
    # Model Name as Comment
    comment = "PDAGNN-"
    # Dataset
    dataset = "Amazon-Book"
    # dataset = "Amazon-CDs"
    # dataset = "MovieLens-1M"
    # Testing Metrics
    topK = [5, 10, 20]
    # Embedding Regularization Weight
    weight_decay_embed = 1e-4
    # Behavior Regularization Weight
    # weight_decay_behavior = 1e-3
    weight_decay_behavior = 0.
    # if Behavior Regularization Term added in the loss function
    ifRegBehav = True
    # if Embedding Regularization Term added in the loss function
    ifRegEmbedding = True
    # if model is loaded from file
    ifLoad = False
    # whether apply dropout
    ifDropOut = True
    # Dropout Ratio
    keep_prob = 0.6
    # Loaded Model Filename
    load_model_name = "PDA-GNN-epoch-5.pth.tar"
    # IS CUDA AVAILABLE
    CUDA_AVAILABLE = torch.cuda.is_available()
    # DEVICE
    DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"

# def argParser():
#     parser = argparse.ArgumentParser(description="PDAGNN - Propagation-Depth-Aware
#                                        Graph Neural Networks for Recommendation")
#     parser.add_argument('--batch_size', type=int, default=2048,
#                         help="batch size")
#     parser.add_argument('--em_dim', type=int, default=64,
#                         help="the embedding size of PDA-GNN")
#     parser.add_argument('--lr', type=float, default=1e-3,
#                         help="learning rate")
#     return parser.parse_args()