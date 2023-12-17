import torch_geometric
# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.nn import Linear
from torch.nn import init
from torch.nn import Tanh
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)


# BASE model
class GCN_Less_layer_model_tanh(torch.nn.Module):
    def __init__(self,data_num_features,embedding_size):
        torch.manual_seed(seed_value)
        # Init parent
        super(GCN_Less_layer_model_tanh, self).__init__()

        # define layer
        self.initial_conv = GCNConv(data_num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
    
        # define linear layer
        self.out = Linear(embedding_size*2, 1)
    
    def init_weights(self):
        # Use Xavier/Glorot initialization for GCNConv layers
        for layer in [self.initial_conv, self.conv1]:
            if isinstance(layer, GCNConv):
                # layer.reset_parameters()
                if isinstance(layer, GCNConv):
                    init.normal_(layer.lin.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

        # Initialize weights for the Linear layer
        init.xavier_normal_(self.out.weight)
        if self.out.bias is not None:
            init.zeros_(self.out.bias)

    def forward(self, x, edge_index, batch_index):
        
        # first layer
        hidden = self.initial_conv(x, edge_index)
        hidden = Tanh(hidden)
        # second layer
        hidden = self.conv1(hidden, edge_index)
        hidden = Tanh(hidden)

        # global pooling
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
                            

        # apply linear layer
        out = self.out(hidden)
        return out
