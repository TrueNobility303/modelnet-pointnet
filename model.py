from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv import PPFConv
from torch_cluster import fps
from torch import Tensor

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PointNetLayer, self).__init__('max')
        
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        input = pos_j - pos_i 
        if h_j is not None:
            inputs = torch.cat([h_j, inputs], dim=-1)
        return self.mlp(inputs)  

class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        torch.manual_seed(42)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, 10)
        
    def forward(self, pos, batch):
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        
        h = global_max_pool(h, batch)
        h = self.classifier(h)
        y = torch.sigmoid(h)
        return y


if __name__ == '__main__':
    model = PFFNet()
    print(model)
