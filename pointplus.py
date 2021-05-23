import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, knn, radius, global_max_pool
import matplotlib.pyplot as plt 
from torch_cluster import knn_graph, radius_graph
from torch_geometric.nn import PPFConv
import tqdm 
from torch import nn 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 在PFF的基础上加入PPF卷积

#聚合局部信息
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PPFConv(nn)

    def forward(self, x, pos, norm, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        #可以用radius或nerest构建半径内或者最近邻图
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]),(norm,norm[idx]), edge_index)
        pos, norm, batch = pos[idx], norm[idx], batch[idx]
        return x, pos, norm, batch

#聚合全局信息
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, norm, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, norm, batch

#用列表形式定义多层感知机，在每层线性层中间添加BN层和Relu
def MLP(channels, batch_norm=True):
    channels = channels
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class PointMLP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.lrb = Seq(Lin(in_channels, out_channels), ReLU(), BN(out_channels))
    
    def forward(self,x):
        x = self.lrb(x)
        return x 

#网络主体结构，由两层局部汇聚层，一层全局汇聚层，和最终的分类器组成
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, PointMLP(4,32))
        self.sa2_module = SAModule(0.25, 0.4, PointMLP(32 + 4, 64))
        self.sa3_module = GlobalSAModule(PointMLP(64 + 3, 128))
        self.classifier = Lin(128, 10)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.normal, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, norm, batch = sa3_out
        x = F.relu(self.classifier(x))
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    tot_loss = 0
    tot_num = 0
    for i,data in tqdm.tqdm(enumerate(train_loader)):
        data = data.to(device)
        tot_num += len(data.y)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    
    return tot_loss / tot_num


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    path = ('dataset')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(128,include_normals=True)
    train_dataset = ModelNet(path, '10', True, transform=transform)
    test_dataset = ModelNet(path, '10', False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    accs = []
    try:
        for epoch in range(200):
            loss = train(epoch)
            test_acc = test(test_loader)
            print('epoch',epoch,'loss',loss,'acc',test_acc)
            losses.append(loss)
            accs.append(test_acc)
        plt.figure()
        plt.plot(losses)
        plt.plot(accs)
        plt.legend(['loss','acc'])
        plt.savefig('dump/pointnet.png')
    except KeyboardInterrupt:
        plt.figure()
        plt.plot(losses)
        plt.plot(accs)
        plt.legend(['loss','acc'])
        plt.savefig('dump/pointnet_curve.png')
        
