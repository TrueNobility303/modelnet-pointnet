import torch
from torch_geometric.datasets.modelnet import ModelNet
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import visualize_points, visualize_graph
import torch
from torch_geometric.transforms import SamplePoints
from torch_cluster import knn_graph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from model import PointNet
import tqdm 
from torch_geometric.transforms import SamplePoints
import matplotlib.pyplot as plt 
from torch_cluster import fps
from model import PFFNet

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

dataset = ModelNet(root='./dataset', name='10',train=True,transform=SamplePoints(128))
testset = ModelNet(root='./dataset', name='10',train=False,transform=SamplePoints(128))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
testloader = DataLoader(testset, batch_size=10, shuffle=True)

class PointNetTrainer():
    def __init__(self):
        self.model = PointNet().to(device)
        #self.model = PFFNet().to(device)
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()  
        self.epoches = 20

    def train(self,loader):
        self.model.train()
        
        total_loss = 0
        tot_num = 0
        for i,data in enumerate(tqdm.tqdm(loader)):
            tot_num += len(data.y)
            data = data.to(device)
            self.optimizer.zero_grad()  
            logits = self.model(data.pos, data.batch)  
            loss = self.criterion(logits, data.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()

        return total_loss / tot_num

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        tot_num = 0
        for i,data in enumerate(loader):
            if i > 50:
                break
            tot_num += len(data.y)
            data = data.to(device)
            logits = self.model(data.pos, data.batch)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == data.y).sum()

        #print(total_correct,tot_num)
        return total_correct / tot_num

    def work(self,train_loader,test_loader):
        plt.figure()
        losses = []
        accs = []
        for epoch in range(self.epoches):
            loss = self.train(train_loader)
            acc = self.test(test_loader)
            print("epoch",epoch,"loss",loss,"acc",acc)
            losses.append(loss)
            accs.append(acc)
        
        plt.plot(losses)
        plt.plot(accs)
        plt.legend(['loss','acc'])
        plt.savefig('dump/curve.png')

if __name__ == '__main__':
    model = PointNetTrainer()
    model.work(dataloader,testloader)



