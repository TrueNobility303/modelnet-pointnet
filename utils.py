import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.datasets.modelnet import ModelNet
import torch

def visualize_points(pos, savepath):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:,2])
    plt.axis('off')
    plt.savefig(savepath)

def visualize_graph(pos,edge_index,savepath):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (src, dst) in edge_index.t().tolist():
        src = pos[src].tolist()
        dst = pos[dst].tolist()
        plt.plot([src[0], dst[0]], [src[1], dst[1]])
    ax.scatter(pos[:, 0], pos[:, 1], pos[:,2])
    plt.axis('off')
    plt.savefig(savepath)

dataset = ModelNet(root='./dataset', name='10')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    data = dataset[2]
    visualize_points(data.pos,"dump/point.png")
    data.edge = knn_graph(data.pos, k=12)
    visualize_graph(data.pos, data.edge, 'dump/point_graph.png')
