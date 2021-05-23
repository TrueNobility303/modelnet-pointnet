# modelnet 3D物体分类

## 摘要

在ModelNet数据集上，实现了基于点云的3D物体分类模型
* PointNet
* PointNet++
* PPF

在10min以内的训练内，可以达到70%以上的准确率


## 文件说明
* /dump /backup 为输出内容
* model.py 网络定义
* utils.py 定义可视化点云等函数
* main.py point.py pointplus.py 使用了不同的网络结构，实现点云分类主函数，
