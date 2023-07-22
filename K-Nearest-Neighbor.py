import numpy as np

# 曼哈顿距离L1-distance : d1(I_1, I_2) = \sum abs(I_1^{pixel} - I_2^{pixel})
class NearestNeighbor:  # 最近邻
    def __init__(self):
        pass

    def train(self, x, y):
        """size(x）= N × D, 即共N行，每行包含了一个样本"""
        """size(y) = N, 对应着N个样本的标签"""
        """最近邻的训练器只是简单地记住所有的训练数据"""
        self.xtr = x
        self.ytr = y

    def predict(self, x):
        """size(x) = M × D, 是一个包含 M 个样本的测试集"""
        num_test = x.shape[0]
        ypred = np.zeros(num_test, dtype=self.ytr.dtye)

        for i in range(num_test):
            distances = np.sum(np.abs(self.xtr - x[i, :]), axis=1) # 计算L1距离
            min_index = np.argmin(distances)
            ypred[i] = self.ytr[min_index]

        return ypred


""" 
k最近邻：选取训练集中距离测试样本最近的k个点进行投票
该方法对噪声样本具有鲁棒性
"""

# 欧几里得距离L2-distance : d2(I_1, I_2) = sqrt( \sum (I_1^{pixel} - I_2^{pixel})^2 )
"""
L1距离取决于你选择的坐标系统，如果你转动坐标轴会改变点之间的L1距离
而L2距离是一个确定的距离，与坐标轴无关
因此，如果输入特征向量中的一些值有一些重要意义，那么L1距离会更合适
而如果特征向量只是某个空间中的通用向量，我们不知道其中的元素代表的实际意义，那么L2距离可能会更合适 
"""



