# 图像分类

### 数据驱动方法：

1. 收集包含图像和标签的数据集
2. 使用机器学习算法训练一个分类器
3. 在新图像样本上对分类器进行评估

## KNN（k nearest neighbors）

比较两幅图片的metric:

 $L1-distance : d(I_1, I_2) = \sum |I_1^{pixel} - I_2^{pixel}|$

![image-20230722161115661](C:\Users\AnHengyu\AppData\Roaming\Typora\typora-user-images\image-20230722161115661.png)

在不同距离度量下与原点等距离的点的集合：

![image-20230722164210719](C:\Users\AnHengyu\AppData\Roaming\Typora\typora-user-images\image-20230722164210719.png)



## Linear 线性分类器

![image-20230722172247454](C:\Users\AnHengyu\AppData\Roaming\Typora\typora-user-images\image-20230722172247454.png)

线性分类器的分类更像一个模板匹配问题，W和b中的每一行对应某一类图像的模板。

