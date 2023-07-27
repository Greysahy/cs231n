# cs231n
2023年夏 cs231n计算机视觉课程学习记录

## assignment 1

**1. KNN（L2-distance）的三种实现方式：双层循环，单层循环，无循环向量化实现**

主要通过numpy的广播机制加速运算

向量化实现原理：$(a - b)^2 = a^2 + b^2 - 2ab$

$a$为测试集样本，维度 (M, D)

$b$为训练集样本，维度 (N, D)

```python
ab = np.dot(a, b.T) # 维度(M， N)
a2 = np.sum(np.square(a), axis=1).reshape(M, 1)
b2 = np.sum(np.square(b), axis=1).reshape(1, N)
L2-distance = -2 * ab + a2 + b2  # 维度(M, N)
```

reshape之后，$a^2$和$b^2$就都满足了广播的条件，可以直接计算L2-distance

**2. SVM loss的两种实现方式：循环和向量化实现**
$$
SVM\ loss = 
\left\{
\begin{aligned}
%\nonumber
0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ {s_i  + 1(1为阈值) < s_j}\\
s_i - s_j + 1\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ {else}\\
\end{aligned}
\right.
\\
\frac{\partial{L}}{\partial{s_i}} = 1, \frac{\partial{L}}{\partial{s_j}} = -1(损失不为0时) \\
\frac{\partial{L}}{\partial{w_i}} = \frac{\partial{L}}{\partial{s_i}} \cdot \frac{\partial{s_i}}{\partial{w_i}} = \pm1 \cdot x_i 
$$
**SVM损失值与梯度值计算的向量化实现**

```python
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - X: A numpy array of shape (N, D) containing a minibatch of data.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W
	"""
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	# 计算损失
    num_train = X.shape[0]  
    scores = np.dot(X, W)  # (N, C)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)  # 标签类别对应的分数,(N, ) 
    
    margins = scores - correct_class_score + 1.0  # si - sj + 1,(N, C)
    margins[np.arange(num_train), y] = 0  # 对于标签类别不计算损失
    margins[margins < 0] = 0.0  # 将小于0的损失值计为0
    
    loss = np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)  # L2正则， 为了抵消梯度的正则系数，乘0.5

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1.0  # si对si-sj+1求导为1
    x_count = np.sum(margins, axis=1)  # 对每一个样本，有多少类别的分数计算了si-sj+1的损失
    margins[np.arange(num_train), y] -= x_count  # 标签类别对损失累计一个 -1 的梯度（sj对si-sj+1求导）
    dW = np.dot(X.T, margins) /num_train # s = xi * wi，对参数w求导=x,因
    dW = dW + reg * W  # L2正则
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```

**3. Softmax loss的两种实现方式：循环和向量化实现**
$$
Softmax\ loss = -ln({y_j}\cdot p_j),y_j = 1,  p_j = \frac{e^{s_j}}{\sum_ke^{s_k}} \\
\frac{\partial{L}}{\partial{p_j}} = -\frac{1}{p_j}\\
对标签类别分数s_j的导数 = \frac{\partial{p_j}}{\partial{s_j}} = \frac{e^{s_j} \cdot (\sum_ke^{s_k} - e^{s_j})}{(\sum_ke^{s_k})^2} = p_j \cdot (1 - p_j) \\
对非标签类别分数s_i的导数 = \frac{\partial{p_j}}{\partial{s_i}} = \frac{0 \cdot \sum_ke^{s_k} - e^{s_j} \cdot e^{s_i}}{(\sum_ke^{s_k})^2} = -p_i p_j \\
\frac{\partial{s_i}}{\partial{w_i}} = x_i \\
根据链式法则： \\
\frac{\partial{L}}{\partial{w_j}} = -\frac{1}{p_j} \cdot p_j \cdot (1 - p_j) \cdot x_i = p_j - 1\\
\frac{\partial{L}}{\partial{w_i}} = -\frac{1}{p_j} \cdot (-p_i  p_j) \cdot x_i = p_i
$$
**Softmax损失值与梯度值计算的向量化实现**

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    # compute loss
    scores = np.exp(np.dot(X, W))  # exp(s_i)， (N, C)
    p = scores / np.sum(scores, axis=1).reshape(num_train, 1)  #计算出每一个类别的概率p_i (N, C)
    loss += np.sum(-np.log(p[np.arange(num_train), y]))  # 按照公式，损失值等于标签类别概率的负ln值
    
    # compute gradient
    # dW (D, C) X(N, D) 
    p[np.arange(num_train), y] -= 1  # 标签类别的梯度等于当前类别概率 - 1， 非标签类别的梯度等于当前类别概率
    dW += np.dot(X.T, p)  # 根据公式乘x计算损失对参数w的梯度

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)  # L2正则
    dW /= num_train
    dW += reg * W  # l2正则

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
```
