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

```python
def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		
        # 向量化运算计算KNN L2-distance
        test_train = np.dot(X, self.X_train.T)  #  (num_test, num_train)
        square_test = np.sum(np.square(X), axis=1, keepdims=True)  #  (num_test, 1)
        square_train = np.sum(np.square(self.X_train), axis=1, keepdims=True).T #  (1, num_train)
        dists = np.sqrt(-2 * test_train + square_test + square_train)  # broadcast square_test & square_train

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```



**2. SVM loss的两种实现方式：循环和向量化实现**

![svm](https://github.com/Greysahy/cs231n/blob/main/images/svm.png)

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

![softmax](https://github.com/Greysahy/cs231n/blob/main/images/softmax.png)

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

**4. 双层全连接神经网络的实现**

**Fully-connectd 前向传播**

```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_ = x.reshape(x.shape[0], w.shape[0])  # 对输入进行flatten
    out = np.dot(x_, w) + b  # FC前向操作：x_input与weights相乘

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache
```

**Fully-connected 反向传播**

根据链式法则，损失对当前层参数的梯度 = 上游梯度dout * 本层局部梯度 

对于FC层，Local gradient 计算如下：

dx : 对于$x_i$,当前层输出中仅有$w_i * x_i$这一项和其相关，因此dx = w * dout

dw: 与dx同理，dw = x * dout

db：wx + b对偏置项b直接求导导数为1，因此db = 1 * dout

```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = dout.shape[0]  
    M = dout.shape[1]
    D = w.shape[0]

    dx = np.dot(dout, w.T).reshape(x.shape)  # 
    dw = np.dot(x.reshape(N,D).T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```

**ReLU的前向/反向传播**

前向传播 ： x = max(0, x)

反向传播 ： dx = [0(x <= 0) / 1(x > 0)] * dout

```python
def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x[x > 0] = 1
    x[x <= 0] = 0
    dx = dout * x
    return dx
```

**双层全连接神经网络的搭建（依赖前面实现的各种功能）**

**要求：**

搭建一个双层的全连接神经网络，包含ReLU的非线性激活和softmax损失

假设输入维度为D，隐藏层维度为H ，分类类别数为C

网络结构为： Fully-connected - relu - Fully-connected - softmax.

不需要实现梯度下降

将可学习的参数（W1, b1, W2, b2）存放在一个字典里

**损失与梯度计算**

```python
def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        # 前向传播过程
        affine_output, affine_cache = affine_forward(X, W1, b1)  # 1st FC layer
        relu_output, relu_cache = relu_forward(affine_output)  # ReLU激活
        scores, cache = affine_forward(relu_output, W2, b2)  # 2rd FC layer
 
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
 		
        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (W1 * W1).sum() + 0.5 * self.reg * (W2 * W2).sum()  # 正则化
        
        # 反向传播过程
        dx, grads['W2'], grads['b2'] = affine_backward(dx, cache)
        dx = relu_backward(dx, relu_cache)
        dx, grads['W1'], grads['b1'] = affine_backward(dx, affine_cache)
        
        grads['W1'] += self.reg * W1  # 正则化
        grads['W2'] += self.reg * W2  # 正则化

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```

**使用高级特征代替原始像素值**

本节作业直接调用接口提取高级特征，作为输入进行训练，训练过程的代码与之前的作业差别不大。
