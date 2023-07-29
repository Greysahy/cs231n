## assignment 2

**1.实现任意深度的全连接神经网络 **

网络结构为：`{fully-connected - [batch/layer norm] - ReLU - [dropout]} * (L - 1) - fully-connected - softmax`

共`L`层

**参数初始化**

利用循环对每一层的weights和bias进行初始化

通常weights使用`np.random.normal(mean, std, size)`进行初始化

bias初始化为全0

```python
def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 利用循环对每一层的weights和bias进行初始化
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        for i in range(len(hidden_dims)):
          self.params[f'W{i + 2}'] = np.random.normal(0, weight_scale, (hidden_dims[i], hidden_dims[i + 1]))
          self.params[f'b{i + 2}'] = np.zeros(hidden_dims[i + 1]).reshape(1, -1)
          if i >= len(hidden_dims) - 2:
            break

        self.params[f'W{self.num_layers}'] = np.random.normal(0, weight_scale, (hidden_dims[-1], num_classes))
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes).reshape(1, -1)

        # for k, v in list(self.params.items()):
        #   print(k)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
```

**前向传播 & 反向传播**

利用assignment1中实现的`affine_forward`, `affine_backward`, `relu_forward` ,`relu_backward`接口进行全连接神经网络的前向和反向传播

**前向传播**

注意：在前向传播的过程中需要保存`x,w,b`， 这些值将在反向传播求梯度时被使用

```python
forward_cache = {}  # 前向传播缓存
        for i in range(self.num_layers - 1):
          X, cache = affine_forward(X, self.params[f'W{i + 1}'], self.params[f'b{i + 1}'])
          forward_cache[f'fc{i + 1}'] = cache
          X, cache = relu_forward(X)
          forward_cache[f'relu{i + 1}'] = cache

        X, cache = affine_forward(X, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        forward_cache[f'fc{self.num_layers}'] = cache
        scores = X  # 将最后一层全连接输出的num_classes个值保存为分数，这些分数会用于test模式
```

**反向传播**

```python
		loss, dx = softmax_loss(X, y)  # 从softmax开始，计算交叉熵损失并进行梯度回传
        loss += 0.5 * self.reg * np.sum(np.square(self.params[f'W{self.num_layers}']))  # 损失累计L2正则

        # 反向传播
        dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(dx, 	forward_cache[f'fc{self.num_layers}'])
        grads[f'W{self.num_layers}'] += self.reg * self.params[f'W{self.num_layers}']

        for i in reversed(range(self.num_layers - 1)):
          dx = relu_backward(dx, forward_cache[f'relu{i + 1}'])
          dx, grads[f'W{i + 1}'], grads[f'b{i + 1}'] = affine_backward(dx,forward_cache[f'fc{i + 1}'])  
          
          
          grads[f'W{i + 1}'] += self.reg * self.params[f'W{i + 1}']  # 梯度L2正则，只需要对权重weights做正则化
          loss += 0.5 * self.reg * np.sum(np.square(self.params[f'W{i + 1}']))  # 损失L2正则
```

**2. 实现不同的optimizer（优化器）**

这一部分代码实现比较简单，把公式敲出来就行了。

**SGD-Momentum**

朴素的SGD optimizer在梯度下降中可能会困于局部最优或鞍点（grad = 0的位置）。

SGD-Momentum维护了一个不随时间变化的速度，并将梯度估计添加到该速度上。该速度可以保存一部分原有的运动状态（动量），从而使得梯度下降可以越过局部最优或鞍点。形象地说，可以SGD-Momentum的过程看作一个小球在山顶滚下山的过程。当遇到小坑（局部最优）或平地（鞍点）时，小球自身的速度使其不会被困在原地。

```
v = momentum * v - learning_rate * dw
w += v
```

```python
def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config
```

**RMSProp**

在`Adagrad`（累计梯度的平方项并使用这个平方项对梯度进行放缩, `dw /= sqrt(grad_squared + 1e-7)）`的基础上引入了动量更新，让平方梯度按照一定的比率下降，从而避免 `Adagrad `可能会出现的收敛过慢的问题。

```python
def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 对梯度平方项做动量更新 
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * np.square(dw)
    next_w = w - config['learning_rate'] * (dw / np.sqrt(config['cache'] + config['epsilon']))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config

```

**Adam**

我们想要结合动量和平方项放缩特点，于是有了**接近Adam的算法：**

```python
v = 0
grad_squared = 0
	while True:
        compute dw
        v = beta1 * v + (1 - beta1) * dx
        grad_squared = beta2 *  grad_squared + (1 - beta2) * dw * dw
        
        w -= learning_rate * v/ (sqrt(grad_squared) + 1e-7)
```

在上面的算法中`v`和`grad_squared`都被初始化为0，那么在训练初期可能会发生接近`0/0`的运算，导致溢出或者步进过大。

因此，Adam使用了一个简单的初始化校正。为了纠正`v , grad_squared`和我们真正想要的`E(dw), E(dw^2)`之间的差距，我们可以计算一下`E(v)`和`E(grad_squared)`.

以`E(grad_squared)`为例，计算结果为`E(grad_squared)` = `E(dw ^ 2) * (1 - beta2 ** t) `

由`E(ax) = aE(x)`可知， 我们`v`和`grad_squared`除掉`(1 - beta ** t)`即是对`dw`和`dw^2`的无偏估计

**python伪代码：**

```python
v = 0
grad_squared = 0
	for t in range(num_iterations):  # 实际t初始值设为1
        compute dw
        v = beta1 * v + (1 - beta1) * dx
        grad_squared = beta2 *  grad_squared + (1 - beta2) * dw * dw
        v_unbias = v / (1 - beta1 ** t)  # v的无偏估计 
        grad_squared_unbias = grad_squared / (1 - beta2 ** t)  # 梯度平方的无偏估计
        
        w -= learninig_rate * v_unbias / (sqrt(grad_squared_unbias) + 1e-7)
```

```python
def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    config['t'] += 1 
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw # 对梯度做动量更新
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * np.square(dw)  # 对梯度平方项做动量更新
    
    first_unbias = config['m'] / (1 - config['beta1'] ** config['t'])  # v的无偏估计 
    second_unbias = config['v'] / (1 - config['beta2'] ** config['t'])  # 梯度平方的无偏估计
    
    next_w = w - config['learning_rate'] * (first_unbias/(np.sqrt(second_unbias) + config['epsilon']))  # RMSProp 优化

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config

```

