## Convolutional neural networks

**卷积层前向传播**

```python
def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape 
    F, C, HH, WW = w.shape  
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_, W_))
    
    # np.pad默认填充0
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)))
    
    for out_channel in range(F):
        for i in range(H_):
          for j in range(W_):
            r = i * stride
            c = j * stride
            x_value = x_pad[:, :, r:r + HH, c:c + WW]  # (N, C, HH, WW)
            conv_kernel = w[out_channel, :, :, :] # (1, C, HH, WW)
            conv_res = np.sum(conv_kernel * x_value, axis=(1, 2, 3))
            out[:, out_channel, i, j] = conv_res
    
    out += b.reshape(1, F, 1, 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
```

**卷积层反向传播**

计算过程和前向传播类似

`dx:`x * w --> out， 对某位置上的x，`dx` = `sum`(与`x`乘过的`w` × 他们的乘积所得的out对应的上游梯度`dout`值)

`dw:`与求`dx`类似，对于某位置的w，`dw` = sum(与`w`乘过的`x` × 他们的乘积所得的out对应的上游梯度`dout`值)

`db:`bias项只是最后加在了各个filter的计算结果上，因此梯度值为1，大小`(F,)`

**`dx`和`dw`实际上可以在一套循环中求得，为了代码可读性，这里分别求解`dx`和`dw`**

```python
def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
      - x: Input data of shape (N, C, H, W)
      - w: Filter weights of shape (F, C, HH, WW)
      - b: Biases, of shape (F,)
      - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 卷积输出shape(N,F,H,W)
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    N, F, H_, W_ = dout.shape
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)))
    
    # 计算dx 
    dx = np.zeros_like(x_pad)  # 所有和x做过乘法的w的F个通道的w * 其对应的dout
    for out_channel in range(F):
        for i in range(H_):
          for j in range(W_):
            r = i * stride
            c = j * stride
            x_value = x_pad[:, :, r:r + HH, c:c + WW] # N, C, HH, WW
            conv_kernel = w[out_channel, :, :, :] # (1, C, HH, WW)
            dout_corres = dout[:, out_channel, i, j]
            local_grad = np.zeros_like(x_value) + conv_kernel.reshape(1, C, HH, WW) 
            dx[:, :, r:r + HH, c:c + WW] += local_grad * dout_corres.reshape(N, 1, 1, 1)
    
    dx = dx[:, :, 1:H + 1, 1: W + 1]
    # 计算db
    db = np.sum(dout, axis=(0, 2, 3)) 
    # 计算dw
    dw = np.zeros_like(w)  # 所有和当前位置w做过乘积的x * 其求得的out对应位置上的dout
    for out_channel in range(F):
      for i in range(H_):
        for j in range(W_):
          r = i * stride
          c = j * stride
          x_value = x_pad[:, :, r:r + HH, c:c + WW]  # (N, C, HH, WW)
          dw[out_channel, :, :, :] += np.sum(x_value * dout[:, out_channel, i, j].reshape(-1, 1, 1, 1), axis=0)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```

**最大池化前向传播**

根据最大池化的运行规则完成代码即可，难度不大

```python
def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_, W_))
    for i in range(H_):
      for j in range(W_):
        r = i * stride
        c = j * stride
        out[:, :, i, j] = np.max(x[:, :, r:r+pool_height, c:c+pool_width], axis=(2,3))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache
```

**最大池化反向传播**

重复最大池化正向过程，每次池化窗口中的最大值梯度累计加一，其他元素梯度不变。

```python
def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    dx = np.zeros(x.shape)
    
    for i in range(N):
        for c in range(C):
            cnt = 0
            for p in range(0, H - pool_height + 1, stride):
                for q in range(0, W - pool_width + 1, stride):
                    x_region = x[i, c, p:(p+pool_height), q:(q+pool_width)]
                    idx = np.argmax(x_region)
                    dx[i, c, p + int(idx/pool_height), q + int(idx%pool_height)] = dout[i, c, int(cnt/H_out), int(cnt%H_out)]
                    cnt += 1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```

**三层卷积神经网络搭建**

网络结构为：`conv - relu - 2x2 max pool - affine - relu - affine - softmax`

初始化权重并调用之前完成的接口即可。

在初始化权重时要注意维度的匹配，不要忘记`maxpooling`会降采样。

由于池化窗口的大小是确定的，我们可以直接计算出第一个全连接层的输入维度（卷积层输出维度 / 4）。

```python
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(0, weight_scale,(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*input_dim[1]*input_dim[2]//4, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_pool_out, self.params['W2'], self.params['b2'])
        scores, affine_cache= affine_forward(affine_relu_out, self.params['W3'], self.params['b3'])
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(self.params['W3'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W1'])))
        
        # 反向传播
        dx, grads['W3'], grads['b3'] = affine_backward(dx, affine_cache)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, affine_relu_cache)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, conv_relu_pool_cache)
        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

```

