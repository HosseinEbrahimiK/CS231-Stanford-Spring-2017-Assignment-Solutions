import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None

        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_relu_forward(out1, self.params['W2'], self.params['b2'])
        
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, ds = softmax_loss(scores, y) 
        loss += (self.reg/2) * (np.sum(self.params['W1']**2, axis=None) + np.sum(self.params['W2']**2, axis=None))
        
        dout1, grads['W2'], grads['b2'] = affine_relu_backward(ds, cache2)
        grads['W2'] += (self.reg * self.params['W2'])
        
        _, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)
        grads['W1'] += (self.reg * self.params['W1'])
        
        return loss, grads

    
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        for layer in range(self.num_layers):
        
            if layer == 0:
                self.params['W'+str(layer+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[layer])
                self.params['b'+str(layer+1)] = np.zeros(hidden_dims[layer])
                if self.use_batchnorm:
                    self.params['gamma'+str(layer+1)] = np.ones(hidden_dims[layer])
                    self.params['beta'+str(layer+1)] = np.zeros(hidden_dims[layer])
                
            elif layer == self.num_layers-1:
                self.params['W'+str(layer+1)] = weight_scale * np.random.randn(hidden_dims[layer-1], num_classes)
                self.params['b'+str(layer+1)] = np.zeros(num_classes)
                
            else:
                self.params['W'+str(layer+1)] = weight_scale * np.random.randn(hidden_dims[layer-1], hidden_dims[layer])
                self.params['b'+str(layer+1)] = np.zeros(hidden_dims[layer])
                if self.use_batchnorm:
                    self.params['gamma'+str(layer+1)] = np.ones(hidden_dims[layer])
                    self.params['beta'+str(layer+1)] = np.zeros(hidden_dims[layer])
                
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers+1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        cache = {'relu' : [None for _ in range(self.num_layers+1)], 'dropout': [None for _ in range(self.num_layers+1)], 
                'bn': [None for _ in range(self.num_layers+1)]}
        
        out = [None for _ in range(self.num_layers+1)]
        out[0] = X.copy()

        for i in range(1, self.num_layers+1):
            ind = str(i)
            if i == self.num_layers:
                out[i], cache['relu'][i] = affine_forward(out[i-1], self.params['W'+ind], self.params['b'+ind])
                scores = out[i].copy() 
            else:
                if self.use_batchnorm:
                    out[i], cache['bn'][i] = affine_bn_relu_forward(out[i-1], self.params['W'+ind], self.params['b'+ind],           self.params['gamma'+ind], self.params['beta'+ind], self.bn_params[i])
                else:
                    out[i], cache['relu'][i] = affine_relu_forward(out[i-1], self.params['W'+ind], self.params['b'+ind])
                    
                if self.use_dropout:
                    out[i], cache['dropout'][i] = dropout_forward(out[i], self.dropout_param)

        # If test mode return early
        
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        dout = [None for _ in range(self.num_layers+1)]
        loss, dout[-1] = softmax_loss(scores, y)
        loss += (self.reg/2) * np.sum([(np.sum(self.params['W'+str(i)]**2, axis=None)) for i in range(1, self.num_layers+1)])
        
        for i in range(self.num_layers, 0, -1):
            ind = str(i)
            if i == self.num_layers:
                dout[i-1], grads['W'+ind], grads['b'+ind] = affine_backward(dout[i], cache['relu'][i])
                grads['W'+ind] += self.reg * self.params['W'+ind]
            else:
                if self.use_dropout:
                    dout[i] = dropout_backward(dout[i], cache['dropout'][i])
                    
                if self.use_batchnorm:
                    dout[i-1], grads['W'+ind], grads['b'+ind], grads['gamma'+ind], grads['beta'+ind] =affine_bn_relu_backward(dout[i], cache['bn'][i])
                    grads['W'+ind] += self.reg * self.params['W'+ind]
                else:
                    dout[i-1], grads['W'+ind], grads['b'+ind] = affine_relu_backward(dout[i], cache['relu'][i])
                    grads['W'+ind] += self.reg * self.params['W'+ind]
            
        return loss, grads
