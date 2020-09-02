from builtins import range
import numpy as np


def affine_forward(x, w, b):

    dims = x.shape
    z = x.reshape(dims[0], np.prod(dims[1:]))
    out = np.matmul(z, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):

    x, w, b = cache
    dims = x.shape
    z = x.reshape(dims[0], np.prod(dims[1:]))

    db = np.sum(dout, axis=0)
    dw = np.matmul(z.T, dout)
    dx = np.matmul(dout, w.T).reshape(dims)

    return dx, dw, db


def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    mask = np.where(x > 0, 1, 0)
    dx = dout * mask
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        
        mean_batch = np.mean(x, axis=0)
        var_batch = np.var(x, axis=0)
        
        x_hat = (x - mean_batch) / np.sqrt(var_batch + eps)
        out = gamma * x_hat + beta
        cache = {'x':x, 'x_hat':x_hat, 'mean_batch':mean_batch, 'var_batch':var_batch, 'gamma':gamma, 'eps':eps}
        
        running_mean = momentum * running_mean + (1 - momentum) * mean_batch
        running_var = momentum * running_var + (1 - momentum) * var_batch
        
    elif mode == 'test':
        
        out = gamma * ((x - running_mean) / np.sqrt(running_var + eps)) + beta
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    x = cache['x']
    x_hat = cache['x_hat']
    mean_batch = cache['mean_batch']
    var_batch = cache['var_batch']
    gamma = cache['gamma']
    eps = cache['eps']
   
    N = x.shape[0]
    
    dgamma = np.diag(np.matmul(x_hat.T, dout))
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma
    
    dvar_batch = np.diag(np.matmul(dx_hat.T, x - mean_batch)) * (-0.5 * (var_batch + eps) ** -1.5)
    dmean_batch = -np.sum(dx_hat, axis=0) / np.sqrt(var_batch + eps)
    dx = dx_hat / (np.sqrt(var_batch + eps))
    
    dmean_batch += (np.sum(x - mean_batch) * (-2 / N)) * dvar_batch # actually this part is zero
    dx += (2 * (x - mean_batch) / N) * dvar_batch
    
    dx += (dmean_batch / N)
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    
    x_hat = cache['x_hat']
    var_batch = cache['var_batch']
    gamma = cache['gamma']
    eps = cache['eps']
    
    N = x_hat.shape[0]
    
    dgamma = np.diag(np.matmul(x_hat.T, dout))
    dbeta = np.sum(dout, axis=0)
    
    dx = (N * gamma * dout - dbeta * gamma - dgamma * gamma * x_hat) / (N * np.sqrt(var_batch + eps))
    
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = (mask * x)
    elif mode == 'test':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = (dout * mask)
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    dims = x.shape
    out_H = int(1 + (dims[2] + 2 * pad - w.shape[2]) / stride)
    out_W = int(1 + (dims[3] + 2 * pad - w.shape[3]) / stride)
    
    out = np.zeros((x.shape[0], w.shape[0], out_H, out_W))
    
    width, height = w.shape[3], w.shape[2]
    
    z = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values= ((0, 0), (0, 0), (0, 0), (0, 0)))
    
    for n_tr in range(dims[0]):
        for n_f in range(w.shape[0]):
            
            posx, posy = 0, 0
            
            for i in range(out_H):
                
                for j in range(out_W):
                    
                    tmp = z[n_tr, :, posx:posx+height, posy:posy+width]
                    vecx = tmp.reshape(np.prod(tmp.shape))
                    vecw = w[n_f].reshape(np.prod(w[n_f].shape))
                    out[n_tr, n_f, i, j] = np.dot(vecx, vecw) + b[n_f]
                    
                    posy += stride
                
                posy = 0
                posx += stride

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, stride, pad = cache[0], cache[1], cache[2], cache[3]['stride'], cache[3]['pad']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values= ((0, 0), (0, 0), (0, 0), (0, 0)))
    
    dx, dw, db = np.zeros_like(x_pad), np.zeros_like(w), np.zeros_like(b)
    
    width, height = w.shape[3], w.shape[2]
    dims = dout.shape
    
    for n_tr in range(dims[0]):
        for n_f in range(dims[1]):
            
            posx, posy = 0, 0
            
            for i in range(dims[2]):
                
                for j in range(dims[3]):
                    
                    db[n_f] += dout[n_tr, n_f, i, j]
                    dw[n_f] += dout[n_tr, n_f, i, j] * x_pad[n_tr, :, posx:posx+width, posy:posy+height]
                    dx[n_tr, :, posx:posx+height, posy:posy+width] += dout[n_tr, n_f, i, j] * w[n_f]
                    
                    posy += stride
                
                posy = 0
                posx += stride
    
    dx = dx[:, :, 1:-1, 1:-1]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    dims = x.shape
    
    out_H = int(((dims[2] - pool_height) / stride) + 1)
    out_W = int(((dims[3] - pool_width) / stride) + 1)
    
    out = np.zeros((dims[0], dims[1], out_H, out_W))
    
    for n_tr in range(dims[0]):
        for ch in range(dims[1]):
            
            posx, posy = 0, 0
            
            for i in range(out_H):
                for j in range(out_W):
                    
                    out[n_tr, ch, i, j] = np.max(x[n_tr, ch, posx:posx+pool_height, posy:posy+pool_width])
                    posy += stride
                
                posy = 0
                posx += stride
                
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache[0]
    pool_height = cache[1]['pool_height']
    pool_width = cache[1]['pool_width']
    stride = cache[1]['stride']
    
    dx = np.zeros_like(x)
    dims = dout.shape
    
    for n_tr in range(dims[0]):
        for ch in range(dims[1]):
            
            posx, posy = 0, 0
            
            for i in range(dims[2]):
                for j in range(dims[3]):
                    
                    mask = np.zeros((pool_height, pool_width))
                    window = x[n_tr, ch, posx:posx+pool_height, posy:posy+pool_width].copy()
                  
                    mask[np.unravel_index(np.argmax(window, axis=None), window.shape)] = 1
                    
                    dx[n_tr, ch, posx:posx+pool_height, posy:posy+pool_width] = mask * dout[n_tr, ch, i, j]
                    posy += stride
                    
                posy = 0
                posx += stride
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    dims = x.shape
    x = x.reshape(np.prod(dims[0]*dims[2]*dims[3]), dims[1])
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(dims)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    dims = dout.shape
    dout = dout.reshape(np.prod(dims[0]*dims[2]*dims[3]), dims[1])
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(dims)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
