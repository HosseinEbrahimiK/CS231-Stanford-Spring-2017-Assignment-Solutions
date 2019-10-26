import numpy as np
import matplotlib.pyplot as plt
from load_data import load_CIFAR10
from math import sqrt, ceil


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        score1 = np.matmul(X, W1) + b1
        relu_score1 = np.maximum(score1, 0)
        scores = np.matmul(relu_score1, W2) + b2

        if y is None:
            return scores

        loss = None
        scores = np.exp(scores)
        row_sum = np.sum(scores, axis=1)
        scores = (scores.T / row_sum).T
        loss = (np.sum(-np.log(scores[np.arange(N), y]))/N) + reg*(np.sum(W1*W1, axis=None) + np.sum(W2*W2, axis=None))
        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        pass
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        scores[np.arange(scores.shape[0]), y] -= 1

        grads['b2'] = np.sum(scores, axis=0) / N
        grads['W2'] = (np.matmul(relu_score1.T, scores) / N) + (reg * 2 * W2)

        mask = np.where(score1 > 0, 1, 0)
        temp = np.matmul(scores, W2.T) * mask

        grads['b1'] = np.sum(temp, axis=0) / N
        grads['W1'] = (np.matmul(X.T, temp) / N) + (reg * 2 * W1)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            idx = np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['b2'] -= -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):

        y_pred = None
        s1 = np.matmul(X, self.params['W1']) + self.params['b1']
        s1 = np.maximum(s1, 0)
        s = np.matmul(s1, self.params['W2']) + self.params['b2']
        s = np.exp(s)
        exp_sum = np.sum(s, axis=1)
        s = (s.T / exp_sum).T
        y_pred = np.argmax(s, axis=1)

        return y_pred


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
      a naive implementation of numerical gradient of f at x
      - f should be a function that takes a single argument
      - x is the point (numpy array) to evaluate the gradient at
      """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad


def visualize_grid(Xs, ubound=255.0, padding=1):

    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
          if next_idx < N:
            img = Xs[next_idx]
            low, high = np.min(img), np.max(img)
            grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)

            next_idx += 1
          x0 += W + padding
          x1 += W + padding
        y0 += H + padding
        y1 += H + padding

    return grid


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


if __name__ == "__main__":
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    net = init_toy_model()
    X, y = init_toy_data()

    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()

    print('Difference between your scores and correct scores:')
    print(np.sum(np.abs(scores - correct_scores)))

    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133

    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))

    loss, grads = net.loss(X, y, reg=0.05)

    for param_name in grads:
        f = lambda W: net.loss(X, y, reg=0.05)[0]
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    num_training = 49000
    num_validation = 1000
    num_test = 1000

    X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10

    net = TwoLayerNet(input_size, hidden_size, num_classes)

    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=1000, batch_size=200,
                      learning_rate=1e-4, learning_rate_decay=0.95,
                      reg=0.25, verbose=True)

    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)

    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

    show_net_weights(net)

    h = [130]
    lr = [0.001]
    reg = [0.3]

    best_net = None
    best_val = -1

    for hiddenSize in h:
        for learningRate in lr:
            for regular in reg:

                net = TwoLayerNet(input_size, hiddenSize, num_classes)
                stat = net.train(X_train, y_train, X_val, y_val,
                                 num_iters=1500, batch_size=200,
                                 learning_rate=learningRate, learning_rate_decay=0.95,
                                 reg=regular, verbose=True)

                validation_acc = (net.predict(X_val) == y_val).mean()

                if best_val < validation_acc:
                    best_val = validation_acc
                    best_net = net

                print("************************")
                print("hidden size :", hiddenSize)
                print("learning rate :", learningRate)
                print("reg :", regular)
                print()
                print("Accuracy on validation : ", validation_acc)
                print("************************")

    test_acc = (best_net.predict(X_test) == y_test).mean()
    print('Test accuracy: ', test_acc)

    show_net_weights(best_net)
