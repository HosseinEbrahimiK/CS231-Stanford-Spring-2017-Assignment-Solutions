import numpy as np
import time
import matplotlib.pyplot as plt
from random import randrange
from load_data import load_CIFAR10


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            idx = np.random.choice(X.shape[0], batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grad = self.loss(X_batch, y_batch, reg)
            self.W += -learning_rate * grad
            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):

        y_pred = np.zeros(X.shape[0])
        score = np.exp(np.matmul(X, self.W))
        row_sum = np.sum(score, axis=1)
        y_pred = np.argmax((score.T / row_sum).T, axis=1)
        return y_pred


class Softmax(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


def softmax_loss_naive(W, X, y, reg):

    num_tr = X.shape[0]
    num_classes = W.shape[1]

    loss = 0.0
    dW = np.zeros_like(W)

    score = np.matmul(X, W)

    for i in range(num_tr):
        tmp_loss = 0
        for j in range(num_classes):
            tmp_loss += np.exp(score[i][j])

        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += (X[i].T / tmp_loss) * np.exp(score[i][j])

        dW[:, y[i]] += (X[i] * ((np.exp(score[i][y[i]]) / tmp_loss) - 1))

        loss += -(score[i][y[i]] - np.log(tmp_loss))

    loss = (loss / num_tr) + reg * np.sum(W * W, axis=None)

    dW = (dW / num_tr) + (2 * reg * W)

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):

    num_tr = X.shape[0]

    loss = 0.0
    dW = np.zeros_like(W)

    score = np.matmul(X, W)
    exp_score = np.exp(score)
    row_sum = np.sum(exp_score, axis=1)

    loss += np.sum(np.log(row_sum)) - np.sum(score[np.arange(num_tr), y])
    loss = (loss / num_tr) + reg * np.sum(W * W, axis=None)

    coefficient = (exp_score.T / row_sum).T
    coefficient[np.arange(num_tr), y] -= 1

    dW = np.matmul(X.T, coefficient)
    dW = (dW / num_tr) + (2 * reg * W)

    return loss, dW


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_CIFAR10("cifar-10-batches-py")
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    W = np.random.randn(3073, 10) * 0.0001
    loss, _ = softmax_loss_naive(W, X_dev, y_dev, 0.0)

    # As a rough sanity check, our loss should be something close to -log(0.1).
    print('loss: %f' % loss)
    print('sanity check: %f' % (-np.log(0.1)))

    _, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad, 10)

    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
    grad_check_sparse(f, W, grad, 10)

    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
    print('Gradient difference: %f' % grad_difference)

    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    for reg in regularization_strengths:
        for lr in learning_rates:

            classifier = Softmax()
            classifier.train(X_train, y_train, learning_rate=lr, reg=reg, batch_size=300, num_iters=2000, verbose=True)
            train_accuracy = np.mean(classifier.predict(X_train) == y_train)
            val_accuracy = np.mean(classifier.predict(X_val) == y_val)

            results[(lr, reg)] = (train_accuracy, val_accuracy)

            if best_val < val_accuracy:
                best_val = val_accuracy
                best_softmax = classifier

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy,))

    w = best_softmax.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()
