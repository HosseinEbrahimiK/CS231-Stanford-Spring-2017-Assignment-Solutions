import numpy as np
import matplotlib.pyplot as plt
from load_data import load_CIFAR10
from random import randrange
import time


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
        score = np.matmul(X, self.W)
        y_pred = np.argmax(score, axis=1)
        return y_pred


class LinearSVM(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):

        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                dW[:, j] = dW[:, j] + X[i, :]
                dW[:, y[i]] = dW[:, y[i]] - X[i, :]
                loss += margin

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * np.vstack([W[:-1, :], np.zeros(num_classes, )])

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape)

    scores = np.matmul(X, W)
    score_yj = np.choose(y, scores.T)
    scores = (scores.T - score_yj).T
    scores += np.ones(scores.shape)
    np.maximum(scores, 0, scores)
    scores[np.arange(scores.shape[0]), y] = 0
    loss = (np.sum(scores, axis=None) / X.shape[0]) + np.sum(W*W, axis=None)

    one_mask = np.where(scores != 0, 1, 0)
    sum_row = one_mask.sum(axis=1)
    one_mask[np.arange(one_mask.shape[0]), y] -= sum_row
    dW = np.matmul(X.T, one_mask)

    dW = (dW / X.shape[0]) + (2 * reg * np.vstack([W[:-1, :], np.zeros(W.shape[1], )]))

    return loss, dW


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


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
    # print(mean_image[:10])  # print a few of the elements
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))  # visualize the mean image
    # plt.show()

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    W = np.random.randn(3073, 10) * 0.0001

    loss, grad0 = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss,))

    loss1, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad)

    loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
    grad_check_sparse(f, W, grad)

    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    print('difference: %f' % (loss_naive - loss_vectorized))

    tic = time.time()
    _, grad_naive0 = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Naive loss and gradient: computed in %fs' % (toc - tic))

    tic = time.time()
    _, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

    difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('difference: %f' % difference)

    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                          num_iters=1500, verbose=True)
    toc = time.time()
    print('That took %fs' % (toc - tic))

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [2.5e4, 5e4]

    results = {}
    best_val = -1
    best_svm = None

    for reg in regularization_strengths:
        for lr in learning_rates:

            classifier = LinearSVM()
            classifier.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=2000, verbose=True)
            train_accuracy = np.mean(classifier.predict(X_train) == y_train)
            val_accuracy = np.mean(classifier.predict(X_val) == y_val)

            results[(lr, reg)] = (train_accuracy, val_accuracy)

            if best_val < val_accuracy:
                best_val = val_accuracy
                best_svm = classifier

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    import math

    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

    w = best_svm.W[:-1, :]  # strip out the bias
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
