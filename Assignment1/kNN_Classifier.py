import numpy as np
import matplotlib.pyplot as plt
from load_data import load_CIFAR10


class KNearestNeighbor:

    def __init__(self):
        self.X_tr = np.zeros(0)
        self.y_tr = np.zeros(0)

    def train(self, X, y):
        self.X_tr = X
        self.y_tr = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_tr = self.X_tr.shape[0]
        num_te = X.shape[0]

        dists = np.zeros((num_te, num_tr))

        for i in range(num_te):
            for j in range(num_tr):
                dists[i][j] = np.linalg.norm(X[i] - self.X_tr[j])

        return dists

    def compute_distances_one_loop(self, X):

        num_tr = self.X_tr.shape[0]
        num_te = X.shape[0]

        dists = np.zeros((num_te, num_tr))

        for i in range(num_te):
            dists[i, :] = np.linalg.norm(X[i] - self.X_tr, axis=1)

        return dists

    def compute_distances_no_loops(self, X):

        num_tr = self.X_tr.shape[0]
        num_te = X.shape[0]

        dists = np.zeros((num_te, num_tr))
        N_train = np.tile(np.linalg.norm(self.X_tr, axis=1)**2, (num_te, 1))
        N_test = np.tile(np.linalg.norm(X, axis=1)**2, (num_tr, 1)).T
        dists = np.sqrt(N_train + N_test - 2 * np.matmul(X, self.X_tr.T))

        return dists

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]

        temp = np.reshape(self.y_tr[np.argsort(dists, axis=1)[:, :k].flatten()], (num_test, k))

        u, indices = np.unique(temp, return_inverse=True)
        axis = 1
        y_pred = [np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(temp.shape),
                                        None, np.max(indices) + 1), axis=axis)]

        return y_pred


def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def cross_validation(X, y, k_choices, num_folds=1):

    X_train_folds = np.array_split(X, num_folds)
    y_train_folds = np.array_split(y, num_folds)

    k_to_accuracies = {}

    classifier = KNearestNeighbor()

    for i in range(len(k_choices)):
        print(i)
        accuracies = list()
        for j in range(num_folds):

            Xval_fold = X_train_folds[j]
            yval_fold = y_train_folds[j]

            Xtr_fold = np.array(X_train_folds[:j] + X_train_folds[j+1:])
            Xtr_fold = np.reshape(Xtr_fold, (Xtr_fold.shape[0]*Xtr_fold.shape[1], Xtr_fold.shape[2]))

            ytr_fold = np.array(y_train_folds[:j] + y_train_folds[j+1:])
            ytr_fold = np.reshape(ytr_fold, (ytr_fold.shape[0]*ytr_fold.shape[1]))

            classifier.train(Xtr_fold, ytr_fold)
            dist = classifier.compute_distances_no_loops(Xval_fold)

            pred_y = classifier.predict_labels(dist, k=k_choices[i])
            accuracies.append(np.sum(pred_y == yval_fold) / len(yval_fold))

        k_to_accuracies[k_choices[i]] = accuracies

    return k_to_accuracies


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_CIFAR10("cifar-10-batches-py")

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7

    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_two_loops(X_test)
    print(dists.shape)

    plt.imshow(dists, interpolation='none')
    plt.show()

    y_test_pred = classifier.predict_labels(dists, k=1)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct / num_test)
    print(accuracy)
    dists = classifier.compute_distances_no_loops(X_test)
    y_test_pred = classifier.predict_labels(dists, k=5)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct / num_test)
    print(accuracy)

    dists_one = classifier.compute_distances_one_loop(X_test)
    difference = np.linalg.norm(dists - dists_one, ord='fro')

    dists_two = classifier.compute_distances_no_loops(X_test)
    difference = np.linalg.norm(dists - dists_two, ord='fro')
    print('Difference was: %f' % (difference,))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')

    two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
    print('Two loop version took %f seconds' % two_loop_time)

    one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
    print('One loop version took %f seconds' % one_loop_time)

    no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
    print('No loop version took %f seconds' % no_loop_time)

    kChoices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_accuracies = cross_validation(X_train, y_train, kChoices, num_folds=5)
    for k in sorted(k_accuracies):
        for accuracy in k_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in kChoices:
        accuracies = k_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    plt.xlabel("k")
    plt.ylabel("accuracies for Cross-Validation")
    plt.show()

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_accuracies.items())])
    plt.errorbar(kChoices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    print("*******************************")

    best_k = 10
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
