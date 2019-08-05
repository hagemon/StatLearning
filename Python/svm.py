import math
import mat
import itertools
import random
from sklearn import datasets
import numpy as np  # deal dataset
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, c, kernel='linear'):
        self.c = c
        kernel_fn = {
            'linear': self.linear_kernel,
            'rbf': self.gaussian_kernel
        }
        self.kernel = kernel_fn[kernel]
        self.alpha = []
        self.b = 0.
        self.data, self.label = [], []
        self.eps = 1e-5

    @staticmethod
    def gaussian_kernel(x, z, sigma=1.):
        return math.exp(-sum(mat.sub_square(x, z))/(2*sigma**2))

    @staticmethod
    def linear_kernel(x, z):
        return mat.dot(x, z)

    def cal_gram(self, data):
        return [list(itertools.starmap(
            self.kernel, zip([data[i]]*len(data), data)
        )) for i in range(len(data))]

    def train(self, data, label):
        self.data, self.label = data, label
        self.alpha = [0. for _ in range(len(data))]
        self.b = 0.
        gram = self.cal_gram(data)
        self.smo(gram, label)
        print(self.alpha)

    def predict(self, x):
        def pred(xx, i):
            return self.alpha[i] * self.label[i] * self.kernel(xx, self.data[i])
        res = list(itertools.starmap(
            pred, zip(x, [k for k in range(len(self.data))])
        ))
        print(res)
        return [self.sign(r) for r in res]

    @staticmethod
    def sign(x):
        return 1 if x > 0 else -1

    def smo(self, gram, label):
        # select alpha heuristically
        def select_alpha():
            nonlocal label
            nonlocal update_flag
            nonlocal tol
            check_all = False
            while 1:
                update_count = 0
                if not check_all:
                    select_range = [k for k in range(len(label)) if 0 < self.alpha[k] < self.c]
                else:
                    select_range = range(len(label))
                for i in select_range:
                    a = self.alpha[i]
                    gi = sum([self.alpha[j] * label[j] * gram[i][j] for j in range(len(label))]) + self.b
                    r = label[i]*(gi-label[i])
                    # against kkt condition
                    if (r < -self.eps and a < self.c) or (r > self.eps and a > 0):
                        inner_range = [j for j in range(len(label)) if j != i]
                        random.shuffle(inner_range)
                        for j in inner_range:
                            gj = sum([self.alpha[k] * label[k] * gram[j][k] for k in range(len(label))]) + self.b
                            # heuristically select alpha j
                            yield i, j, gi-label[i], gj-label[j]
                            if update_flag:
                                update_count += 1
                                break  # finish search j for this i
                if update_count == 0:  # have not update alpha
                    if check_all:  # no alpha against kkt
                        break
                    else:  # finish searching alpha in boundary
                        check_all = True
                else:  # loop again
                    check_all = False
            yield None, None, None, None

        # SMO algorithm
        gen = select_alpha()
        tol = 1e-5
        update_flag = False

        for _ in range(10000):  # max iter 10000
            try:
                i, j, ei, ej = next(gen)
            except StopIteration as _:
                break
            if i is None:
                break
            update_flag = False
            eta = gram[i][i] + gram[j][j] - 2 * gram[i][j]
            ai, aj = self.alpha[i], self.alpha[j]
            if eta <= 0:
                continue
            delta = label[j]*(ei-ej)/eta
            if abs(delta) < tol:  # change too little
                continue
            aj_new = self.alpha[j] + delta
            if label[i] != label[j]:
                l_bound, h_bound = max(0, aj-ai), min(self.c, self.c+aj-ai)
            else:
                l_bound, h_bound = max(0, aj+ai-self.c), min(self.c, aj+ai)
            if aj_new > h_bound:
                aj_new = h_bound
            elif aj_new < l_bound:
                aj_new = l_bound
            ai_new = self.alpha[i] + label[j]*label[i]*(self.alpha[j]-aj_new)
            self.alpha[i], self.alpha[j] = ai_new, aj_new
            bi_new = -ei - label[i]*gram[i][i]*(ai_new-ai) - label[j]*gram[j][i]*(aj_new-aj) + self.b
            bj_new = -ej - label[j]*gram[i][j]*(ai_new-ai) - label[j]*gram[j][j]*(aj_new-aj) + self.b
            if 0 < ai < self.c:
                self.b = bi_new
            elif 0 < aj < self.c:
                self.b = bj_new
            else:
                self.b = (bi_new + bj_new) / 2
            update_flag = True


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    Y = (iris["target"] == 2).astype(np.float64)

    perm = [i for i in range(len(X))]
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]

    Y[Y == 0] = -1

    train_x = X[:100].tolist()
    train_y = Y[:100].tolist()
    test_x = X[100:].tolist()
    test_y = Y[100:].tolist()

    svm = SVM(1., 'linear')
    svm.train(train_x, train_y)
    result = svm.predict(test_x)

    print(train_x)
    print(train_y)

    print(test_x)
    print(test_y)

    prec = sum([1 if result[i] == test_y[i] else 0 for i in range(50)])/50

    print(prec)

    ww = [[svm.alpha[i]*svm.label[i]*x for x in svm.data[i]] for i in range(len(svm.data))]
    w = [0., 0.]
    for www in ww:
        w[0] += www[0]
        w[1] += www[1]
    print(w, svm.b)

    fn = lambda x: -(x*w[0]+svm.b)/w[1]
    plt.scatter(X[:100][Y[:100] == 1][:, 0], X[:100][Y[:100] == 1][:, 1], c='r')
    plt.scatter(X[:100][Y[:100] == -1][:, 1], X[:100][Y[:100] == -1][:, 1], c='g')
    plt.plot([0, 7], [fn(0), fn(7)])
    plt.show()
