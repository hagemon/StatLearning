import math
import itertools
from utils import get_iris
from abc import ABCMeta, abstractmethod


class Node:
    def __init__(self, val, dim, left=None, loss=0.0, sub_tree_loss=0.0, right=None, leaf=False):
        self.val = val
        self.dim = dim
        self.left = left
        self.right = right
        self.loss = loss
        self.leaf = leaf
        self.sub_tree_loss = sub_tree_loss
        self.comp = 1
        self.pruned = False

    def set_sub_loss(self, loss):
        self.sub_tree_loss = loss

    def set_complex(self, comp):
        self.comp = comp

    def prune(self):
        self.pruned = True


class Tree(metaclass=ABCMeta):
    def __init__(self, n_min=10, cls=True):
        self.root = None
        self.n_min = n_min
        self.fold = 10
        self.cls = cls
        self.alpha = math.inf
        self.pruned_nodes = [None]

    def fit(self, data, label):
        self.root = self.build_tree(data, label)
        # n_data = len(data)
        # roots = []
        # for i in range(self.fold):  # k-fold evaluation
        #     eval_start = i*n_data//self.fold
        #     eval_end = eval_start + n_data//self.fold
        #     train_data = [data[j] for j in range(len(data)) if eval_start <= j <= eval_end]
        #     train_label = [label[j] for j in range(len(data)) if eval_start <= j <= eval_end]
        #     root = self.build_tree(train_data, train_label)
        #     roots.append(root)
        # self.root = roots[roots.index(min(roots))]

    def predict(self, data):
        def pred(d, node):
            if node.leaf:
                return node.val
            if d[node.dim] < node.val:
                return pred(d, node.left)
            return pred(d, node.right)

        if not data or not self.root:
            return None

        res = list(itertools.starmap(pred, zip(data, [self.root]*len(data))))
        return res

    # def prune(self, node):
    #     if node.left and node.right and not node.pruned:
    #         self.prune(node.left)
    #         self.prune(node.right)
    #         sub_tree_loss = node.left.sub_tree_loss+node.right.sub_tree_loss
    #         comp = node.left.comp + node.right.comp + 1
    #         gt = (sub_tree_loss-node.loss)/(comp-1)
    #         if gt < self.alpha:
    #             self.alpha = gt
    #             self.pruned_nodes[-1] = node
    #         node.set_sub_loss(sub_tree_loss)

    @abstractmethod
    def build_tree(self, data, label):
        pass

    @abstractmethod
    def build_leaf(self, label):
        pass

    @abstractmethod
    def select_attr(self, data, label):
        pass


class ClaTree(Tree):
    def build_leaf(self, label):
        val = max([(c, label.count(c)) for c in set(label)], key=lambda x: x[1])[0]  # vote for cls
        loss = 1-sum([1 if label[i] == val else 0 for i in range(len(label))])/len(label)
        return Node(val=val, dim=None, loss=loss, sub_tree_loss=loss, leaf=True)

    def build_tree(self, data, label):
        if all([label[i] == label[0] for i in range(len(label))]):
            return Node(val=label[0], dim=None, leaf=True)

        if len(data) <= self.n_min:
            return self.build_leaf(label)

        attr, loss, seg = self.select_attr(data, label)

        left = [(data[i], label[i]) for i in range(len(data)) if data[i][attr] == seg]
        right = [(data[i], label[i]) for i in range(len(data)) if data[i][attr] != seg]

        if len(left) == 0 or len(right) == 0:  # all elements equal
            return self.build_leaf(label)

        left_data, left_label = zip(*left)
        right_data, right_label = zip(*right)

        return Node(
            val=seg,
            dim=attr,
            left=self.build_tree(left_data, left_label),
            right=self.build_tree(right_data, right_label),
            loss=loss
        )

    def select_attr(self, data, label):
        n_attr = len(data[0])
        a_error = [math.inf for _ in range(n_attr)]  # error for each attr
        a_min_s = [0.0 for _ in range(n_attr)]  # min s for each attr

        for i in range(n_attr):
            vals = sorted(set([data[j][i] for j in range(len(data))]))
            if len(vals) == 1:
                continue
            a_gini = 0
            min_s = 0
            min_gini = math.inf
            for k, s in enumerate(vals):
                s_gini = 1
                num_s = sum([1 if data[j][i] == s else 0 for j in range(len(data))])
                label_s = [label[j] for j in range(len(data)) if data[j][i] == s]
                for l in set(label_s):
                    s_gini -= math.pow(label_s.count(l)/num_s, 2)
                if s_gini < min_gini:
                    min_gini = s_gini
                    min_s = s
                a_gini += float(num_s)/len(data)*s_gini
            a_error[i] = a_gini
            a_min_s[i] = min_s

        loss = min(a_error)  # min loss among all attributes
        ind = a_error.index(loss)  # attribute with min loss
        seg = a_min_s[ind]  # s with min loss
        return ind, loss, seg


class RegTree(Tree):
    def __init__(self, n_min=10, cls=False):
        super(RegTree, self).__init__(n_min, cls)
    
    def build_leaf(self, label):
        if not self.cls:
            val = sum(label) / len(label)
            loss = sum([math.pow(label[i]-val, 2) for i in range(len(label))])
            return Node(val=val, dim=None, loss=loss, sub_tree_loss=loss, leaf=True)
        else:
            val = max([(c, label.count(c)) for c in set(label)], key=lambda x: x[1])[0]  # vote for cls
            loss = 1 - sum([1 if label[i] == val else 0 for i in range(len(label))]) / len(label)
            return Node(val=val, dim=None, loss=loss, leaf=True)

    def build_tree(self, data, label):
        if len(data) <= self.n_min:
            return self.build_leaf(label)

        attr, loss, seg = self.select_attr(data, label)

        left = [(data[i], label[i]) for i in range(len(data)) if data[i][attr] <= seg]
        right = [(data[i], label[i]) for i in range(len(data)) if data[i][attr] > seg]

        if len(left) == 0 or len(right) == 0:  # all elements equal
            return self.build_leaf(label)

        left_data, left_label = zip(*left)
        right_data, right_label = zip(*right)

        return Node(
            val=seg,
            dim=attr,
            left=self.build_tree(left_data, left_label),
            right=self.build_tree(right_data, right_label),
            loss=loss
        )

    def select_attr(self, data, label):
        n_attr = len(data[0])
        a_error = [math.inf for _ in range(n_attr)]  # error for each attr
        a_min_s = [0.0 for _ in range(n_attr)]  # min s for each attr

        for i in range(n_attr):
            vals = sorted(set([data[j][i] for j in range(len(data))]))
            if len(vals) == 1:
                continue
            min_s = 0
            min_e = math.inf
            vals = [(vals[j] + vals[j + 1]) / 2 for j in range(len(vals) - 1)]
            s_error = [math.inf for _ in range(len(vals))]  # error for each s
            for k, s in enumerate(vals):
                r1 = [label[j] for j in range(len(label)) if data[j][i] < s]
                r2 = [label[j] for j in range(len(label)) if data[j][i] >= s]
                c1 = sum(r1) / len(r1)
                c2 = sum(r2) / len(r2)
                e = sum([math.pow(label[j] - c1, 2) for j in range(len(label)) if data[j][i] < s])
                e += sum([math.pow(label[j] - c2, 2) for j in range(len(label)) if data[j][i] >= s])
                if e < min_e:
                    min_e = e
                    min_s = s
                s_error[k] = e
            a_error[i] = min(s_error)
            a_min_s[i] = min_s

        loss = min(a_error)  # min loss among all attributes
        ind = a_error.index(loss)  # attribute with min loss
        seg = a_min_s[ind]  # s with min loss
        return ind, loss, seg


if __name__ == '__main__':
    for _ in range(10):
        train_x, train_y, test_x, test_y = get_iris()
        cart = RegTree(n_min=5, cls=True)
        cart.fit(train_x, train_y)
        result = cart.predict(test_x)
        prec = sum([result[i] == test_y[i] for i in range(len(test_y))])/len(test_y)
        print(prec)
