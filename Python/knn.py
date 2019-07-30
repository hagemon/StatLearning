import heapq
import math


class Node:
    def __init__(self, data, dim, left, right, is_leaf=False):
        self.data = data
        self.dim = dim
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

    def __lt__(self, other):
        return self.data[0] < other.data[0]


class Tree:
    def __init__(self, data):
        self.data = data
        self.n_ft = len(data[0])
        self.root = self._build_tree(self.data, 0)

    def _build_tree(self, data, dim):
        # select next dim
        # also can be done by find dim with highest variance
        next_dim = (dim + 1) % self.n_ft

        if len(data) == 0:
            return None
        elif len(data) == 1:
            return Node(data[0], next_dim, None, None, is_leaf=True)

        left, mid, right = self._div(data, dim)
        node = Node(
            mid, dim,
            self._build_tree(left, next_dim),
            self._build_tree(right, next_dim)
        )
        return node

    def prev(self, node):
        if not node:
            return
        print(node.data)
        self.prev(node.left)
        self.prev(node.right)

    def search_knn(self, target, k):
        """
        find k nearest neighbourhood data to target

        :param target: target data
        :param k: number of nearest data to be search
        :return: searched data
        """
        heap = []  # min heap
        min_dist = math.inf
        if self.root is None or len(self.data) == 0:
            return heap

        def search(node):
            if not node:
                return
            nonlocal heap
            nonlocal min_dist
            if node.is_leaf:
                dist = self.dist(target, node.data)
                if dist < min_dist:
                    if len(heap) < k:
                        heapq.heappush(heap, (-dist, node))
                        if len(heap) == k:
                            min_dist = -heap[0][0]
                    else:
                        heapq.heappushpop(heap, (-dist, node))
                        min_dist = -heap[0][0]
            else:
                if target[node.dim] <= node.data[node.dim]:
                    search(node.left)
                    if node.data[node.dim] - target[node.dim] <= min_dist:
                        search(node.right)
                else:
                    search(node.right)
                    if target[node.dim] - node.data[node.dim] <= min_dist:
                        search(node.left)
                dist = self.dist(target, node.data)
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, node))
                    if len(heap) == k:
                        min_dist = -heap[0][0]
                elif dist < min_dist:
                    heapq.heappushpop(heap, (-dist, node))
                    min_dist = -heap[0][0]

        search(self.root)
        heap.sort(key=(lambda x: -x[0]))
        res = [node.data for _, node in heap]
        return res

    def dist(self, a, b):
        return sum([(a[i]-b[i])**2 for i in range(self.n_ft)])

    @staticmethod
    def _div(data, dim):
        """
        find median and division using partition method in quick sort
        :param data: data of current node
        :param dim: select dim to calculate median
        :return: left data, median and right data
        """
        def find_k(d, begin, end, k):
            guard = d[begin][dim]
            left, right = begin, end
            while left <= right:
                while left <= right and d[left][dim] <= guard:
                    left += 1
                while left <= right and d[right][dim] >= guard:
                    right -= 1
                if left < right:
                    d[left], d[right] = d[right], d[left]
            d[right], d[begin] = d[begin], d[right]

            if left - begin == k:
                return
            elif left - begin < k:
                find_k(d, left, end, k - (left - begin))
            else:
                find_k(d, begin, right - 1, k)

        mid = len(data) // 2 + 1
        find_k(data, 0, len(data)-1, mid)
        return data[:mid-1], data[mid-1], data[mid:]


if __name__ == '__main__':
    # train_data, train_label, test_data, test_label = utils.get_data()
    train_data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    tree = Tree(train_data)
    result = tree.search_knn([2, 3], 3)
    print(result)

