import operator
import itertools


def sub(a, b):
    assert len(a) == len(b)
    return list(itertools.starmap(operator.sub, zip(a, b)))


def sub_square(a, b):
    assert len(a) == len(b)
    return itertools.starmap(operator.pow, zip(sub(a, b), [2]*len(a)))


def mul(a, b):
    assert len(a) == len(b)
    return list(itertools.starmap(operator.mul, zip(a, b)))


def dot(a, b):
    assert len(a) == len(b)
    return sum(mul(a, b))


if __name__ == '__main__':
    x = [1, 2, 3]
    z = [4, 5, 6]
    print(list(sub_square(x, z)))
