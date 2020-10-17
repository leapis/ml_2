#imports
import random
import math

import numpy as np

#params


#functions
def generate_data(m):
    k = 21
    xs = np.array([generate_vector() for _ in range(m)])
    y = np.empty(m)
    gen_weight = lambda i : math.pow(0.9, i + 1)
    for i in range(m):
        x = xs[i]
        y[i] = most_common(x[8:15]) if x[0] else most_common(x[1:8])
    return xs, y

def generate_vector():
    x = np.empty(21)
    x[0] = random.randint(0, 1)
    for i in range(1,15):
        if random.randint(1, 4) == 4:
            x[i] = 1 - x[i-1]
        else:
            x[i] = x[i-1]
    for i in range(15, 21):
        x[i] = random.randint(0, 1)
    return x

def gen_test(k):
    return np.random.randint(0, 2, k)

def dataset_entropy(labels):
    m = len(labels)
    if m == 0:
        return 0
    p0 = np.sum(labels) / m
    p1 = (m - np.sum(labels)) / m
    if p0 > 0: p0 *= math.log2(p0)
    if p1 > 0: p1 *= math.log2(p1)
    return - (p0 + p1)

def information_gain(dataset, labels):
    entropy = dataset_entropy(labels)
    k = len(dataset[0])
    info_gain = np.empty(k)
    for i in range(k):
        rows0 = [idx for idx, row in enumerate(dataset) if row[i] == 0]
        rows1 = [idx for idx, row in enumerate(dataset) if row[i] == 1]
        gain0 = len(rows0) / len(dataset) * dataset_entropy(labels[rows0])
        gain1 = len(rows1) / len(dataset) * dataset_entropy(labels[rows1])
        info_gain[i] = entropy - gain0 - gain1
    return info_gain

class Node:

    def __init__(self, index):
        self.index = index
        self.leaf = False
        self.zero = None
        self.one = None
        self.val = None #0 or 1

    def __str__(self):
        return f"Node: index:{self.index}, leaf:{self.leaf}, val:{self.val}\n"

def split_dataset(dataset, index_split):
    zero = [i for i, val in enumerate(dataset) if val[index_split] == 0]
    one  = [i for i, val in enumerate(dataset) if val[index_split] == 1]
    return zero, one

def most_common(labels):
    return int(sum(labels) >= len(labels) / 2)

def perfectMatch(dataset, index, val, labels):
    return False
    for i, row in enumerate(dataset):
        if row[index] == val and labels[i] != 1:
            return False
        elif row[index] != val and labels[i] == 1:
            return False
    return True

def perfectSplit(gain, dataset, labels):
    return False, -1
    k = len(dataset[0])
    for i in range(k):
        if perfectMatch(dataset, i, 0, labels) or perfectMatch(dataset, i, 1, labels):
            return True, i
    return False, -1

def generate_tree_depth(dataset, labels, depth, maxdepth):
    if depth > maxdepth:
        root = Node(None)
        root.leaf = True
        root.val = most_common(labels)
        return root
    if sum(labels) == len(labels) or sum(labels) == 0:
        root = Node(None)
        root.leaf = True
        root.val = labels[0]
        return root
    gain = information_gain(dataset, labels)
    perfect, split = perfectSplit(gain, dataset, labels)
    index_split = split if perfect else np.argmax(gain)
    root = Node(index_split)
    zero, one = split_dataset(dataset, index_split)
    if zero == []:
        root.leaf = True
        root.val = most_common(labels)
    elif one == []:
        root.leaf = True
        root.val = most_common(labels)
    else:
        root.zero = generate_tree_depth(dataset[zero], labels[zero], depth + 1, maxdepth)
        root.one = generate_tree_depth(dataset[one], labels[one], depth + 1, maxdepth)
    return root

def generate_tree(dataset, labels):
    if sum(labels) == len(labels) or sum(labels) == 0:
        root = Node(None)
        root.leaf = True
        root.val = labels[0]
        return root
    gain = information_gain(dataset, labels)
    perfect, split = perfectSplit(gain, dataset, labels)
    index_split = split if perfect else np.argmax(gain)
    root = Node(index_split)
    zero, one = split_dataset(dataset, index_split)
    if zero == []:
        root.leaf = True
        root.val = most_common(labels)
    elif one == []:
        root.leaf = True
        #print(dataset, labels)
        #exit(0)
        root.val = most_common(labels)
    else:
        root.zero = generate_tree(dataset[zero], labels[zero])
        root.one = generate_tree(dataset[one], labels[one])
    return root

def generate_tree_size(dataset, labels, size):
    if sum(labels) == len(labels) or sum(labels) == 0:
        root = Node(None)
        root.leaf = True
        root.val = labels[0]
        return root
    if len(labels) <= size:
        root = Node(None)
        root.leaf = True
        root.val = most_common(labels)
        return root
    gain = information_gain(dataset, labels)
    perfect, split = perfectSplit(gain, dataset, labels)
    index_split = split if perfect else np.argmax(gain)
    root = Node(index_split)
    zero, one = split_dataset(dataset, index_split)
    if zero == []:
        root.leaf = True
        root.val = most_common(labels)
    elif one == []:
        root.leaf = True
        #print(dataset, labels)
        #exit(0)
        root.val = most_common(labels)
    else:
        root.zero = generate_tree_size(dataset[zero], labels[zero], size)
        root.one = generate_tree_size(dataset[one], labels[one], size)
    return root

def index_order_generation(dataset, labels, ordering):
    if sum(labels) == len(labels) or sum(labels) == 0:
        root = Node(None)
        root.leaf = True
        root.val = labels[0]
        return root
    if ordering == []:
        root = Node(None)
        root.leaf = True
        root.val = labels[np.bincount(labels).argmax()]
    index_split = ordering[0]
    root = Node(index_split)
    zero, one = split_dataset(dataset, index_split)
    if zero == [] or one == []:
        return index_order_generation(dataset, labels, ordering[1:])
    else:
        root.zero = index_order_generation(dataset[zero], labels[zero], ordering[1:])
        root.one = index_order_generation(dataset[one], labels[one], ordering[1:])
    return root


def printTree(root, high = float("inf")):
    count = 0
    queue = [("root", root)]
    while queue:
        count += 1
        if count > high:
            return
        tag, node = queue.pop(0)
        print(tag, node)
        if not node.leaf:
            queue.append(("zero", node.zero))
            queue.append(("one", node.one))

def countTree(root):
    if root.leaf:
        return 1
    else:
        return countTree(root.zero) + countTree(root.one)

def count_irr(root):
    seen = {}
    def dfs(node):
        if node.leaf:
            return
        if node.index > 14:
            seen[node.index] = True
        dfs(node.zero)
        dfs(node.one)
    dfs(root)
    return len(seen)

def run_tree(root, vector):
    if root.leaf:
        return root.val
    index = root.index
    if vector[index] == 1:
        return run_tree(root.one, vector)
    else: #if vector[index] == 0
        return run_tree(root.zero, vector)

def tree_error(tree, dataset, labels):
    errors = sum( [ run_tree(tree, vector) == labels[i] for i, vector in enumerate(dataset)] )
    return errors / len(dataset)

def distribution_error(tree, m, k=21, RUNS=1000):
    error = 0
    for _ in range(RUNS):
        x, y = generate_data(m)
        error += tree_error(tree, x, y)
    return error/RUNS

def generate_document_data(mRange, RUNS = 1):
    data = []
    for m in mRange:
        m = int(m)
        data.append(0)
        for _ in range(RUNS):
            x, y = generate_data(m)
            tree = generate_tree(x, y)
            data[-1] += 1 - distribution_error(tree, 10 * m, RUNS=100)
        data[-1] /= RUNS
        print(data[-1])
    return data
        
def generate_document_data_2(mRange, RUNS = 10):
    data = []
    for m in mRange:
        m = int(m)
        data.append(0)
        for _ in range(RUNS):
            x, y = generate_data(m)
            tree = generate_tree(x, y)
            data[-1] += count_irr(tree)
        data[-1] /= RUNS
        print(data[-1])
    return data

def generate_document_data_d(dRange, RUNS = 1):
    distribution = []
    training = []

    x, y = generate_data(8000)
    testx, testy = generate_data(2000)

    for d in dRange:
        tree = generate_tree_depth(x, y, 0, d)
        training.append(1 - tree_error(tree, x, y))
        distribution.append(1 - tree_error(tree, testx, testy))
        #print("t", training[-1])
        print(d, distribution[-1])
    return (training, distribution)

def generate_document_data_s(sRange, RUNS = 1):
    distribution = []
    training = []

    x, y = generate_data(8000)
    testx, testy = generate_data(2000)

    for s in sRange:
        tree = generate_tree_size(x, y, s)
        training.append(1 - tree_error(tree, x, y))
        distribution.append(1 - tree_error(tree, testx, testy))
        #print("t", training[-1])
        print(int(1000 * training[-1])/1000)
    return (training, distribution)


def generate_document_data_2d(mRange,d, RUNS = 10):
    data = []
    for m in mRange:
        m = int(m)
        data.append(0)
        for _ in range(RUNS):
            x, y = generate_data(m)
            tree = generate_tree_depth(x, y, 0, d)
            data[-1] += count_irr(tree)
        data[-1] /= RUNS
        print(data[-1])
    return data

def generate_document_data_2s(mRange,s, RUNS = 10):
    data = []
    for m in mRange:
        m = int(m)
        data.append(0)
        for _ in range(RUNS):
            x, y = generate_data(m)
            tree = generate_tree_size(x, y, s)
            data[-1] += count_irr(tree)
        data[-1] /= RUNS
        print(data[-1])
    return data

def generate_document_data_2d_err(mRange,d, RUNS = 10):
    data = []
    for m in mRange:
        m = int(m)
        data.append(0)
        for _ in range(RUNS):
            x, y = generate_data(m)
            tree = generate_tree_depth(x, y, 0, d)
            data[-1] += distribution_error(tree, m, RUNS=10) #count_irr(tree)
        data[-1] /= RUNS
        print(1 - data[-1])
    return data

if __name__ == "__main__":
    #generate_document_data_s(range(0, 50))
    for i in (range(1000, 20001, 1000)): print(i)
    """
    x, y = generate_document_data_2d((550, 1), 8, RUNS = 100)
    print("training")
    for i in x: print(i)
    print("distrib")
    for i in y: print(i)
        d = int(d)
    """
    #m = 1500
   # k = 21
    #x, y = generate_data(m)
    #tree = generate_tree_depth(x, y, 0, 5)
    #printTree(tree)
    #print(countTree(tree))
    #print(distribution_error(tree, m, k, 100))
"""
    total = [0]*4
    for _ in range(1):
        x, y = (generate_data(4, 10))
        #print(dataset_entropy(y))
        #print(x,y)
        info = information_gain(x, y)
        total = [t + info[i] for i,t in enumerate(total)]
    print(total)

    x, y = (generate_data(4, 30))
    print(f"L:{sum(y)/len(y)}")
    print("datagen")
    root = generate_tree(x, y)
    #print(x, y)
    print(information_gain(x, y))
    printTree(root)
    print(f"c: {countTree(root)}")
    print(f"e: {tree_error(root, x, y)}")
"""






# theory- you have a group of "effective vectors", that have the most power in determining the result. 
# the middle most "effective vector" is the most powerful at determining the overall likelihood of the set of effective vectors. 
# therefore, it is the most impactful to watch.

# howeveer, the fulltime one cares about both tilt and x1. 
# it needs to know 2 things, hence it can't determine what's best
# there're all bad at the start: 
# if x_4 is 1, then x_1 is likely 1, then y is likely 1
# if x_4 is 0, then x_1 is likely 0, then y is likely 1

# therefore, the idea determiner is either {5,6,7} then 1, or 1 then {5,6,7}- that way we know well both the sway and the determiner.