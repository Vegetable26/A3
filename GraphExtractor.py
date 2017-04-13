from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import numpy as np
import sys
import os
import io

# Matrix[i, j] means there is an edge from i to j

def BuildMatrix(numNodes, edgeFile, directed=True):
    matrix = csr_matrix((numNodes, numNodes), dtype=int)
    # The ego is connected to every node
    matrix[0, :] = 1
    # Everyone is connected to the ego if the graph is undirected
    if not directed:
        matrix[:, 0] = 1
    for line in edgeFile:
        edge = line.split()
        matrix[int(edge[0]), int(edge[1])] = 1
        if not directed:
            matrix[int(edge[1]), int(edge[0])] = 1
    return matrix

def NormalizeRows(matrix):
    return normalize(matrix, axis=1, norm='l1')

def CalculateDistribution(matrix):
    distribution = defaultdict(int)
    edgeCounts = matrix.sum(axis=1) # sum each row
    for count in edgeCounts:
        distribution[count[0, 0]] += 1
    return distribution

def CalculatePageRank(matrix, d):
    n = NormalizeRows(matrix)
    n = np.multiply(n, 1-d)
    normalizedDamping = d / matrix.shape[0]
    # Page rank uses the transpose
    final = np.add(n.todense(), normalizedDamping).transpose()
    w, et = np.linalg.eig(final)
    e = np.transpose(et.real)
    w1, e1 = reduce((lambda x, y : x if (x[0] > y[0]) else y), zip(w, e))
    return np.multiply(np.absolute(normalize(e1, axis=1, norm='l1')), matrix.shape[0])


cwd = os.getcwd()
data_dir = cwd + "/data/" + sys.argv[1]
data_files = os.listdir(data_dir)
data_files = map(lambda x : data_dir + "/" + x, data_files)
print(data_files)
for i in range(len(data_files) / 5):
    edges = io.open((data_files[i * 5 + 1]))
    egofeat = open(data_files[i * 5 + 2])
    features = open(data_files[i * 5 + 3])
    featname = open(data_files[i * 5 + 4])

    # Account for the ego
    numNodes = sum(1 for line in features) + 1
    # Skip graphs that have less than 10 nodes
    if (numNodes < 10):
        continue
    matrix = BuildMatrix(numNodes, edges, directed=False)
    """
    matrix = csr_matrix((4, 4), dtype=int)
    matrix[0, 1] = 1
    matrix[1, 2] = 1
    matrix[0, 2] = 1
    matrix[2, 0] = 1
    matrix[3, 2] = 1
    """
    distribution = CalculateDistribution(matrix)
    pageRank = CalculatePageRank(matrix, 0.15)
    print(pageRank)
    break