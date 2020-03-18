
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from CausalDataGenerator import CausalDataGenerator

RESOURCE_PATH = '/Users/narekghukasyan/Desktop'
GRAPH_SIZE = 10
SAMPLE_SIZE = 1000

def sigmoid(self, row): # [sigmoid(a1), sigmoid(a2), ... ]  <- X[i] = [a1, a2, .. ]
    return 1/(1 + np.exp(-row))

def same(self, X):
    return X

generator = CausalDataGenerator(resourcePath = RESOURCE_PATH, nodeCount= GRAPH_SIZE, sampleSize= SAMPLE_SIZE)

df = generator.generateDataFrame(np.square, generator.sigmoid)
# t = generator.generateDataFrame()
# g = generator.generateTanhDataFrame()

print("X")
print(df.head())
# print("########################################################################")
# print("########################################################################")