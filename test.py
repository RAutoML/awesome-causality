
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from CausalDataGenerator import CausalDataGenerator

RESOURCE_PATH = '/Users/narekghukasyan/Desktop'
GRAPH_SIZE = 10
SAMPLE_SIZE = 1000

def sigmoid(row):
    return 1/(1 + np.exp(-row))

def same(X):
    return X

generator = CausalDataGenerator(resourcePath = RESOURCE_PATH, nodeCount= GRAPH_SIZE, sampleSize= SAMPLE_SIZE)

# df = generator.generateDataFrame(sigmoid, same)
# df = generator.generateDataFrame(np.square, same)
df = generator.generateDataFrame(np.tanh, same)
# t = generator.generateDataFrame()
# g = generator.generateTanhDataFrame()

print("X")
print(df.head())
# print("########################################################################")
# print("########################################################################")