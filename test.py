
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

<<<<<<< HEAD
#1 df = generator.generateDataFrame(same, sigmoid)
#2 df = generator.generateDataFrame(np.tanh, sigmoid)
#3 df = generator.generateDataFrame(sigmoid, sigmoid)

df = generator.generateDataFrame(sigmoid, sigmoid)
print('--------------------------------------------------------------------------------')
print('DataFrame')
print(df)
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('\n')
print('ATE')
ate = (df['Y1'] - df['Y0']).mean()
print(ate)
print('--------------------------------------------------------------------------------')
print('\n')
print('E[Y1]')
print(df['Y1'].mean())
print('--------------------------------------------------------------------------------')
print('\n')
print('E[Y0]')
print(df['Y0'].mean())
=======
# df = generator.generateDataFrame(sigmoid, same)
# df = generator.generateDataFrame(np.square, same)
df = generator.generateDataFrame(np.tanh, same)
# t = generator.generateDataFrame()
# g = generator.generateTanhDataFrame()

print("X")
print(df.head())
# print("########################################################################")
# print("########################################################################")
>>>>>>> 7831e958ad4f31330fdfb45b3cd4db53481d90ed
