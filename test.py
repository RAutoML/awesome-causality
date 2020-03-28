
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

def someMiddleFunc1(X):
    return X * X / 5

def someMiddleFunc(X):
    return X * np.sqrt(np.absolute(X))

def linlog(X):
    return X * np.log(np.absolute(X))

def f(X):
    return np.square(X) * X

generator = CausalDataGenerator(resourcePath = RESOURCE_PATH, nodeCount= GRAPH_SIZE, sampleSize= SAMPLE_SIZE)

#1 df = generator.generateDataFrame(same, sigmoid) 
#2 df = generator.generateDataFrame(np.tanh, sigmoid)
#3 df = generator.generateDataFrame(sigmoid, sigmoid)
#4 df = generator.generateDataFrame(np.square, same)
#5 df = generator.generateDataFrame(sigmoid, np.square)
# df = generator.generateDataFrame(same, sigmoid)


df = generator.generateDataFrame(np.square, sigmoid)
print('--------------------------------------------------------------------------------')

print('DataFrame')
print(df)
# print([i - j for i, j in zip(df['Y1'], df['Y0']) ])

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

