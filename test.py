import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from CausalDataGenerator import CausalDataGenerator

RESOURCE_PATH = 'C:\\Users\\Arman\\Desktop'
GRAPH_SIZE = 10
SAMPLE_SIZE = 10000

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

def relu(X):
    return np.maximum(0,X)

generator = CausalDataGenerator(resourcePath = RESOURCE_PATH, nodeCount= GRAPH_SIZE, sampleSize= SAMPLE_SIZE)

#1 df = generator.generateDataFrame(same, sigmoid) 
#2 df = generator.generateDataFrame(np.tanh, sigmoid)
#3 df = generator.generateDataFrame(sigmoid, sigmoid)
#4 df = generator.generateDataFrame(np.square, same)
#5 df = generator.generateDataFrame(sigmoid, np.square)
# df = generator.generateDataFrame(same, sigmoid)

# df = generator.generateDataFrame(np.tanh, sigmoid)
# df = generator.generateDataFrame(np.square, sigmoid)


normalNoise_0_1 = np.random.normal(size = SAMPLE_SIZE) # ~ N(0,1)
normalNoise_2_5 = np.random.normal(2, 5, size = SAMPLE_SIZE) # ~ N(2,5)
beta_1_3 = np.random.beta(1, 3, size = SAMPLE_SIZE) 
uniform_4_10 = np.random.uniform(4, 10, size = SAMPLE_SIZE)

noiseArray = [normalNoise_0_1, normalNoise_0_1, normalNoise_2_5, beta_1_3, uniform_4_10]
mfArray = [np.tanh, relu, same, same, same]
cMArray = [np.tanh, relu, relu, same, same]
df = generator.generateDataFrame(4, mfArray , cMArray, noiseArray, ['#3355FF', '#E933FF', '#FF9F33', '#FFF933', '#33FFE3'])

print('--------------------------------------------------------------------------------')

print('DataFrame')
print(df)
# print([i - j for i, j in zip(df['Y1'], df['Y0'])])

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
print('std(Y1)')
print(df['Y1'].std())
print('--------------------------------------------------------------------------------')
print('\n')
print('E[Y0]')
print(df['Y0'].mean())
print('std(Y0)')
print(df['Y0'].std())
