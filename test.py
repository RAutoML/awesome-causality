
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from CausalDataGenerator import CausalDataGenerator

GRAPH_SIZE = 10
SAMPLE_SIZE = 1000
generator = CausalDataGenerator(nodeCount= GRAPH_SIZE, sampleSize= SAMPLE_SIZE)
df = generator.generateDataFrame()

print("WITH TREATMENT")
print(df[0].head())

print("########################################################################")
print("########################################################################")

print("WITHOUT TREATMENT")
print(df[1].head())

generator.drawGeneratedGraph()