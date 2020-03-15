# import pygraphviz
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class CausalDataGenerator:
    """

    Attributes
    ----------
    nodeCount : int
        the number of graph nodes
    sampleSize : int
        the size of sample data
    coeffMatrix : ndarray
        modified adjacency matrix
        if node j causes node i (j->i), Xi is from PAj, then in linear Xi = f(PAj, Ui) coefficent of Xj is coeffMatrix[i,j]
    X_withTreatment : ndarray
        rows of this matrix are variables, generated with linear equations, with generateX function and with treatment T = [11...1]
    X_without Treatment : ndarray
        matrix like X_withTreatment, but here we set T = [00...0]
    diTree : DiGraph
        generated directional tree (dag)
    orderedNodeList : ndarray
        topologically sorted nodes of diTree (cause we want to generate some Xi before its descendants)
    """

    def __init__(self, nodeCount = 5, sampleSize = 10):
        self.nodeCount = nodeCount
        self.sampleSize = sampleSize
        self.coeffMatrix = np.zeros((self.nodeCount, self.nodeCount))
        self.X_withTreatment = np.zeros((nodeCount, sampleSize))
        self.X_withoutTreatment = np.zeros((nodeCount, sampleSize))

    def makeDiTree(self):
        """
        Generates a directional tree
        """
        G = nx.random_tree(self.nodeCount)
        H = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
        self.diTree = H
        return H

    def drawGeneratedGraph(self):
        #Plots diTree

        nx.draw(self.diTree, with_labels = True)
        plt.show()

    # def drawGeneratedGraphWithGraphviz(self):
    #     A = nx.to_agraph(self.diTree)
    #     A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    #     A.draw('test.png')
    #     plt.show()

    def printEdges(self):
        #Prints edges of diTree

        for pair in list(self.diTree.edges):
            print(pair[0],pair[1])
    
    def generateCoeffMatrix(self):
        #Generates coeffMatrix
        for pair in list(self.diTree.edges):
            self.coeffMatrix[pair[1]][pair[0]] = random.randint(1,10)

    def orderNodesTopologically(self):
        #Sorts diTree nodes topologically
        self.orderedNodeList = list(nx.topological_sort(self.diTree))

    def isRootNode(self, node):
        #Checks if given node is one of the roots of diTree or not
        row = self.coeffMatrix[node]
        for coeff in row:
            if coeff != 0:
                return False

        return True
        
    def generateX(self):
        """
        Generates variables with creating linear equations using coeffMatrix. We mark one of the root nodes of diTree as 
        T(treatment) and one of the leaf nodes as Y(outcome). All other nodes we bring as observed data.
        """
        self.X_withTreatment[self.orderedNodeList[0]] = np.ones(self.sampleSize)
        self.X_withoutTreatment[self.orderedNodeList[0]] = np.zeros(self.sampleSize)

        for i in self.orderedNodeList[1:]:
            normalNoise = np.random.normal(size = self.sampleSize) 
            self.X_withTreatment[i] += normalNoise
            self.X_withoutTreatment[i] += normalNoise
            if not self.isRootNode(i):
                for j in range(len(self.coeffMatrix[i])):
                    if self.coeffMatrix[i][j]!=0:
                        self.X_withTreatment[i]+=self.coeffMatrix[i][j]*self.X_withTreatment[j]
                        self.X_withoutTreatment[i]+=self.coeffMatrix[i][j]*self.X_withoutTreatment[j]

    def makeDataFrames(self):
        """
        Creates DataFrame with treated(T = [11..1]) outcome    
        """
        df_withTreatment= self.makeDataFrame(self.X_withTreatment)

        """
        and this one, in whick outcome is not treated(T=[00...0])
        """
        df_withoutTreatment= self.makeDataFrame(self.X_withoutTreatment)
        return (df_withTreatment, df_withoutTreatment)
    
    def makeDataFrame(self, X):
        """
        Creates DataFrame from ndarray
        """
        clmns = []
        for i in range(X.shape[0]):
            if i == self.orderedNodeList[0]:
                clmns.append('T')
            elif i == self.orderedNodeList[-1]:
                clmns.append('Y')
            else:
                clmns.append('X' + str(i))
        
        df= pd.DataFrame(data=X.T, index=[str(i) for i in range(X.shape[1])], columns= clmns) # ['X' + str(i) for i in range(self.X_withTreatment.shape[0])]
        return df

    def generateDataFrame(self):
        """
        1. directional tree generation
        2. coefficient matrix for creating equations(equations are defined using that matrix)
        3. ordering nodes in topological order
        4. data generation with defined equations
        5. return of 2 dataframes(in the first dataframe outcome is treated, in the second is not)
        """
        self.makeDiTree() #1
        self.generateCoeffMatrix() #2
        self.orderNodesTopologically() #3
        self.generateX() #4
        return self.makeDataFrames() #5