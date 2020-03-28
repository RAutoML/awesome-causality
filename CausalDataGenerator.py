# import pygraphviz
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import uuid
import math

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
        rows of this matrix are variables, generated with linear equations, with generateCausalX function and with treatment T = [11...1]
    X_without Treatment : ndarray
        matrix like X_withTreatment, but here we set T = [00...0]
    diTree : DiGraph
        generated directional tree (dag)
    orderedNodeList : ndarray
        topologically sorted nodes of diTree (cause we want to generate some Xi before its descendants)
    """

    def __init__(self, resourcePath, nodeCount = 5, sampleSize = 10):
        self.resourcePath = resourcePath
        self.nodeCount = nodeCount
        self.sampleSize = sampleSize
        self.coeffMatrix = np.zeros((self.nodeCount, self.nodeCount))
        self.X = np.zeros((nodeCount, sampleSize))
        self.X_withTreatment = np.zeros((nodeCount, sampleSize))
        self.X_withoutTreatment = np.zeros((nodeCount, sampleSize))

    def makeDiTree(self):
        """
        Generates a directional tree
        """
        G = nx.random_tree(self.nodeCount) # dag X1->..->X1
        H = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])

        self.diTree = H
        return H

    def drawGeneratedGraph(self, path):
        #Plots diTree
        node_colors = []
        for node in self.diTree.nodes():
            if node == 'Y':
                node_colors.append('green')
            elif node == 'T':
                node_colors.append('red')
            else:
                node_colors.append('blue')
        
        # pos = graphviz_layout(self.diTree, prog='dot')
        nx.draw(self.diTree, with_labels = True, node_color = node_colors)
        
        plt.show(block = False)
        plt.savefig(path, format='PNG')

    def printEdges(self):
        #Prints edges of diTree
        for pair in list(self.diTree.edges):
            print(pair[0],pair[1])
    
    def generateCoeffMatrix(self):
        #Generates coeffMatrix
        for pair in list(self.diTree.edges):
            self.coeffMatrix[pair[1]][pair[0]] = random.normalvariate(0, 5) #random.uniform(-1.0, 1.0)

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

    def addTreatmentNode(self):
        mapping = {self.orderedNodeList[-1] : 'Y'}
        nx.relabel_nodes(self.diTree, mapping, copy= False)
        treatmentEdge = ('T', 'Y')
        self.diTree.add_edge(*treatmentEdge)
    
    def colorGraph(self):
        self.diTree.nodes['T']['color'] = '#ff0000'
        self.diTree.nodes['Y']['color'] = '#00ff00'

    # 1. generate X with nodes from orderednodelist range(len(orderednodelist) - 1)
    # 2. set T = [00...0] and generate Y0
    # 3. set T = [11..1] and generate Y1
    # 4. make dataframe (X, Y0, Y1)
    # NOTE: Y1  ~  Y, when T = [11...1]

    def generateX(self, middleFunction, causalMechanism):
        size = len(self.orderedNodeList)
        for i in range(size - 1):
            normalNoise = np.random.normal(size = self.sampleSize) # ~ N(0,1)
            pos = self.orderedNodeList[i]
            self.X[pos] += normalNoise
            if not self.isRootNode(i):
                for j in range(len(self.coeffMatrix[pos])):
                    if self.coeffMatrix[i][j]!=0:
                        self.X[pos] += self.coeffMatrix[pos][j]* middleFunction(self.X[j])
            self.X[pos] = causalMechanism(self.X[pos])
        self.outcomePos = self.orderedNodeList[-1]
        YCoeffs = self.coeffMatrix[self.outcomePos]
        noise = np.random.normal(size = self.sampleSize)

        Y_ = noise
        for i in range(len(YCoeffs)):
            if self.coeffMatrix[self.outcomePos][i] != 0:
                Y_ += self.coeffMatrix[self.outcomePos][i] * middleFunction(self.X[i])
        randomTCoeff = np.random.normal(0,5,self.sampleSize) # N(0,5)
        # randomTCoeff = np.random.exponential(1.5,self.sampleSize)
        # randomTCoeff = np.random.beta(1, 3, self.sampleSize)
        self.Y0 = causalMechanism(Y_ + randomTCoeff * middleFunction(np.zeros(self.sampleSize))) # y_ + Ci * f(0)
        self.Y1 = causalMechanism(Y_ + randomTCoeff * middleFunction(np.ones(self.sampleSize)))  # y_ + Ci * f(1)      3 76 4 2 41 3 42 32
        

        # y0 = S(-1) ~ 0.25 +
        # y1 = S(-0.75) ~ 0.4 +

        # y0 = S(-2) ~ 0.15 -
        # y1 = S(-2.25) ~ 0.1 -

        # y0 = yi + 5 * 0.5
        # y1 = yi + 5 * 0.75

        # y0 = yi + (-5) * 0.5
        # y1 = yi + (-5) * 0.75

    def makeDataFrame(self):
        self.df= pd.DataFrame(data= self.X.T, index=[str(i) for i in range(self.X.shape[1])], columns= ['X' + str(i) for i in range(self.X.shape[0])])
        self.df.drop(columns= ['X' + str(self.outcomePos)], inplace= True)
        self.df['Y0'] = self.Y0
        self.df['Y1'] = self.Y1
        self.saveDataFrame()
        return self.df
        
    def saveDataFrame(self):
        path = self.resourcePath + '/main_folder'
        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(path + '/graphs')
            os.mkdir(path + '/dataframes')
        key = uuid.uuid4().hex
        self.drawGeneratedGraph(path + f'/graphs/{key}.png')
        csv = self.df.to_csv(path + f'/dataframes/{key}.csv')
        
    def generateCausalX(self):
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
                        self.X_withTreatment[i] += self.coeffMatrix[i][j] * self.X_withTreatment[j]
                        self.X_withoutTreatment[i]+=self.coeffMatrix[i][j] * self.X_withoutTreatment[j]

    def makeCausalDataFrames(self):
        """
        Creates DataFrame with treated(T = [11..1]) outcome
        """
        df_withTreatment= self.makeDataFrame(self.X_withTreatment)

        """
        and this one, in whick outcome is not treated(T=[00...0])
        """
        df_withoutTreatment= self.makeDataFrame(self.X_withoutTreatment)
        return (df_withTreatment, df_withoutTreatment)
    
    def makeCausalDataFrame(self, X):
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
        
        df1 = pd.DataFrame(data=X.T, index=[str(i) for i in range(X.shape[1])], columns= clmns) # ['X' + str(i) for i in range(self.X_withTreatment.shape[0])]
        return df1

    def generateDataFrame(self, middleFunction, causalFunction):
        self.makeDiTree()
        self.generateCoeffMatrix()
        self.orderNodesTopologically()
        # --- add func
        self.generateX(middleFunction, causalFunction)
        self.addTreatmentNode()
        # self.colorGraph()
        return self.makeDataFrame()

    def generateCausalDataFrame(self):
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
        self.generateCausalX() #4
        return self.makeCausalDataFrames() #5