from neuralnetwork import *
import pandas as pd
from root_pandas import read_root
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def makeNetwork(numInputs, numHiddenLayers, numInEachLayer):
    network = Network()
    inNodes = [InputNode(i) for i in range(numInputs)]
    outNode = Node()
    network.outNode = outNode
    network.inNodes.extend(inNodes)

    layers = [[Node() for _ in range(numInEachLayer)] for _ in range(numHiddenLayers)]

    # connect input node to node of first layers
    for inputNode in inNodes:
        for node in layers[0]:
            Edge(inputNode, node)

    # connect the nodes in layer n with nodes in layer n + 1 
    for layer1, layer2 in [(layers[i], layers[i + 1]) for i in range(numHiddenLayers - 1)]:
        for node1 in layer1:
            for node2 in layer2:
                Edge(node1, node2)

    # connect last layer nodes to ouput node
    for node in layers[-1]:
        Edge(node, outNode)

    return network

# balance input data for training in case the data is biased against one output
def balanceData(df, numObject, par):

     df_ = df.copy()
     df_gluon = df_.loc[df_['isPhysG'] == 1]
     df_quark = df_.loc[df_['isPhysG'] == 0]

     mini = min(len(df_gluon.index), len(df_quark.index))
     if mini >  numObject:
         mini = numObject    
     return pd.concat(df_gluon.iloc[0:mini], df_quark.iloc[0:mini])

# split data into separate dataframes for training and testing
def createTestTrainData(numObject, split):

    par = ['QG_mult','QG_ptD','QG_axis2','isPhysG']
    df = read_root("MC_sample.root", columns=par)
    df_ = balanceData(df, numObject, par)

    df_train = df_.iloc[0:split]
    df_test = df.iloc[(split + 1):-1]
    return (df_train, df_test)

# an example that can differentiate between jets that originated from a qluon or a quark in a proton-proton collision
def jetTest(numLayers, numNodes, maxIt):
    
    data = createtTestTrainData(1000, 800)
    trainData = data[0]
    testData = data[1] 
    mult, ptd, axis2, gluon = trainData.T
    multT, ptdT, axis2T, gluonT = testData.T

    print("Number of input nodes: {}\nNumber of layers: {}\nNumber of nodes per layer: {}\nIterations: {} ".format(3, numLayers, numNodes, maxIt))
    # create neural network
    network = makeNetwork(3, numLayers, numNodes)
    # create tuples to train the network with
    trainExamples = [((mult[x],ptd[x],axis2[x]), gluon[x]) for x in  range(len(mult))]
    # train the network
    training = network.train(trainExamples, learningRate=0.25, maxIterations = maxIt)
    # create tuples to test the network with
    testExamples = [((multT[x],ptdT[x],axis2T[x]), gluonT[x]) for x in  range(len(multT))]
    # test the network
    output = network.test(testExamples)

    # draw the result
    binning = np.arange(0.0,1.0,0.05)
    plt.hist(output[0], bins=binning, alpha=0.8, label="Gluons", normed=1)   
    plt.hist(output[1], bins=binning, alpha=0.8, label="Quarks", normed=1)
    plt.legend()
    plt.show()   
   
    # print error 
    errors = [abs(gluon[x] - network.getOutput((mult[x],ptd[x],axis2[x]))) for x in range(len(mult))]
    print("Avg error: %.4f" % (sum(errors) * 1.0 / len(errors)))
    

jetTest(1, 20, 100)
