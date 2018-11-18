import random
import math
from collections import deque

# first activation function, the sigmoid curve
def actFunc1(x):
    return 1.0 / (1.0 + math.exp(-x))

# first update weights function
def updateWFunc1(learningRate, out, error, inp):
    return learningRate * out * (1 - out) * error * inp

# we keep track on current input, output and error value to be able to update weights
# inEdges keep track on thea the edges connected to a node in layer n from all nodes in layer n-1
# outEdges keep track on thea the edges connected from a node in layer n to all nodes in layer n+1
# addBias() connects a node with value 1.0 to the node. This incorporates the bias into the model, it is
# simple the value 1.0 * w_i (where w_i is the weight associated with this bias node)
class Node:

    def __init__(self):
        self.input = None
        self.output = None
        self.error = None
        self.inEdges = []
        self.outEdges = []
        self.addBias()

    def getOutput(self, inputVal):
        # if we just calculated the output, return it, otherwise calculate it
        if self.output is not None:
            return self.output

        self.input = []
        weightedSum = 0

        # for each input edge calculate the sum of the product between the input to the node times the weight of the input edge
        for edge in self.inEdges:
            theInput = edge.source.getOutput(inputVal)
            self.input.append(theInput)
            weightedSum += edge.weight * theInput

        self.output = actFunc1(weightedSum)
        return self.output

    def getError(self, ans):
        # if we just calculated the error, return it, otherwise calculate it
        if self.error is not None:
            return self.error

        assert self.output is not None
    
        # if output node error is caluclated through ans (= the correct value of the output)
        if self.outEdges == []:
            self.error = ans - self.output
        # otherwise backpropagate to find the error
        else:
            self.error = sum([edge.weight * edge.target.getError(ans) for edge in self.outEdges])

        return self.error

    def updateWeights(self, learningRate):
        # if input, output and error is None, we have already updated the weights for this edge
        if (self.input is not None and self.output is not None and self.error is not None):

            for i, edge in enumerate(self.inEdges):
                edge.weight += updateWFunc1(learningRate, self.output, self.error, self.input[i])

            for edge in self.outEdges:
                edge.target.updateWeights(learningRate)

            self.error = None
            self.input = None
            self.output = None

    def clearOutput(self):
        # clear the output of all nodes before we calculate new outputs
        if self.output is not None:
            self.output = None
            for edge in self.inEdges:
                edge.source.clearOutput()

    def addBias(self):
        self.inEdges.append(Edge(BiasNode(), self))

# class for the input nodes, getOutput() simple returns the value of the input vector
class InputNode(Node):

    def __init__(self, index):
        Node.__init__(self)
        self.index = index;

    def getOutput(self, inputVal):
        self.output = inputVal[self.index]
        return self.output

    # starts propagation of network to get error
    def getError(self, ans):
        for edge in self.outEdges:
            edge.target.getError(ans)

    # starts propagation of network to update weights
    def updateWeights(self, learningRate):
        for edge in self.outEdges:
            edge.target.updateWeights(learningRate)

    def clearOutput(self):
        self.output = None

    # no bias for the input nodes
    def addBias(self):
        pass

# works like the class InputNode but the output is just 1.0 (the value of the bias will be the weight of the edge
# connected to the target node
class BiasNode(InputNode):

    def __init__(self):
        Node.__init__(self)

    def getOutput(self, inputVal):
        return 1.0

# every edge get a random weight value between [0,1] when initialised 
# furthermore, the target and source nodes get connected to eachother with the associated edge
class Edge:

    def __init__(self, source, target):
        self.weight = random.uniform(0, 1)
        self.source = source
        self.target = target

        source.outEdges.append(self)
        target.inEdges.append(self)


class Network:

    def __init__(self):
        self.inNodes = []
        self.outNode = None

    def getOutput(self, inputVal):
        self.outNode.clearOutput()

        output = self.outNode.getOutput(inputVal)
        return output

    def backPropagate(self, ans):
        for node in self.inNodes:
            node.getError(ans)

    def updateWeights(self, learningRate):
        for node in self.inNodes:
            node.updateWeights(learningRate)

    def train(self, trainingData, learningRate=0.9, maxIterations=10000):
       
        # queue for animation of the network learning in class main()
        # not yet eligable for jet example
        # q = deque()
        for _ in range(maxIterations):
           
            tempx = []
            tempy = []  
        
            # inputData is the input the network will learn from
            # ans is the correct answer of the inputData, the network will use this
            # to caluclate error and with that update weights
            for inputData, ans in trainingData:
                output = self.getOutput(inputData)
                #q.append((inputData[0],output))
                #tempx.append(inputData[0])
                #tempy.append(output)

                self.backPropagate(ans)
                self.updateWeights(learningRate)
          
            #q.append((tempx, tempy))
          
        return q 
     
    def test(self, testData):

        outputGluon = []
        outputQuark = []

        for inputData, ans in testData:
            output = self.getOutput(inputData)
            if int(ans) is 1:
                outputGluon.append(output)
            else: 
                outputQuark.append(output)

        return (outputGluon, outputQuark)
 
