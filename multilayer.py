"""
Multilayer Adaline
"""
import random
import numpy as np
import pylab
import matplotlib.pyplot as plt

NEURONS = 2
LAYERS = 2
NEURONS_OUTPUT = 1
INPUTFILE = 'input-values.txt'
DESIREDFILE = 'desired-values.txt'
ERROR = 0.1
ETA = 0.4
BOTTOM_LIMIT = -1
TOP_LIMIT = 1


class DataAccessObject:
    inputData = []
    desiredOutput = []

    def __init__(self):
        self.get_data()

    def get_data(self):
        file = open(INPUTFILE, "r")
        # next(file)
        for line in file:
            fields = line.split(" ")
            size = len(fields)
            if not self.inputData:
                for value in range(0, size):
                    self.inputData.append([])
            for value in range(0, size):
                self.inputData[value].append(float(fields[value]))
        file = open(DESIREDFILE, "r")
        # next(file)
        for line in file:
            fields = line.split(" ")
            size = len(fields)
            if not self.desiredOutput:
                for value in range(0, size):
                    self.desiredOutput.append([])
            for value in range(0, size):
                self.desiredOutput[value].append(float(fields[value]))


class Adaline(DataAccessObject):
    inputData = []
    inputColumns = None
    inputRows = None
    weightData = []
    innerWeights = []
    bias = []  # bias
    innerOutput = []
    output = []
    desiredOutput = []
    desiredColumns = None
    desiredRows = None
    innerV = []
    lastv = []
    innerSummation = []
    summation = []
    iteration = 0
    errorAsc = 0
    errorDesc = 0

    def __init__(self):
        self.inputData = DataAccessObject.inputData.copy()
        self.desiredOutput = DataAccessObject.desiredOutput.copy()
        self.inputColumns = len(self.inputData)
        self.inputRows = len(self.inputData[0])
        self.desiredColumns=len(self.desiredOutput)
        self.desiredRows = len(self.desiredOutput[0])
        self.generate_random_values()
        self.main_algorithm()

    def generate_random_values(self):
        for dataSet in range(0, self.desiredColumns):
            self.weightData.append([])
            for neuron in range(0, NEURONS):
                self.weightData[dataSet].append([])
                for value in range(0, self.inputColumns):
                    self.weightData[dataSet][neuron].append(random.uniform(BOTTOM_LIMIT, TOP_LIMIT))
#                    self.weightData[dataSet][neuron].append(value)
        for dataSet in range(0, self.desiredColumns):
            self.bias.append([])
            for layer in range(0, LAYERS):
                self.bias[dataSet].append(random.uniform(BOTTOM_LIMIT, TOP_LIMIT))   
#                self.bias[dataSet].append(layer)   
        for column in range(0, self.desiredColumns):
            self.innerWeights.append([])
            for neuron in range(0, NEURONS):
                self.innerWeights[column].append(random.uniform(BOTTOM_LIMIT, TOP_LIMIT))
#                self.innerWeights[column].append(neuron)

    def calculate_output(self):
        for dataSet in range(0, self.desiredColumns):
            self.innerOutput.append([])
            self.output.append([])
            self.innerSummation.append([])
            self.innerV.append([])
            self.lastv.append([])
            self.summation.append([])
            for neuron in range(0, NEURONS):
                self.innerSummation[dataSet].append([])
                self.innerV[dataSet].append([])
                self.innerOutput[dataSet].append([])
                for row in range(0, self.inputRows):
                    self.innerSummation[dataSet][neuron].append(0) #Start summation in 0 
                    self.innerV[dataSet][neuron].append(0) 
                    for columnInput in range(0,self.inputColumns):
                        self.innerSummation[dataSet][neuron][row] += \
                            self.inputData[columnInput][row] * self.weightData[dataSet][neuron][columnInput]
                    self.innerV[dataSet][neuron][row] = self.innerSummation[dataSet][neuron][row] + self.bias[dataSet][1] #changed, summed bias
                    self.innerOutput[dataSet][neuron].append(self.activation_function(self.innerV[dataSet][neuron][row]))
            for neuron in range(0, NEURONS):
                
                for row in range(0, self.inputRows):
                    self.summation[dataSet].append(0) 
                    self.summation[dataSet][row] += self.innerOutput[dataSet][row] * self.innerWeights[dataSet][] 
        self.iteration += 1
        print("Weights: " + str(self.weightData))
        print("innerWeights: " + str(self.innerWeights))
        print("innerOutpout: " + str(self.innerOutput))
        print("innerSummation: " + str(self.innerSummation))
        print("innerV: "+ str(self.innerV))


    def activation_function(self, value):
        output = 1 / (1 + np.exp(value * -1))
        return output

    def is_the_desired_output(self):
        for column in range(0, self.desiredColumns):
            for row in range(0,self.inputRows):
                error = self.desiredOutput[column][row] - self.output[column][row]
                if abs(error) <= ERROR:
                    pass
                else:
                    return False
        return True

    def training(self):
        for column in range(0, self.desiredColumns):
            for row in range(0,self.inputRows):
                error = self.desiredOutput[column][row] - self.output[column][row]
                if abs(error) > ERROR:
                    self.bias[column] = self.bias[column] + (ETA * error * self.output[column][row] *(1 - self.output[column][row])) 
                    for columnInput in range(0, self.inputColumns):
                        self.weightData[column][columnInput] = self.weightData[column][columnInput] +\
                                                    (ETA * error * self.inputData[columnInput][row] * self.output[column][row] *(1 - self.output[column][row])) 
               
    def print_plot(self):
        for column in  range(0, self.desiredColumns):
            #x2 = (self.bias[column] / self.weightData[column][1] - self.weightData[column][0]/ self.weightData[column][1] * self.inputData[0][0])
            x = np.linspace(-2, 2, 4)
            formulaPlot = (-1 * self.bias[column] / self.weightData[column][1]) - (self.weightData[column][0] / self.weightData[column][1] * x)
            if self.desiredOutput[column][0] == 0:
                plt.plot(0,0, 'x', color = 'red')
            else:
                plt.plot(0,0, 'ro', color = 'green')
            if self.desiredOutput[column][1] == 0:
                plt.plot(0,1, 'x', color = 'red')
            else:
                plt.plot(0,1, 'ro', color = 'green')
            if self.desiredOutput[column][2] == 0:
                plt.plot(1,0, 'x', color = 'red')
            else:
                plt.plot(1,0, 'ro', color = 'green')
            if self.desiredOutput[column][3] == 0 :
                plt.plot(1,1, 'x', color = 'red')
            else:
                plt.plot(1,1, 'ro', color = 'green')
            pylab.plot(x, formulaPlot, color = "blue")
            pylab.show()

    def main_algorithm(self):
        self.calculate_output()
        '''
        self.print_data()
        while not self.is_the_desired_output():
            print(self.iteration)
            self.training()
            self.calculate_output()
            self.print_data()
        print("Finished")
        self.print_plot()
        '''        

    def print_data(self):
        print("Iteraciones = " + str(self.iteration))
        print("ETA = " + str(ETA))
        print("Salida deseada:")
        print(self.desiredOutput)
        print("Salidas:")
        print(self.output)


dao = DataAccessObject()
adaline = Adaline()

