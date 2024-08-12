import random
import math
class Network:
    def __init__(self, numberOfNeurons = [], numberOfInputs = 10):
        self.numberOfNeurons = numberOfNeurons
        self.numberOfNeurons.append(1)
        self.numberOfLevels = len(self.numberOfNeurons)

        self.numberOfInputs = numberOfInputs
        self.levels = []

        for level in range(self.numberOfLevels):
            if level == 0:
                numberOfWeights = self.numberOfInputs
            else:
                numberOfWeights = self.numberOfNeurons[level - 1]

            levelInit = self._Level(self.numberOfNeurons[level], numberOfWeights)
            self.levels.append(levelInit)

    def __str__(self):
        result = []
        for i, level in enumerate(self.levels):
            result.append(f"\nLevel {i + 1} with {level.numberOfNeurons} neurons, each with {level.numberOfWeights} weights:")
            for neuron in level.neurons:
                result.append(f"  Weights: {neuron.weights[1:]}, Bias: {neuron.weights[0]}")
        return "\n".join(result)

    class _Level:
        def __init__(self, numberOfNeurons, numberOfWeights):
            self.numberOfNeurons = numberOfNeurons
            self.numberOfWeights = numberOfWeights
            self.neurons = [self._Neuron(numberOfWeights) for _ in range(numberOfNeurons)]

        class _Neuron:
            def __init__(self, numberOfWeights, bias=-1):
                self.numberOfWeights = numberOfWeights
                self.bias = bias
                self.weights = [self.bias]

                for _ in range(numberOfWeights):
                    randomValue = random.uniform(-1,1)
                    self.weights.append(randomValue)

    def neuronsOfLevel(self, level,P=False):
        levelNeurons = []
        if level < 0 and level >= len(self.levels):
            print(f"Level {level} is out of range.")
            return None
        for neuron in self.levels[level].neurons:
            weightList = []
            for weight in neuron.weights:
                weightList.append(weight)
            levelNeurons.append(weightList)
        if P:
            print(levelNeurons)
        return levelNeurons
    def interact(self,data,level):
        if level >=0 or level < self.numberOfNeurons:
            neuronInteraction = 0
            activation = []
            for neuron in self.levels[level].neurons:
                neuronInteraction = 0
                for value, weight in zip(data, neuron.weights[1:]):
                    neuronInteraction += value*weight
                neuronInteraction += neuron.weights[0]
                neuronInteraction = math.tanh(neuronInteraction)
                activation.append(neuronInteraction)
            return activation

    def feedForward(self, point):
        input = point
        for i in range(self.numberOfLevels):
            input = self.interact(input, i)
        output = input
        return output

    def backPropagation(self, point, target, neuronsList):
        activations = [point]
        for i in range(self.numberOfLevels):
            input = self.interact(input,i)
            activations.append(input)

        output = activations[-1][0]
        delta = [(output - target)*(1 - output**2)]

        for lvl in range(len(self.numberOfLevels-1,0,-1)):
            for neuron in neuronsList[lvl]:
                pass

numberOfNeurons = [4, 5, 3]  # Number of neurons in each level
numberOfInputs = 3
network = Network(numberOfNeurons, numberOfInputs)


print("_____________")
input_data = [0.5, -0.2, 0.8]
output = network.feedForward(input_data)
print(output)
print("output")
targets = [-20, 13.4, 42]
print(network)
print(network.neuronsOfLevel(1))