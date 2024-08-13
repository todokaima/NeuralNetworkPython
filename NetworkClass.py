import random
import math
class Network:
    def __init__(self, numberOfNeurons = [], numberOfInputs = 10, learningRate = 0.1, regressor = False):
        self.numberOfNeurons = numberOfNeurons
        self.numberOfNeurons.append(1)
        self.numberOfLevels = len(self.numberOfNeurons)
        self.learningRate = learningRate
        self.numberOfInputs = numberOfInputs
        self.activations = []
        self.interactions = []
        self.levels = []
        self.Regressor = regressor
        self.errorVector = []
        for level in range(self.numberOfLevels):
            if level == 0:
                numberOfWeights = self.numberOfInputs
            else:
                numberOfWeights = self.numberOfNeurons[level - 1]

            self.levels.append(self._Level(self.numberOfNeurons[level], numberOfWeights))
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
    def weights(self, layer, neuronNumber=None):
        weightAsk = []
        for neuron in self.levels[layer].neurons:
            weightVector = neuron.weights
            weightAsk.append(weightVector)
        if neuronNumber is not None:
            print(weightAsk[neuronNumber])
            return weightAsk[neuronNumber]
        else:
            print(weightAsk)
            return weightAsk
    def interact(self, input, level):
        if 0 <= level < self.numberOfLevels:
            interactions = []
            activations = []
            current_level = self.levels[level]  # Get the specific level
            for neuron in current_level.neurons:
                neuronInteraction = 0
                neuronActivation = 0
                for value, weight in zip(input, neuron.weights[1:]):
                    neuronInteraction += value * weight
                neuronInteraction += neuron.weights[0]  # Adding the bias
                neuronActivation = math.tanh(neuronInteraction)

                interactions.append(neuronInteraction)
                activations.append(neuronActivation)

                self.activations.append(activations)
                self.interactions.append(interactions)
        if level == self.numberOfLevels:
            current_level = self.levels[level]  # Get the specific level
            for neuron in current_level.neurons:
                neuronInteraction = 0
                neuronActivation = 0
                for value, weight in zip(input, neuron.weights[1:]):
                    neuronInteraction += value * weight
                neuronInteraction += neuron.weights[0]
                if self.Regressor:
                    neuronActivation = tanh(neuronInteraction)
                else:
                    neuronActivation = neuronInteraction

                interactions.append(neuronInteraction)
                activations.append(neuronActivation)

                self.activations.append(activations)
                self.interactions.append(interactions)


        return activations
    def clear(self):
        self.interactions = []
        self.activations = []
    def feedForward(self, input, target):
        for i in range(self.numberOfLevels):
            input = self.interact(input, i)
        output = input
        self.errorVector = [output[i] - target[i] for i in range(len(target))]
        return output
    def backPropagate(self, input, target):
        delta = [self.errorVector[i] * (1 - self.activations[-1][i] ** 2) for i in range(len(self.activations[-1]))]
        errors = [delta]

        for i in range(self.numberOfLevels - 2, -1, -1):
            layer_errors = []
            for j in range(self.numberOfNeurons[i]):
                error = sum(errors[-1][k] * self.levels[i + 1].neurons[k].weights[j + 1] for k in
                            range(self.numberOfNeurons[i + 1])) * (1 - self.activations[i][j] ** 2)
                layer_errors.append(error)
            errors.append(layer_errors)

        errors.reverse()

        for i in range(self.numberOfLevels):
            for j in range(self.numberOfNeurons[i]):
                for k in range(len(self.levels[i].neurons[j].weights) - 1):
                    self.levels[i].neurons[j].weights[k + 1] -= self.learningRate * errors[i][j] * (
                        self.activations[i - 1][k] if i > 0 else input[k])
                self.levels[i].neurons[j].weights[0] -= self.learningRate * errors[i][j]  # Update bias

    def train(self,input, target):
        self.clear()
        self.feedForward(input, target)
        self.backPropagate(input,target)

    def __str__(self):
        result = []
        for i, level in enumerate(self.levels):
            result.append(
                f"\nLevel {i + 1} with {level.numberOfNeurons} neurons, each with {level.numberOfWeights} weights:")
            for neuron in level.neurons:
                result.append(f"  Weights: {neuron.weights[1:]}, Bias: {neuron.weights[0]}")
        return "\n".join(result)+"\n"


network = Network([2], numberOfInputs=2, learningRate=0.5)

# XOR training data
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Print the network's initial state
print("Initial Network:")
print(network)

# Train the network
epochs = 10000
for epoch in range(epochs):
    for input_data, target in training_data:
        network.train(input_data, target)

# Print the network's final state
print("Final Network:")
print(network)

# Test the network on the training data
print("Network Output after Training:")
for input_data, target in training_data:
    output = network.feedForward(input_data, target)
    print(f"Input: {input_data}, Target: {target}, Output: {output}")


"""network = Network([5,2,3])
print(network)
network.weights(2)"""

import matplotlib.pyplot as plt

# Assume the Network class is defined here as you have provided.

# Function to generate noisy sine wave data
def generate_data(samples=100, noise=0.1):
    data = []
    for i in range(samples):
        x = random.uniform(-2 * math.pi, 2 * math.pi)
        y = math.sin(x) + random.uniform(-noise, noise)
        data.append(([x], [y]))
    return data

# Initialize the network with 1 input, one hidden layer with 10 neurons, and 1 output neuron
network = Network([10], numberOfInputs=1, learningRate=0.01)

# Generate training data
training_data = generate_data(samples=100, noise=0.1)

# Train the network
epochs = 10000
for epoch in range(epochs):
    for input_data, target in training_data:
        network.train(input_data, target)

# Test the network
test_data = generate_data(samples=50, noise=0.0)  # Test without noise

inputs = [x[0][0] for x in test_data]
targets = [x[1][0] for x in test_data]
predictions = []

for input_data, target in test_data:
    output = network.feedForward(input_data, target)
    predictions.append(output[0])

# Plot the results
plt.plot(inputs, targets, 'bo', label='Target')
plt.plot(inputs, predictions, 'ro', label='Prediction')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.title('Sine Function Approximation by Neural Network')
plt.show()
