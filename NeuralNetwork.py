import math
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1,1)
        self.preva = 0
        self.prevz = 0
        
    def activation(self, x):
        return 1/(1 + math.exp(-x))
        
    def forwardpropagation(self, inputs):
        self.inputs = inputs
        z = sum([w*x for w,x in zip(self.inputs, self.weights)]) + self.bias
        a = self.activation(z)
        self.preva = a
        self.prevz = z
        return a
    
class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]
        self.prevactions = []
        
    def forwardpropagation(self, inputs):
        activations = [neuron.forwardpropagation(inputs) for neuron in self.neurons]
        self.prevactions = activations
        return activations
    
class NeuralNetwork:
    def __init__(self, layerdimensions):
        self.layers = []
        for i in range(1, len(layerdimensions)):
            self.layers.append(Layer(layerdimensions[i], layerdimensions[i-1]))
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forwardpropagation(inputs)
        return inputs
    
    def print_network(self, input_vector):
        print("=== Neural Network ===")
        print(f"Input Vector: {input_vector}")
        
        for i in range(1, len(self.layers)):
            print(f"\nLayer {i}: ")
            outputs = self.layers[i].forwardpropagation(input_vector)
            
            for jindex, neuron in enumerate(self.layers[i].neurons, start = 1):
                print(f"\tNeuron {jindex}: Activation Output = {neuron.preva} | Weighted Sum = {neuron.prevz} \n Weights: {[round(w, 4) for w in neuron.weights]} Bias: {round(neuron.bias, 4)}")
            input_vector = outputs
            
        print(f"\nOutput Vector/Final Activation: {self.forward(input_vector)}")
              

    
