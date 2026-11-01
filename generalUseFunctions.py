import numpy as np
import time
import random
import sympy
# Time measurement
def mesure_time(function):
    def wrapper(img, fltr):
        nonlocal function
        innit = time.time()
        b = function(img, fltr)
        final = time.time()
        print(f"{(final - innit)} - time passed")
        return b
    return wrapper


#***

# RelU

def relU(arr:np.array):
    arr[arr < 0] = 0
    return arr


#***

# Normalization


def normalization(ins:np.array):
    ma = ins.max()
    mi = ins.min()
    ins = (ins - mi) / (ma - mi)
    return ins

#***
# Logarithmic Normalization

def logarithmicNormalization(arr:np.array):
    ma = arr.max()
    return np.log(arr) / np.log(ma)

#**

# initialize randomly

def initialize_randomly(previous_outputs:int, next_layer_neuron_count:int):
    weights = []
    for i in range(previous_outputs * next_layer_neuron_count):
        weights.append(random.random())
    weights.extend([previous_outputs, next_layer_neuron_count])
    biases = []
    for i in range(next_layer_neuron_count):
        biases.append(random.random())
    return {"weights": weights,"biases": biases}

#***
# Forwardpass
def forwardpass(previous_outputs, connection_weights, neuron_biases):
    neuron_c = len(neuron_biases)
    neurons = np.array([0] * neuron_c)
    for i in range(neuron_c):
        neurons[i] = (connection_weights[i] * previous_outputs).sum() + neuron_biases[i]
    neurons = relU(neurons)
    neurons = logarithmicNormalization(neurons)
    return neurons


# BackPro
# Hala yapım aşamasında. Zor malesef :(
(* def fonk(previous_values, current_values, past, weights, biases):
    w = sympy.Matrix(weights)
    b = sympy.Matrix(biases)
    x = sympy.Matrix(previous_values)
    z = w.row(0).dot(x) + b.row(0)[0]
    print(z.diff(w.row(0)[0]))



    

fonk(np.random.random(16),np.random.random(4),1, np.random.random(64).reshape(4,16),np.random.random(16)) *)

