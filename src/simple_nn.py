# based on https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter6%20-%20Intro%20to%20Backpropagation%20-%20Building%20Your%20First%20DEEP%20Neural%20Network.ipynb
import numpy as np
import matplotlib.pyplot as plt
import src.SimpleLayer as sl

# Methods ---------------------------------------------
# activation function: sets all negative numbers to zero
# Otherwise returns x
def relu(x: np.array) -> np.array :
    return (x > 0) * x

# This is the derivative of the above relu function, since the derivative of 1x is 1
def relu2deriv(output: np.array) -> np.array:
    return 1.0*(output > 0) # returns 1 for input > 0
    # return 0 otherwise

# create a layer
def create_layer(layer_name: str, neuron_count: int, target: sl.SimpleLayer = None) -> 'SimpleLayer':
    layer = sl.SimpleLayer(layer_name, neuron_count, relu, relu2deriv, target)
    layer_array.append(layer)
    return layer

# variables ------------------------------------------
np.random.seed(0)
alpha = 0.2
# the samples. Columns are the things we're sampling, rows are the samples
streetlights_array = np.array( [[ 1, 0, 1 ],
                                [ 0, 1, 1 ],
                                [ 0, 0, 1 ],
                                [ 1, 1, 1 ]])
num_streetlights = len(streetlights_array[0])
num_samples = len(streetlights_array)

# The data set we want to map to. Each entry in the array matches the corresponding streetlights_array row
walk_vs_stop_array = np.array([[1],
                               [1],
                               [0],
                               [0]])

# set up the dictionary that will store the numpy weight matrices
layer_array = []

error_plot_mat = [] # for drawing plots

#set up the layers from last to first, so that there is a target layer
output = create_layer("output", 1)
output.set_neurons([1])
''' # If we want to have four layers (two hidden), use this and comment out the other hidden code below
hidden2 = create_layer("hidden2", 2, output)
hidden2.set_neurons([1, 2])
hidden = create_layer("hidden", 4, hidden2)
hidden.set_neurons([1, 2, 3, 4])
'''
# If we want to have three layers (one hidden), use this and comment out the other hidden code above
hidden = create_layer("hidden", 4, output)
hidden.set_neurons([1, 2, 3, 4])

input = create_layer("input", 3, hidden)
input.set_neurons([1, 2, 3])

for layer in reversed(layer_array):
    print ("--------------")
    print (layer.to_string())

iter = 0
max_iter = 1000
epsilon = 0.001
error = 2 * epsilon
while error > epsilon:
    error = 0
    sample_error_array = []
    for sample_index in range(num_samples):
        input.set_neurons(streetlights_array[sample_index])
        for layer in reversed(layer_array):
            layer.train()

        delta = output.calc_delta(walk_vs_stop_array[sample_index])
        sample_error = np.sum(delta ** 2)
        error += sample_error

        for layer in layer_array:
            layer.learn(alpha)

        # Gather data for the plots
        sample_error_array.append(sample_error)
        # print("{}.{} Error = {:.5f}".format(iter, sample_index, sample_error))

    error /= num_samples
    sample_error_array.append(error)
    error_plot_mat.append(sample_error_array)
    if (iter % 10) == 0 :
        print("{} Error = {:.5f}".format(iter, error))
    iter += 1
    # stop even if we don't converge
    if iter > max_iter:
        break

print("\n--------------evaluation")
for sample_index in range(len(streetlights_array)):
    input.set_neurons(streetlights_array[sample_index])
    for layer in reversed(layer_array):
        layer.train()
    prediction = float(output.neuron_row_array)
    observed = float(walk_vs_stop_array[sample_index])
    accuracy = 1.0 - abs(prediction - observed)
    print("sample {} - input: {} = pred: {:.3f} vs. actual:{} ({:.2f}% accuracy)".
          format(sample_index, input.neuron_row_array, prediction, observed, accuracy*100.0))

# plots ----------------------------------------------
fig_num = 1
f1 = plt.figure(fig_num)
plt.plot(error_plot_mat)
plt.title("error")
names = []
for i in range(num_samples):
        names.append("sample_{}".format(i))
names.append("average")
plt.legend(names)

for layer in reversed(layer_array):
    if layer.target != None:
        fig_num += 1
        layer.plot_weight_matrix(fig_num)

for layer in reversed(layer_array):
    fig_num += 1
    layer.plot_neuron_matrix(fig_num)

plt.show()