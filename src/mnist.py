'''
Copyright 2019 Philip Feldman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

# based on https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.datasets import mnist
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
np.random.seed(1)
alpha = 0.005
pixels_per_image = 28*28
num_hidden = 40

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,pixels_per_image) / 255, y_train[0:1000])

num_images = len(images)
num_labels = len(labels)
num_x_test = len(x_test)
num_y_test = len(y_test)


one_hot_labels = np.zeros((num_labels,10))
for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(num_x_test, pixels_per_image) / 255
test_labels = np.zeros((num_y_test,10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1


# set up the dictionary that will store the numpy weight matrices
layer_array = []

error_plot_mat = [] # for drawing plots

#set up the layers from last to first, so that there is a target layer
num_label_neurons = len(labels[0])
output = create_layer("output", num_label_neurons)
#output.set_neurons([1])
''' # If we want to have four layers (two hidden), use this and comment out the other hidden code below
hidden2 = create_layer("hidden2", 2, output)
hidden2.set_neurons([1, 2])
hidden = create_layer("hidden", 4, hidden2)
hidden.set_neurons([1, 2, 3, 4])
'''
# If we want to have three layers (one hidden), use this and comment out the other hidden code above
hidden = create_layer("hidden", num_hidden, output)
#hidden.set_neurons([1, 2, 3, 4])

input = create_layer("input", pixels_per_image, hidden)
#input.set_neurons([1, 2, 3])

for layer in reversed(layer_array):
    print ("--------------")
    print (layer.to_string())

iter = 0
max_iter = 10
epsilon = 0.1
error = 2 * epsilon
while error > epsilon:
    error = 0
    sample_error_array = []
    for sample_index in range(num_images):
        # print("iter = {}, sample = {}".format(iter, sample_index))
        inp_vec = images[sample_index:sample_index+1][0]
        input.set_neurons(inp_vec)
        for layer in reversed(layer_array):
            layer.train()

        targ_vec = labels[sample_index:sample_index+1][0]
        delta = output.calc_delta(targ_vec)
        sample_error = np.sum(delta ** 2)
        error += sample_error

        for layer in layer_array:
            layer.learn(alpha)

        # Gather data for the plots
        sample_error_array.append(sample_error)
        # print("{}.{} Error = {:.5f}".format(iter, sample_index, sample_error))

    error /= num_images
    sample_error_array.append(error)
    error_plot_mat.append(sample_error_array)
    if (iter % 1) == 0 :
        print("{} Error = {:.5f}".format(iter, error))
    iter += 1
    # stop even if we don't converge
    if iter > max_iter:
        break

print("\n--------------evaluation")
num_test_images = len(test_images)
for sample_index in range(num_test_images):
    inp_vec = test_images[sample_index:sample_index+1][0]
    input.set_neurons(inp_vec)
    for layer in reversed(layer_array):
        layer.train()
    prediction_vec = output.neuron_row_array[0]
    targ_vec = test_labels[sample_index:sample_index+1][0]
    diff_vec = np.subtract(prediction_vec, targ_vec)
    observed = mean_squared_error(prediction_vec, targ_vec)
    accuracy = 1.0 - observed
    if sample_index % 100 == 0:
        print("sample {} observed:{} ({:.2f}% accuracy)".
              format(sample_index, observed, accuracy*100.0))

# plots ----------------------------------------------
fig_num = 1
f1 = plt.figure(fig_num)
plt.plot(error_plot_mat)
plt.title("error")
if num_images < 10:
    names = []
    for i in range(num_images):
        names.append("sample_{}".format(i))
    names.append("average")
    plt.legend(names)

for layer in reversed(layer_array):
    if layer.target != None:
        fig_num += 1
        layer.plot_weight_matrix("weights", fig_num)

for layer in reversed(layer_array):
    fig_num += 1
    layer.plot_neuron_matrix(fig_num)

plt.show()