# based on https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter6%20-%20Intro%20to%20Backpropagation%20-%20Building%20Your%20First%20DEEP%20Neural%20Network.ipynb
import numpy as np
import matplotlib.pyplot as plt
import types
import typing

# methods --------------------------------------------
class SimpleLayer:
    name = "unset"
    neuron_row_array = None
    neuron_col_array = None
    weight_row_mat = None
    weight_col_mat = None
    plot_mat = [] # for drawing plots
    neuron_history_mat = []
    num_neurons = 0
    delta = 0 # the amount to move the source layer
    target = None
    source = None
    activation_func = None
    derivative_func = None

    # set up the layer with the number of neurons, the next layer in the sequence, and the activation/backprop functions
    def __init__(self, name, num_neurons: int, activation_ptr: types.FunctionType, deriv_ptr: types.FunctionType, target: 'SimpleLayer' = None):
        self.reset()
        self.activation_func = activation_ptr
        self.derivative_func = deriv_ptr
        self.name = name
        self.num_neurons = num_neurons
        self.neuron_row_array = np.zeros((1, num_neurons))
        self.neuron_col_array = np.zeros((num_neurons, 1))
        # We only have weights if there is another layer below us
        for i in range(num_neurons):
            self.neuron_history_mat.append([])
        if(target != None):
            self.target = target
            target.source = self
            self.weight_row_mat = 2 * np.random.random((num_neurons, target.num_neurons)) - 1
            self.weight_col_mat = self.weight_row_mat.T

    def reset(self):
        self.name = "unset"
        self.target = None
        self.neuron_row_array = None
        self.neuron_col_array = None
        self.weight_row_mat = None
        self.weight_col_mat = None
        self.plot_mat = [] # for drawing plots
        self. neuron_history_mat = []
        self.num_neurons = 0
        self.delta = 0 # the amount to move the source layer
        self.target = None
        self.source = None

    # Fill neurons with values
    def set_neurons(self, val_list: typing.List):
        # print("cur = {}, input = {}".format(self.neuron_array, val_list))
        for i in range(0, len(val_list)):
            self.neuron_row_array[0][i] = val_list[i]
        self.neuron_col_array = self.neuron_row_array.T

    def get_plot_mat(self) -> typing.List:
        return self.plot_mat

    # In training, the basic goal is to set a value for the layer's neurons, based on the weights in the source layer mediated by an activation function.
    # This matrix is simply the source neurons times this layer's neurons. For example, if the source layer had three neurons and this layer had four, then
    # the (source) weight matrix would be 3*4 = 12 weights.
    def train(self):
        # if not the bottom layer, we can record values for plotting
        if(self.target != None):
            self.plot_mat.append(self.nparray_to_list(self.weight_row_mat))

        # if we're not the top layer, propagate weights
        if self.source != None:
            src = self.source
            # set our neuron values as the dot product of the source neurons, and the source weights
            self.neuron_row_array = np.dot(src.neuron_row_array, src.weight_row_mat)

            # No activation function to output layer
            if(self.target != None):
                # Adjust the values based on the activation function. This introduces nonlinearity.
                # For example, the relu function clamps all negative values to zero
                self.neuron_row_array = self.activation_func(self.neuron_row_array)

            # Transpose the neuron array and save for learn()
            self.neuron_col_array = self.neuron_row_array.T

        for i in range(self.num_neurons):
            self.neuron_history_mat[i].append(self.neuron_row_array[0][i])


    # In learning, the basic goal is to adjust the weights that set this layer's neurons (in this implementation, the source layer). This is done
    # by backpropagating the error delta from this layer to the source layer. Since we only want to adjust the weights that participated in the
    # training, we need to take the derivative of the activation function in train(). Again, the weight matrix is simply the source neurons times
    # this layer's neurons. For example, if the source layer had three neurons and this layer had four, then the (source) weight matrix would be 3*4 = 12 weights.
    def learn(self, alpha):
        # if there is a layer above us
        if self.source != None:
            src = self.source

            # calculate the error delta scalar array, which is the amount this layer needs to change,
            # multiplied across the weights used to set this layer (in the source)
            delta_scalar = np.dot(self.delta, src.weight_col_mat)

            # determine the backpropagation distribution. In the case of Relu, it's just one or zero
            delta_threshold = self.derivative_func(src.neuron_row_array)

            # set the amount the source layer needs to change, based on this layer's delta distributed over the source
            # neurons
            src.delta = delta_scalar * delta_threshold

            # create the weight adjustment matrix by taking the dot product of the source layer's neurons (as columns) and the
            # scaled, thresholded  row of deltas based on this layer's error delta and the source's weight layer
            mat = np.dot(src.neuron_col_array, self.delta)

            # add some percentage of the weight adjustment matrix to the source weight matrix
            src.weight_row_mat += alpha * mat
            src.weight_col_mat = src.weight_row_mat.T

    # given one or more goals (that match the number of neurons in this layer), determine the delta that, when added to the
    # neurons, would reach that goal
    def calc_delta(self, goal: np.array) -> float:
        self.delta = goal - self.neuron_row_array
        return self.delta

    # helper function to turn a NumPy array to a Python list
    def nparray_to_list(self, vals: np.array) -> typing.List[float]:
        data = []
        for x in np.nditer(vals):
            data.append(float(x))
        return data

    def to_string(self):
        target_name = "no target"
        source_name = "no source"
        if self.target != None:
            target_name = self.target.name
        if self.source != None:
            source_name = self.source.name
        return "layer {}: \ntarget = {}\nsource = {}\nneurons (row) = {}\nweights (row) = \n{}".format(self.name, target_name, source_name, self.neuron_row_array, self.weight_row_mat)

    # create a line chart of the plot matrix that we've been building
    def plot_weight_matrix(self, var_name: str, fig_num: int):
        title = "{} to {} {}".format(self.name, self.target.name, var_name)
        plt.figure(fig_num)
        np_mat = np.array(self.plot_mat)

        i = 0
        for row in np_mat.T:
            cstr = "C{}".format(i % self.num_neurons)
            plt.plot(row, linewidth = int(i / self.num_neurons)+1, color=cstr)
            i += 1

        names = []
        num_weights = self.num_neurons * self.target.num_neurons
        for i in range(num_weights):
            src_n = i % self.num_neurons
            targ_n = int(i/self.num_neurons)
            names.append("{} s:t[{}:{}]".format(var_name, src_n, targ_n))
        plt.legend(names)
        plt.title(title)

    def plot_neuron_matrix(self, fig_num: int):
        title = "{} neuron history".format(self.name)
        plt.figure(fig_num)
        np_mat = np.array(self.neuron_history_mat)

        plt.plot(np_mat.T, '-o', linestyle=' ', ms=2)

        names = []
        for i in range(self.num_neurons):
            names.append("neuron {}".format(i))
        plt.legend(names)
        plt.title(title)