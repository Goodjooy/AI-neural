# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import neuralNetworkFramework
from IPython import get_ipython

# %%
import numpy
import matplotlib.pyplot as mp

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
with open("train\\mnist_train.csv", "r") as target:
    data_list = target.readlines()[:500]


data_list[0]
# %%
len(data_list)


# %%
# initialization train data

all_values = data_list[1].split(",")
img_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

mp.imshow(img_array, cmap="Greys", interpolation="None")


# %%

scaled_input = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

print(scaled_input)


# %%
# target putput
out_node = 10
targets = numpy.zeros(out_node)+0.01
targets[int(all_values[0])] = 0.99

targets

# %%
len(all_values)

# %%

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learn_rate = 0.3

n = neuralNetworkFramework.NeuralNetwork(input_nodes,
                                         hidden_nodes,
                                         output_nodes,
                                         learn_rate)

#train

for single_train in data_list:
    #input data
    all_values = data_list[1].split(",")
    scaled_input = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

    #target data
    out_node = 10
    targets = numpy.zeros(out_node)+0.01
    targets[int(all_values[0])] = 0.99

    n.train(scaled_input,targets)

print(n.w_hiden_out)
print(n.w_in_hidden)
# %%
