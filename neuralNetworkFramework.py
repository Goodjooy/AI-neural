# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy
import matplotlib.pyplot as mp
import scipy.special


# %%
class NeuralNetwork(object):
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, learn_rate=0.3):
        self.in_nodes = input_nodes
        self.out_nodes = output_nodes
        self.hide_nodes = hidden_nodes

        self.learn_rate = learn_rate

        # 创建输入层与隐藏层链接矩阵
       # self.w_in_hidden = numpy.random.rand(
       #     self.hide_nodes, self.in_nodes)-0.5
        # self.w_hiden_out = numpy.random.rand(
        #    self.out_nodes, self.hide_nodes)-0.5

        # complex init weight
        self.w_in_hidden = numpy.random.normal(
            0.0,
            pow(self.in_nodes, -0.5),
            (self.hide_nodes, self.in_nodes))

        self.w_hiden_out = numpy.random.normal(
            0.0,
            pow(self.hide_nodes, -0.5),
            (self.out_nodes, self.hide_nodes))

        self.activite_function = lambda x: scipy.special.expit(x)

    def working(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        # from input node enter hdden node
        hidden_input = numpy.dot(self.w_in_hidden, inputs)
        # use activite functiuon ,generate hiden out put
        hidden_output = self.activite_function(hidden_input)

        # from hdden nodes enter output nodes
        output_input = numpy.dot(self.w_hiden_out, hidden_output)
        # use activate function ,generate final out put
        final_output = self.activite_function(output_input)

        return hidden_output, final_output

    def train(self, input_list, target_list):
        # conven input to 2d
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hidden_list, result_list = self.working(input_list)

        # error deal

        out_put_error = targets-result_list
        # hidden error = weight_hidden_output(T转置矩阵)*error_out_put
        # 隐藏层的误差为 隐藏层与输出层的权重矩阵 （点乘） 最终结果的误差
        hidden_error = numpy.dot(self.w_hiden_out.T, out_put_error)

        # update weight that close to target
        # dw=[learn_rate]*error*singmoid(out_out)*(1-singmoid(out_put))·(last_level_out_put)(T)

        v1 = (out_put_error*result_list*(1.0-result_list))
        v2 = numpy.transpose(hidden_list)

        self.w_hiden_out += self.learn_rate * numpy.dot(v1, v2)
        self.w_in_hidden += self.learn_rate * \
            numpy.dot((hidden_error*hidden_list*(1.0-hidden_list)),
                      numpy.transpose(inputs))

    def query(self, input_list):
        _, final_output = self.working(input_list)

        return final_output


# %%
input_nodes = 784
hidden_nodes = 400
output_nodes = 10

learn_rate = 0.2

n = NeuralNetwork(input_nodes,
                  hidden_nodes,
                  output_nodes,
                  learn_rate)

with open("train\\mnist_train.csv", "r") as target:
    data_list = target.readlines()[:10000]
# train
for i in range(2):
    for single_train in data_list:
        # input data
        all_values = single_train.split(",")
        scaled_input = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

        # target data
        out_node = 10
        targets = numpy.zeros(out_node)+0.01
        targets[int(all_values[0])] = 0.99

        n.train(scaled_input, targets)

# print(n.w_in_hidden)

# %%
mp.imshow(n.w_hiden_out)
mp.imshow(n.w_in_hidden)

# %%
with open("train\\mnist_test.csv", "r") as target:
    test_data = target.readlines()

total=0
pass_test=0
for test in test_data:

    all_values = test.split(",")

    imarr = numpy.asfarray(all_values[1:]).reshape((28, 28))

    v = n.query((numpy.asfarray(all_values[1:])/255*0.99)+0.01)

    v=list(zip(*v))[0]

    maxd=max(v)

    i=v.index(maxd)

    if i==int (all_values[0]):
        pass_test+=1

    total+=1;

print((pass_test/total))


# %%
with open("res.json","w")as target:
    import json

    json.dump({
        "wih":n.w_in_hidden.tolist(),
        "who":n.w_hiden_out.tolist()
    },target,indent=4)
# %%
a=list(zip([1],[2],[3]))
print(a)
# %%
