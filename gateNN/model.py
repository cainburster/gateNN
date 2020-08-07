import os
import sys
import numpy as np
import tensorflow as tf

# gate simulation with relating parameter
kernal_pattern = {'AND': {'weight': [[1],[1],[1]], 'bias': [-1]}, # c = ab, 0001
           'NAND': {'weight': [[-1],[-1],[-1]], 'bias': [1]}, # c = ~(ab), 1110
           'AND2_PN': {'weight': [[1],[-1],[-1]], 'bias': [-1]}, # c = a(~b), 0010
           'NAND2_PN': {'weight': [[-1],[1],[1]], 'bias': [1]}, # c = ~(a(~b)), 1101
           'AND2_NP': {'weight': [[-1],[1],[-1]], 'bias': [-1]}, # c = (~a)b, 0100
           'NAND2_NP': {'weight': [[1],[-1],[1]], 'bias': [1]}, # c = ~(~(a)b), 1011
           'NOR': {'weight': [[-1],[-1],[1]], 'bias': [-1]}, # c = (~a)(~b), 1000
           'OR': {'weight': [[1],[1],[-1]], 'bias': [1]}, # c = ~((~a)(~b)), 011
           'XOR': {'weight': [[0],[0],[-1]], 'bias': [0]}, # c = a^b, 0110
           'XNOR': {'weight': [[0],[0],[1]], 'bias': [0]}, # c = a|b, 1001
          }

normal_pattern = {'AND': {'weight': [[1],[1]], 'bias': [-1]}, # c = ab, 0001
           'NAND': {'weight': [[-1],[-1]], 'bias': [1]}, # c = ~(ab), 1110
           'AND2_PN': {'weight': [[1],[-1]], 'bias': [-1]}, # c = a(~b), 0010
           'NAND2_PN': {'weight': [[-1],[1]], 'bias': [1]}, # c = ~(a(~b)), 1101
           'AND2_NP': {'weight': [[-1],[1]], 'bias': [-1]}, # c = (~a)b, 0100
           'NAND2_NP': {'weight': [[1],[-1]], 'bias': [1]}, # c = ~(~(a)b), 1011
           'NOR': {'weight': [[-1],[-1]], 'bias': [-1]}, # c = (~a)(~b), 1000
           'OR': {'weight': [[1],[1]], 'bias': [1]}, # c = ~((~a)(~b)), 011
          }

def judge_gate_type(lut):
    gates_type = ["LOW", "AND", "AND2_PN", "BUF1", 
                  "AND2_NP", "BUF2", "XOR", "OR",
                  "NOR", "XNOR", "NOT2", "NAND2_NP",
                  "NOT1", "NAND2_PN", "NAND", "HIGH"]
    default_type = [None, None, None, None]
    for i, t in zip([0,1,2,3], ["-1, -1", "-1, 1", "1, -1", "1, 1"]):
        if t in lut:
            default_type[i] = [1] if lut[t] == "1" else [0]
        else:
            default_type[i] = [0, 1]
    ans = []
    for i in default_type[0]:
        for j in default_type[1]:
            for k in default_type[2]:
                for l in default_type[3]:
                    index = i * 8 + j * 4 + k * 2 + l
                    ans.append(gates_type[index])
    return ans

def ternary_option(num):
    if np.abs(num) < 0.1:
        return 0
    elif num >= 0.1:
        return 1
    else:
        return -1
fun = np.vectorize(ternary_option)

def binary_option(num):
    if num > 0:
        return 1
    else:
        return -1
fun2 = np.vectorize(binary_option)


class DAGmodel:
    def __init__(self, input_names, connections, output_names, model_type="kernel", fan_out_training=True, training_rate=0.01, optimizer="momentum", has_bias=True):
        assert model_type in ["kernel", "normal"]
        #tf.reset_default_graph()
        self.input_nodes = input_names.copy()
        self.output_nodes = output_names.copy()
        with tf.name_scope('input'):
            self.trueInputs = tf.placeholder(tf.float32, shape = (None, len(self.input_nodes)), name = 'ins')
            self.trueOut = tf.placeholder(tf.float32, shape = (None, len(self.output_nodes)), name = 'outs')
        self.build_model(connections, model_type, fan_out_training, has_bias)
        self.build_loss()
        self.build_optimizer(training_rate, optimizer)
    
    def build_gates(self, trainable, namelist, connection, model_type, fan_out_training, has_bias):
        # sign function forward, tanh backward
        def binary_activation(activation):
            t = tf.clip_by_value(activation, -3.5, 3.5)
            return t + tf.stop_gradient(tf.sign(activation) - t)
        
        # clip forward and backward for training node
        def train_activation(activation):
            t = tf.clip_by_value(activation, -1.5, 1.5)
            return t + tf.stop_gradient(tf.sign(activation) - t)
        
        def clip_activation(activation):
            return tf.clip_by_value(activation, -1, 1)
        
        def binaryDense(ins, 
                        unit, 
                        activation, 
                        bias=True, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                        bias_initializer=tf.zeros_initializer(), 
                        trainable=True, 
                        name="binarydense"):
            in_units = ins.get_shape().as_list()[1]
            #print(ins.shape)
            with tf.variable_scope(name):
                w = tf.get_variable("weight", [in_units, unit], initializer=kernel_initializer, trainable=trainable)
                w = tf.clip_by_value(w, -1, 1)
                #print(w.shape)

                #w = activation(w)

                #affined_in = tf.stack([self.nodes[in1], self.nodes[in2]], axis = 1)
                #self.nodes[out] = tf.reshape(linear_model(affined_in), [-1])
                dot = tf.matmul(ins, w)
                if bias:
                    b = tf.get_variable('bias', [unit], initializer=bias_initializer, trainable=trainable)
                    dot = tf.nn.bias_add(dot, b)
                out = tf.reshape(activation(dot), [-1])
                tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, out) 
                return out
        
        in1, in2, out = namelist
        if trainable == True:
            if model_type=="kernel":
                affined_in = tf.stack([self.nodes[in1], self.nodes[in2], self.nodes[in1] * self.nodes[in2]], axis = 1)
            elif model_type=="normal":
                affined_in = tf.stack([self.nodes[in1], self.nodes[in2]], axis = 1)
            self.nodes[out] = binaryDense(ins=affined_in, 
                                          unit=1, 
                                          activation=binary_activation, 
                                          bias=has_bias,
                                          kernel_initializer=tf.random_uniform_initializer(-0.7,0.7), 
                                          bias_initializer=tf.zeros_initializer(),
                                          trainable=True,
                                          name=out+"-hidden_act")
            self.gatesSimulation[out] = (True, connection)
        else:
            if fan_out_training and trainable == 'train':
                train_sym = True
                coeff = 0.5
                coeff_bias = 0.5
            else:
                train_sym = False
                coeff = 0.5
                coeff_bias = 0.5
            if connection == 'BUF':
                self.nodes[out] = self.nodes[in1]
            elif connection == 'NOT':
                self.nodes[out] = -1 * self.nodes[in1]
            else:
                if model_type=="kernel":
                    p = kernal_pattern
                    affined_in = tf.stack([self.nodes[in1], self.nodes[in2], self.nodes[in1] * self.nodes[in2]], axis = 1)
                elif model_type=="normal":
                    p = normal_pattern
                    affined_in = tf.stack([self.nodes[in1], self.nodes[in2]], axis = 1)              
                self.nodes[out] = binaryDense(ins=affined_in, 
                                              unit=1, 
                                              activation=binary_activation, 
                                              bias=has_bias, 
                                              kernel_initializer=tf.constant_initializer(np.array(p[connection]['weight']) * coeff), 
                                              bias_initializer=tf.constant_initializer(np.array(p[connection]['bias']) * coeff_bias),
                                              trainable=train_sym, 
                                              name=out+"-route_act")        
                if fan_out_training and trainable == 'train':
                    self.gatesSimulation[out] = (False, connection)


    # connections list ele: [trainable, [in1, in2, out], *connectionPattern]
    def build_model(self, connections, model_type, fan_out_training, has_bias):
        self.nodes = {}
        self.gatesSimulation = {}
        for index, name in enumerate(self.input_nodes):
            self.nodes[name] = self.trueInputs[:,index]
        for item in connections:
            trainable, namelist, connectionPattern = item
            self.build_gates(trainable, namelist, connectionPattern, model_type, fan_out_training, has_bias)
        self.prediction = tf.stack([self.nodes[name] for name in self.output_nodes], axis = 1)
        
    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.trueOut))
    
    def build_optimizer(self, trainingrate, optimizer):
        hidden_var = [item for item in tf.trainable_variables() if "hidden_act" in item.name]
        known_var = [item for item in tf.trainable_variables() if "route_act" in item.name]
        if len(known_var) == 0:
            if optimizer == "momentum":
                opt = tf.train.MomentumOptimizer(trainingrate, 0.9)
            else:
                opt = tf.train.GradientDescentOptimizer(trainingrate)
            self.opt = opt.minimize(self.loss)
        else:
            if optimizer == "momentum":
                optimizer1 = tf.train.MomentumOptimizer(trainingrate, 0.9)
                optimizer2 = tf.train.MomentumOptimizer(trainingrate * 0.02, 0.9)
            else:
                optimizer1 = tf.train.GradientDescentOptimizer(trainingrate)
                optimizer2 = tf.train.GradientDescentOptimizer(trainingrate * 0.02)
            grads = tf.gradients(self.loss, hidden_var + known_var)
            grad1 = grads[:len(hidden_var)]
            grad2 = grads[len(hidden_var):]
            train_op1 = optimizer1.apply_gradients(zip(grad1, hidden_var))
            train_op2 = optimizer2.apply_gradients(zip(grad2, known_var))
            self.opt = tf.group(train_op1, train_op2)   
        #self.opt = optimizer.minimize(self.loss)
