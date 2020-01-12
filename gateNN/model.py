import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


# gate simulation with relating parameter
pattern = {'AND': {'weight': [[1],[1],[1]], 'bias': [-1]}, # c = ab, 0001
           'NAND': {'weight': [[-1],[-1],[-1]], 'bias': [1]}, # c = ~(ab), 1110
           'AND2_PN': {'weight': [[1],[-1],[-1]], 'bias': [-1]}, # c = a(~b), 0010
           'NAND2_PN': {'weight': [[-1],[1],[1]], 'bias': [1]}, # c = ~(a(~b)), 1101
           'AND2_NP': {'weight': [[-1],[1],[-1]], 'bias': [-1]}, # c = (~a)b, 0100
           'NAND2_NP': {'weight': [[1],[-1],[1]], 'bias': [1]}, # c = ~(~(a)b), 1011
           'NOR': {'weight': [[-1],[-1],[1]], 'bias': [-1]}, # c = (~a)(~b), 1000
           'OR': {'weight': [[1],[1],[-1]], 'bias': [1]}, # c = ~((~a)(~b)), 0111
           'XOR': {'weight': [[0],[0],[-1]], 'bias': [0]}, # c = a^b, 0110
           'XNOR': {'weight': [[0],[0],[1]], 'bias': [0]}, # c = a|b, 1001
          }

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
    def __init__(self, input_names, connections, output_names, training_rate=0.01):
        #tf.reset_default_graph()
        self.input_nodes = input_names.copy()
        self.output_nodes = output_names.copy()
        self.build_model(connections)
        self.build_loss()
        self.build_optimizer(training_rate)
        self.build_init()
    
    def build_gates(self, trainable, namelist, connection):
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
        
        in1, in2, out = namelist
        if trainable == True:
            with tf.name_scope(out):  

                linear_model = tf.layers.Dense(units=1, 
                                        activation = train_activation, 
                                        use_bias=True, 
                                        kernel_initializer=tf.random_uniform_initializer(-1.4,1.4),
                                        #bias_initializer=tf.constant_initializer([0.5]),
                                        name='hidden_act') # cannot refuse bias
                
                affined_in = tf.stack([self.nodes[in1], self.nodes[in2], self.nodes[in1] * self.nodes[in2]], axis = 1)
                #affined_in = tf.stack([self.nodes[in1], self.nodes[in2]], axis = 1)
                self.nodes[out] = tf.reshape(linear_model(affined_in), [-1])

                
#                 linear_model = tf.layers.Dense(units=1, 
#                                             activation = train_activation,
#                                             use_bias= True, 
#                                             #kernel_initializer=tf.glorot_normal_initializer,
#                                             kernel_initializer=tf.constant_initializer([-1,1]),
#                                             #bias_initializer=tf.glorot_normal_initializer,
#                                             #bias_initializer=tf.constant_initializer([1.8]),
#                                             name='hidden_1act',) # cannot refuse bias
#                 linear_model2 = tf.layers.Dense(units=2, 
#                                              activation = tf.nn.tanh, 
#                                              use_bias= True, 
#                                              #kernel_initializer=tf.glorot_normal_initializer,
#                                              kernel_initializer = tf.random_uniform_initializer(0.1,0.2), 
#                                              #bias_initializer=tf.glorot_normal_initializer,
#                                              #bias_initializer=tf.constant_initializer([-1,1]),
#                                              name = 'hidden_2act') # cannot refuse bias
                
#                 part = linear_model2(tf.stack([self.nodes[in1], self.nodes[in2]], axis = 1))
#                 self.nodes[out] = tf.reshape(linear_model(part), [-1])
                
            self.gatesSimulation[out] = (self.nodes[in1], self.nodes[in2], self.nodes[out])
        else:
            #if trainable == 'train':
            #    train_sym = True
            #    coeff = 1.8
            if connection == 'BUF':
                self.nodes[out] = self.nodes[in1]
            elif connection == 'NOT':
                self.nodes[out] = -1 * self.nodes[in1]
            else:
                constant_linear = tf.layers.Dense(units=1, 
                                            activation = binary_activation, 
                                            use_bias=True, 
                                            kernel_initializer=tf.constant_initializer(np.array(pattern[connection]['weight']) * 0.5),
                                            bias_initializer=tf.constant_initializer(np.array(pattern[connection]['bias']) * 0.5),
                                            trainable =False) # cannot refuse bias
                with tf.name_scope(out):
                    affined_in = tf.stack([self.nodes[in1], self.nodes[in2], self.nodes[in1] * self.nodes[in2]], axis = 1)
                    self.nodes[out] = tf.reshape(constant_linear(affined_in), [-1])                


    # connections list ele: [trainable, [in1, in2, out], *connectionPattern]
    def build_model(self, connections):
        self.nodes = {}
        self.gatesSimulation = {}
        for index, name in enumerate(self.input_nodes):
            self.nodes[name] = inputs[:,index]
        for item in connections:
            trainable, namelist, connectionPattern = item
            self.build_gates(trainable, namelist, connectionPattern)
        self.prediction = tf.stack([self.nodes[name] for name in self.output_nodes], axis = 1)
        
    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.prediction - trueOut))
    
    def build_optimizer(self, trainingrate):
        optimizer = tf.train.MomentumOptimizer(trainingrate, 0.9)
        self.opt = optimizer.minimize(self.loss)
    
    def build_init(self):
        self.init = tf.global_variables_initializer()
    
    def training(self, train_x, train_y, epoch=1000, batch_size=64, savedfile=sys.stdout, validation_data=None):      
        with tf.Session() as sess:
            sess.run(self.init)
            allvars = tf.global_variables()
            if validation_data:
                valid_x, valid_y = validation_data
            else:
                valid_x = train_x
                valid_y = train_y
            #self.predict(valid_x, valid_y, sess)
            
            
            print('***********initial weight***********', file=savedfile)
            for var in allvars:
                if 'Momentum' not in var.name and 'hidden' in var.name:
                    print(var.name + str(var.eval(session=sess)), file=savedfile)
            
            print('***********training process***********', file=savedfile)
            batchsize = min(train_x.shape[0], batch_size)
            for i in range(epoch):
                # shuffle
                train_x, train_y = shuffle(train_x, train_y)
                # make minibatch
                final_loss = 0
                BT = train_x.shape[0]//batchsize
                for bt in range(BT):
                    ins = train_x[bt*batchsize:(bt+1)*batchsize]
                    outs = train_y[bt*batchsize:(bt+1)*batchsize]
                    _, loss_value = sess.run((self.opt, self.loss), feed_dict = {inputs: ins, trueOut: outs})                
                    final_loss += loss_value
                if i % 1 == 0 or final_loss == 0:
                    final_loss = final_loss/BT
                    print('[epoch %5d] temporary losses: %.8f' % (i, final_loss), file=savedfile)

                    
                    if final_loss == 0:
                        break
                    # validation
#                     acc = self.predict(valid_x, valid_y, sess, False)
#                     if final_loss < 0.002 and acc == 100:
#                         break
            
            #testing
            #acc = self.predict(data, sess, showprediction = True)
#             print('accuracy of simulation: %.3f' % acc, file=savedfile)
                        
            print('***********final weight***********', file=savedfile)
            for var in allvars: 
                if 'Momentum' not in var.name and 'hidden' in var.name:
                    print(var.name + str(var.eval(session=sess)), file=savedfile)
            
            self.unknownSimulation(valid_x, valid_y, sess, savedfile = savedfile)
        return #acc
            
    def predict(self, x, y, sess, showprediction = False):            
        if showprediction:
            print('***********prediction for original map: ***********')
        trueNum = 0
        falseNum = 0
        batchsize = min(x.shape[0], 64)
        for bt in range(x.shape[0]//batchsize):
            ins = x[bt*batchsize:(bt+1)*batchsize]
            outs = y[bt*batchsize:(bt+1)*batchsize]
            for labels in zip(outs, fun2(sess.run(self.prediction, feed_dict = {inputs: ins, trueOut:outs}))):
                if showprediction:
                    print('true: {}, prediction: {}'.format(labels[0], labels[1]))
                if np.all(labels[0] == labels[1]):
                    trueNum += 1
                else:
                    falseNum += 1
        return trueNum / (trueNum + falseNum) * 100
    
    def unknownSimulation(self, x, y, sess, savedfile=sys.stdout):
        batchsize = min(x.shape[0], 64)
        print('***********start simulating gates***************', file=savedfile)
        for pattern in self.gatesSimulation.keys():
            print('simulate %s' % pattern, file=savedfile)
            lut = {}
            for bt in range(x.shape[0]//batchsize):
                ins = x[bt*batchsize:(bt+1)*batchsize]
                outs = y[bt*batchsize:(bt+1)*batchsize]
                in1, in2, out = sess.run(self.gatesSimulation[pattern], feed_dict = {inputs: ins, trueOut:outs})
                if len(lut) !=4: # if all cases not satisfy, run minibatch case
                    for item in zip(fun2(in1), fun2(in2), fun2(out)):
                        in_lut = '%d, %d' % (item[0], item[1])
                        out_lut = '%d' % item[2]
                        if in_lut not in lut.keys():
                            lut[in_lut] = out_lut
                else:
                    break
            for key in sorted(lut.keys()):
                print("input: %s, output: %s" % (key, lut[key]), file=savedfile)
               
            
def getParser():
    parser = argparse.ArgumentParser(description = "dag model training")
   
    parser.add_argument('-i', "--blif_path", type=str, help="path to blif file", required=True)
    parser.add_argument('-o', "--log_file", type=str, help="path to log file", required=True)
    parser.add_argument('-g', "--missing_gate", type=int, help="The number of missing gate.", default = 1)
    parser.add_argument("--batch_size", type=int, help="The number of batch size.", default=64)
    parser.add_argument("--epoch", type=int, help="The number of epoch.", default=500)
    parser.add_argument("--lr", type=float, help="The number of learning rate.", default=0.02)


    args = parser.parse_args()
    return args                

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

if __name__ == '__main__':
    args = getParser()
    
    from BlifReader import BlifReader
    reader = BlifReader(args.blif_path)
    reader.embedding()
    
    with tf.name_scope('input'):
        inputs = tf.compat.v1.placeholder(tf.float32, shape = (None, reader.inputNumber), name = 'ins')
        trueOut = tf.compat.v1.placeholder(tf.float32, shape = (None, reader.outputNumber), name = 'outs')
        
    table, gates = reader.set_missing_gate(args.missing_gate)
    v_i = reader.fan_in(gates)
    x_train, y_train = reader.generate_condition_pattern(v_i, zero_low=False)
    
    model = DAGmodel(reader.nodeName['input'], table, reader.nodeName['output'], training_rate=args.lr)
    with open(args.log_file, 'w') as fp:
        model.training(x_train, y_train, epoch=args.epoch, batch_size = args.batch_size, savedfile=fp)
