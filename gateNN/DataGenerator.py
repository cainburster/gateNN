import tensorflow as tf
import numpy as np


class DataGenerator:
    def __init__(self, connection, nodeName, method="all"):
        self.method = method
        assert self.method in ["all", "random", "sample", "other"] # other is not implemented yet
        self.L = len(nodeName["input"])
        self._embedding(connection, nodeName)
        if self.method == "all":
            self.pre_input, self.pre_output = self.generate_all_pattern()
        if self.method == "sample":
            self.candidate_bag = {} # {(in1,in2,...):(out1,out2,...),...}

    def _embedding(self, connection, nodeName):
        self.nodes = {}
        
        inputnumber = len(nodeName['input'])
        outputnumber = len(nodeName['output'])
        with tf.name_scope('true_network'):
            self.inputs = tf.placeholder(tf.int32, shape = (None, inputnumber), name = 'ins')
        
            for index, name in enumerate(nodeName['input']):
                self.nodes[name] = self.inputs[:,index]
            for item in connection:
                in1, in2, out, gate_type = item
                if gate_type == "ZERO":
                    self.nodes[out] = self.nodes[in1] * 0
                elif gate_type == "ONE":
                    self.nodes[out] = self.nodes[in1] * 0 + 1
                    #raise TypeError("type of gate cannot be embedded")
                elif gate_type == "BUF":
                    self.nodes[out] = self.nodes[in1]
                elif gate_type == "NOT":
                    self.nodes[out] = 1 - self.nodes[in1]
                elif gate_type == "XOR":
                    self.nodes[out] = self.nodes[in1] + self.nodes[in2] - 2 * self.nodes[in1] * self.nodes[in2]
                elif gate_type == "XNOR":
                    self.nodes[out] = 2 * self.nodes[in1] * self.nodes[in2] - self.nodes[in1] - self.nodes[in2] + 1
                elif gate_type == "AND":
                    self.nodes[out] = self.nodes[in1] * self.nodes[in2]
                elif gate_type == "NAND":
                    self.nodes[out] = 1 - self.nodes[in1] * self.nodes[in2]
                elif gate_type == "AND2_NP":
                    self.nodes[out] = (1 - self.nodes[in1]) * self.nodes[in2]
                elif gate_type == "NAND2_NP":
                    self.nodes[out] = 1 - (1 - self.nodes[in1]) * self.nodes[in2]
                elif gate_type == "AND2_PN":
                    self.nodes[out] = self.nodes[in1] * (1 - self.nodes[in2])
                elif gate_type == "NAND2_PN":
                    self.nodes[out] = 1 - self.nodes[in1] * (1 - self.nodes[in2])        
                elif gate_type == "OR":
                    self.nodes[out] = self.nodes[in1] + self.nodes[in2] - self.nodes[in1] * self.nodes[in2]
                elif gate_type == "NOR":
                    self.nodes[out] = (1 - self.nodes[in1]) * (1 - self.nodes[in2])
                else:
                    raise NotImplementedError("shall not happen")
            self.outputs = tf.stack([self.nodes[name] for name in nodeName['output']], axis = 1)
    
    def _forward(self, inputs, max_batch_size = 64):
        inputs = np.array(inputs).astype(np.int8)
        sample_num = inputs.shape[0]
        batch_size = min(max_batch_size, sample_num)
        batch_number = sample_num // batch_size
        outputs = np.array([])
        with tf.Session() as sess:
            for bn in range(batch_number):
                ins = inputs[bn*batch_size:(bn+1)*batch_size]
                outs = sess.run(self.outputs, feed_dict = {self.inputs: ins})  
                if outputs.size:
                    outputs = np.concatenate((outputs, outs), axis = 0)
                else:
                    outputs = outs
        return outputs

    def generate_all_pattern(self, max_size = 2**18, zero_low = False):
        L = self.L
        N = 2 ** L
        f = "{:0>" + str(L) + "}"
        func = lambda i: list(map(int, f.format(str(bin(i))[2:])))
        
        if N > max_size:
            raise Exception("Generating %d patterns takes too many resources! Please choose a random generating way.")
            
        index_list = list(range(N))
            #np.random.shuffle(index_list)
        inputs = np.array([func(i) for i in index_list])
        #inputs = np.array([func(i) for i in np.random.choice(N, max_size, replace=True)])
        outputs = self._forward(inputs)
        if not zero_low:
            inputs = np.where(inputs==0, -1, inputs)
            outputs = np.where(outputs==0, -1, outputs)
        return inputs, outputs
    
    def generate_random_pattern(self, total_size=2**16, zero_low = False):
        L = self.L
        N = 2 ** L
        f = "{:0>" + str(L) + "}"
        func = lambda i: list(map(int, f.format(str(bin(i))[2:])))
        inputs = np.array([func(i) for i in np.random.choice(N, total_size)])
        outputs = self._forward(inputs)
        if not zero_low:
            inputs = np.where(inputs==0, -1, inputs)
            outputs = np.where(outputs==0, -1, outputs)
        return inputs, outputs
    
    def generator(self, total_size=None):
        # total_size is valid only if random method is chosen
        if self.method == "all":
            return self.pre_input, self.pre_output
        if self.method == "random":
            return self.generate_random_pattern(total_size)
        if self.method == "sample":
            if not self.candidate_bag:
                raise Exception("The sample set is empty! Cannot generate data!")
            inputs = np.stack([list(key) for key in self.candidate_bag.keys()], axis = 0)
            outputs = np.stack([list(self.candidate_bag[key]) for key in self.candidate_bag.keys()], axis = 0)
            return inputs, outputs
        
    def add_sample(self, candidate): # iterative variable, 1-d
        candidate = list(candidate)
        o = self._forward([candidate])[0]
        in_ = [2*c-1 for c in candidate]
        out_ = [2*c-1 for c in o]
        self.candidate_bag[tuple(in_)] = tuple(out_)
        return
    
    def reset_sample(self):
        self.candidate_bag = {}