import collections
import numpy as np
import tensorflow as tf

class Node:
    def __init__(self, name):
        self.name = name
        self.former = []
        self.next = []

    def __str__(self):
        return "name: {}, former: {}, next: {}".format(self.name, self.former, self.next)

    def add_former(self, name):
        if name not in self.former:
            self.former.append(name)

    def add_next(self, name):
        if name not in self.next:
            self.next.append(name)
    
    def is_head(self):
        if len(self.former) == 0:
            return True
        return False
    
    def is_tail(self):
        if len(self.next) == 0:
            return True
        return False

class BlifReader:     
    hidMax = 5
    gate_lib = {
        "0" : "ZERO",
        "1" : "ONE",
        "1 1" : "BUF",
        "1 0" : "NOT",
        "0 1" : "NOT",
        "11 1" : "AND",
        "11 0" : "NAND",
        "01 1" : "AND2_NP",
        "01 0" : "NAND2_NP",
        "10 1" : "AND2_PN",
        "10 0" : "NAND2_PN",
        "00 0" : "OR",
        "00 1" : "NOR",
        "10 1\n01 1" : "XOR",
        "01 1\n10 1" : "XOR",
        "11 0\n00 0" : "XOR",
        "00 0\n11 0" : "XOR",
        "00 1\n11 1" : "XNOR",
        "11 1\n00 1" : "XNOR",
        "10 0\n01 0" : "XNOR",
        "01 0\n10 0" : "XNOR"
    }
    def __init__(self, path, showState = False):
        self.inputFile = path
        self.nodeName = {'input': [],
                 'hidden': [],
                 'output': []}
        self.connection = []
        self.gatesNumber = 0
        self._readNode()
        self.DLT = self._make_doubly_link_table()
        self.inputNumber = len(self.nodeName['input'])
        self.outputNumber = len(self.nodeName['output'])
        if showState:
            self.showNode()

    def _gate_type_distinguish(self,stype):
        stype = stype.rstrip()
        if stype in BlifReader.gate_lib:
            return BlifReader.gate_lib[stype]
        else:
            raise Exception("invalid gate string type")

    def _readNode(self):
        with open(self.inputFile, 'r') as fp:
            fp.readline() # skip comment
            
            name_line = fp.readline()
            assert name_line.startswith(".model")
            self.modelName = name_line[7:].strip()
            
            input_line = fp.readline()
            assert input_line.startswith('.inputs')
            self.nodeName['input']=input_line[8:].split()
            
            output_line = fp.readline()
            assert output_line.startswith('.outputs')
            self.nodeName['output']=output_line[9:].split()
            
            remain = fp.read()
            index = remain.find(".end")
            gate_lines = remain[:index].split(".names ")[1:]
            self.gatesNumber = len(gate_lines)
            for gate_info in gate_lines:
                gate_info = gate_info.rstrip()
                space_pos = gate_info.find("\n")
                names = gate_info[:space_pos].split()
                gate_str = gate_info[space_pos + 1:]
                gate_type = self._gate_type_distinguish(gate_str)

                if len(names) == 3:
                    in1, in2, out = names
                    assert in1 in self.nodeName['input'] or in1 in self.nodeName['hidden'] or in1 in self.nodeName['output']
                    assert in2 in self.nodeName['input'] or in2 in self.nodeName['hidden'] or in2 in self.nodeName['output']
                    if out not in self.nodeName['output']:
                        self.nodeName['hidden'].append(out)
                    self.connection.append((in1, in2, out, gate_type))
                elif len(names) == 2:
                    in1, out = names # has to be NOT or BUF
                    assert in1 in self.nodeName['input'] or in1 in self.nodeName['hidden'] or in1 in self.nodeName['output']
                    if out not in self.nodeName['output']:
                        self.nodeName['hidden'].append(out)
                    self.connection.append((in1, None, out, gate_type))
        return 

    def _make_doubly_link_table(self):
        table = {}
        for name in self.nodeName['input']:
            table[name] = Node(name)
        for name in self.nodeName['hidden']:
            table[name] = Node(name)
        for name in self.nodeName['output']:
            table[name] = Node(name)
        for meta in self.connection:
            in1, in2, out, _ = meta
            table[in1].add_next(out)
            table[out].add_former(in1)
            if in2:
                table[in2].add_next(out)
                table[out].add_former(in2)
        return table
    
    def showNode(self):
        print('Model name: {}'.format(self.modelName))
        print('Input Nodes({}): {}'.format(len(self.nodeName['input']), ', '.join(self.nodeName['input'])))
        hiddenlen = len(self.nodeName['hidden'])
        hiddenShow = self.nodeName['hidden'] if hiddenlen<BlifReader.hidMax else np.append(np.random.choice(self.nodeName['hidden'], BlifReader.hidMax), '...')
        print('Hidden Nodes({}): {}'.format(hiddenlen, ', '.join(hiddenShow)))
        print('Output Nodes({}): {}'.format(len(self.nodeName['output']), ', '.join(self.nodeName['output'])))
        print('Gates number: {}'.format(self.gatesNumber))
        
    def embedding(self):
        self.nodes = {}
        
        inputnumber = len(self.nodeName['input'])
        outputnumber = len(self.nodeName['output'])
        with tf.name_scope('true_network'):
            self.inputs = tf.placeholder(tf.int8, shape = (None, inputnumber), name = 'ins')
        
            for index, name in enumerate(self.nodeName['input']):
                self.nodes[name] = self.inputs[:,index]
            for item in self.connection:
                in1, in2, out, gate_type = item
                if gate_type == "ZERO" or gate_type == "ONE":
                    raise TypeError("type of gate cannot be embedded")
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
            self.outputs = tf.stack([self.nodes[name] for name in self.nodeName['output']], axis = 1)
    
    def forward(self, inputs, max_batch_size = 64):
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

    def generate_pattern(self, max_size = 2**18, zero_low = True):
        L = len(self.nodeName['input'])
        N = 2 ** L
        f = "{:0>" + str(L) + "}"
        func = lambda i: list(map(int, f.format(str(bin(i))[2:])))
        if N <= max_size:
            index_list = list(range(N))
            np.random.shuffle(index_list)
            inputs = np.array([func(i) for i in index_list])
        else:
            print("[WARNING] The number of input patterns has exceeded settled maximum. The size of pattern is reduced to %d" % max_size)
            inputs = np.array([func(i) for i in np.random.choice(N, max_size, replace=True)])
        outputs = self.forward(inputs)
        if not zero_low:
            inputs = np.where(inputs==0, -1, inputs)
            outputs = np.where(outputs==0, -1, outputs)
        return inputs, outputs
    
    def set_missing_gate(self, number = 1, names = []):
        if not names:
            nodes = np.concatenate([self.nodeName['hidden'], self.nodeName['output']], axis = 0)
            names = np.random.choice(nodes, number, replace = False)
        else:
            # make sure the gate name in names is all valid
            assert all(n in self.nodeName['hidden'] or n in self.nodeName['output'] for n in names)
        
        print("[INFO] Gates %s are chosen!" % ', '.join(names))
        table = []
        for c in self.connection:
            in1, in2, out, g_type = c
            if out in names:
                table.append([True, [in1, in2, out], g_type])
            else:
                table.append([False, [in1, in2, out], g_type])
        return table, names
    
    def fan_in_single(self, gate):
        assert gate in self.DLT    
        
        not_counter = set(self.DLT.keys())
        queue = collections.deque()
        queue.appendleft(gate)
        result = []
        while queue:
            node = self.DLT[queue.pop()]
            if node.is_head():
                if node.name not in result:
                    result.append(node.name)
            else:
                for f_node in node.former:
                    if f_node not in queue and f_node in not_counter:
                        queue.appendleft(f_node)                    
            not_counter.remove(node.name)
        return result
    
    def fan_in(self, gates):
        result = set()
        for gate in gates:
            result = result.union(set(self.fan_in_single(gate)))
        return list(result)
    
    def fan_out_single(self, gate):
        assert gate in self.DLT    
        
        not_counter = set(self.DLT.keys())
        queue = collections.deque()
        queue.appendleft(gate)
        result = []
        while queue:
            node = self.DLT[queue.pop()]
            if node.is_tail():
                if node.name not in result:
                    result.append(node.name)
            else:
                for f_node in node.next:
                    if f_node not in queue and f_node in not_counter:
                        queue.appendleft(f_node)                    
            not_counter.remove(node.name)
        return result
    
    def fan_out(self, gates):
        result = set()
        for gate in gates:
            result = result.union(set(self.fan_out_single(gate)))
        return list(result)   
    
    def generate_condition_pattern(self, valid_input, max_size = 2**18, zero_low = True):
        def insert_zero(string, position):
            l = list(string)
            for p in position:
                l.insert(p, 0)
            return l
        
        L = len(self.nodeName['input'])
        zero_posi = []
        for i, node in enumerate(self.nodeName['input']):
            if node not in valid_input:
                zero_posi.append(i)
        zero_posi = sorted(zero_posi)
        
        N = 2 ** len(valid_input)
        f = "{:0>" + str(len(valid_input)) + "}"
        func = lambda i: list(map(int, insert_zero(f.format(str(bin(i))[2:]), zero_posi)))
        if N <= max_size:
            index_list = list(range(N))
            np.random.shuffle(index_list)
            inputs = np.array([func(i) for i in index_list])
        else:
            print("[WARNING] The number of input patterns has exceeded settled maximum. The size of pattern is reduced to %d" % max_size)
            inputs = np.array([func(i) for i in np.random.choice(N, max_size, replace=True)])
        outputs = self.forward(inputs)
        if not zero_low:
            inputs = np.where(inputs==0, -1, inputs)
            outputs = np.where(outputs==0, -1, outputs)
        return inputs, outputs