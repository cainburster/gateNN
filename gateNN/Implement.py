import sys
import os
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

from gateNN.BlifProcessor import BlifReader, BlifWriter
from gateNN.DataGenerator import DataGenerator
from gateNN.MissingSet import MissingSet
from gateNN.model import DAGmodel

class Gate:
    gate_types = ["LOW", "AND", "AND2_PN", "BUF1", 
                  "AND2_NP", "BUF2", "XOR", "OR",
                  "NOR", "XNOR", "NOT2", "NAND2_NP",
                  "NOT1", "NAND2_PN", "NAND", "HIGH"]
    inputs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    inputs_k = np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
    
    def __init__(self, name, gate_type, has_bias=True):
        self.name = name
        self.type = gate_type
        assert self.type in self.gate_types
        self.has_bias = has_bias
    
    def add_weight(self, weight_var):
        self.weight = weight_var
    
    def add_bias(self, bias_var):
        if not self.has_bias:
            raise Exception("gate %s shall have no bias" % self.name)
        self.bias = bias_var
    
    def judge(self, sess, savedfile=sys.stdout):
        w = self.weight.eval(session=sess)
        if w.shape[0] == 3:
            ans = np.dot(self.inputs_k, w)
        else:
            ans = np.dot(self.inputs, w)
        if self.has_bias:
            b = self.bias.eval(session=sess)
            ans += b
        ans = np.sign(ans)
        res = 0
        for val in ans:
            res <<= 1
            res += 1 if val==1 else 0
        return self.gate_types[res]
        
    def isFunctionallyEqual(self, sess):
        judged_type = self.judge(sess)
        return judged_type == self.type
    
class Implement:
    
    def __init__(self, blif_file_path, 
                       result_folder,
                       abc_path,
                       vacant_gate=1, 
                       model_type="kernel", 
                       fan_out_training=True, 
                       loader_method="random",
                       data_per_epoch=4096,
                       batch_size=64,
                       training_rate=0.05,
                       optimizer="momentum",
                       maximum_epoch=100,
                       has_bias=True):
        
        self.blif_file_path = os.path.abspath(blif_file_path)
        self.result_folder = os.path.abspath(result_folder)
        self.abc_path = os.path.abspath(abc_path)
        self.vacant_gate = vacant_gate 
        self.model_type = model_type
        self.fan_out_training = fan_out_training 
        self.loader_method = loader_method
        self.data_per_epoch = data_per_epoch
        self.batch_size = batch_size
        self.training_rate = training_rate
        self.optimizer = optimizer
        self.maximum_epoch = maximum_epoch
        self.has_bias = has_bias
        
        self.reader = BlifReader(blif_file_path)
        self.reader.showNode()
        self.writer = BlifWriter()
        self.data_loader = DataGenerator(self.reader.connection, self.reader.nodeName, method = loader_method)
        #dg2.generator()
        self.settler = MissingSet(self.reader.connection, self.reader.nodeName)
        candidates, conn_info = self.settler.createVacantBoard(vacant_gate, ["new_n377_", "new_n310_", "new_n314_", "new_n426_", "new_n296_", "new_n204_", "new_n272_", "new_n141_", "new_n443_", "new_n348_"])
        
        # output connection with weight of missing gates
        self.fan_in_respond = [set() for i in range(len(self.reader.nodeName["output"]))]
        for i, gate in enumerate(self.reader.nodeName["output"]):
            s = self.settler.fan_in(gate)
            for c in candidates:
                if c in s:
                    self.fan_in_respond[i].add(c)
        
        self.model = DAGmodel(input_names=self.reader.nodeName['input'], 
                              connections=conn_info, 
                              output_names=self.reader.nodeName['output'],
                              model_type=model_type,
                              fan_out_training=fan_out_training, 
                              training_rate=training_rate,
                              optimizer=self.optimizer,
                              has_bias=self.has_bias)
        
        self.train_gates = {}
        self.fan_out_gates = {}
        for trainable, gates, gate_type in conn_info:
            in1,in2,out = gates
            if trainable == True:
                self.train_gates[out] = Gate(out, gate_type, self.has_bias)
            elif trainable == "train":
                self.fan_out_gates[out] = Gate(out, gate_type, self.has_bias)
    
    
    def run(self):
        os.makedirs(self.result_folder, exist_ok=True)
        log_file = os.path.join(self.result_folder, "log.txt")
        with open(log_file, "a") as fp:
            if self.loader_method == "sample":
                flag = self.training_sample(savedfile=fp)
            else:
                flag = self.training(self.data_per_epoch, self.maximum_epoch, self.batch_size, savedfile=fp)
        if flag:
            with open(os.path.join(self.result_folder, "success"), "w") as fp:
                pass
        else:
            with open(os.path.join(self.result_folder, "failure"), "w") as fp:
                pass
        
    def repeat_run(self, times):
        os.makedirs(self.result_folder, exist_ok=True)
        general_log = os.path.join(self.result_folder, "summary_log.txt")
        import math
        t = int(math.log10(times))+1
        f = "log-{:0>"+str(t)+"}.txt"
        cnt = set()
        for i in range(1, times+1):
            print("Experiment %d start" % i)
            log_file = os.path.join(self.result_folder, f.format(i))
            with open(log_file, "a") as fp:
                judge = self.training(self.data_per_epoch, self.maximum_epoch, self.batch_size, savedfile=fp)
            if not judge:
                cnt.add(i)
            print("Experiment %d finish" % i)
         
        with open(general_log, "w") as fp:
            fp.write("{}/{} ({:.2f}%) Success\n".format(times-len(cnt), times, (times-len(cnt))/times*100))
            fp.write("Failure case: " + str(cnt))
    
    def training(self, data_per_epoch=4096, epoch=1000, batch_size=64, savedfile=sys.stdout):      
        
        tempBlif = os.path.join(self.result_folder, "tempblif.blif")
        temp_val = os.path.join(self.result_folder, "temp_val.txt")
                 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            allvars = tf.global_variables()
            #self.predict(valid_x, valid_y, sess)
            
            for var in allvars:
                if 'Momentum' not in var.name:
                    if 'hidden' in var.name:
                        name = var.name[:var.name.find("-hidden_act")]
                        if "weight" in var.name:
                            self.train_gates[name].add_weight(var)
                        if self.has_bias and"bias" in var.name:
                            self.train_gates[name].add_bias(var)
                    if 'route' in var.name:
                        name = var.name[:var.name.find("-route_act")]
                        if name not in self.fan_out_gates:
                            continue
                        if "weight" in var.name:
                            self.fan_out_gates[name].add_weight(var)
                        if self.has_bias and "bias" in var.name:
                            self.fan_out_gates[name].add_bias(var)
            
            self.print_parameters(sess, savedfile)
            
            print('***********training process***********', file=savedfile)
            
            for i in range(epoch):
                # shuffle
                train_x, train_y = self.data_loader.generator(data_per_epoch)
                train_x, train_y = shuffle(train_x, train_y)
                batchsize = min(batch_size, train_x.shape[0])
                # make minibatch
                final_loss = 0
                BT = train_x.shape[0]//batchsize
                for bt in range(BT):
                    ins = train_x[bt*batchsize:(bt+1)*batchsize]
                    outs = train_y[bt*batchsize:(bt+1)*batchsize]
                    _, loss_value = sess.run((self.model.opt, self.model.loss), 
                                             feed_dict = {self.model.trueInputs: ins, self.model.trueOut: outs})  
                    final_loss += loss_value
                
                final_loss = final_loss/BT
                print('[epoch %5d] temporary losses: %.8f' % (i+1, final_loss), file=savedfile)
                if final_loss == 0:
                    # check if fan_out gates have changed
                    if self.fan_out_training:
                        for name in self.fan_out_gates.keys():
                            if not self.fan_out_gates[name].isFunctionallyEqual(sess):
                                print("[Failure]fan-out gate %s got changed!" % name, file=savedfile)
                                self.print_parameters(sess, savedfile)
                                return False # a failure training can be judged from this point
                        # all fan_out gates remain the same functionality
                    
                    # check training gates fulfill the satisfaction
                    cur_result = {}
                    for name in self.train_gates.keys():
                        cur_result[name] = self.train_gates[name].judge(sess)

                    self.writer.write_modified(self.reader.connection, 
                                               self.reader.nodeName, 
                                               cur_result,
                                               modelName="temp", 
                                               fileName=tempBlif)
                    os.system('cd {}; ./abc -c \"cec {} {}\" > {}'.format(self.abc_path, self.blif_file_path, tempBlif, temp_val))
                    with open(temp_val) as fp:
                        contents = fp.read()
                        if "Networks are equivalent" in contents:
                            self.print_parameters(sess, savedfile)
                            print('***********result***********', file=savedfile)
                            print("result " + str(cur_result), file=savedfile)  
                            print("[Success] training success!", file=savedfile)
                            return True
                        elif "Networks are NOT EQUIVALENT" in contents:
                            print("[CEC NOT EQUIVALENT]satisfactary condition has not been achieved", file=savedfile)
            self.print_parameters(sess, savedfile) 
            print("[Failure] Reached maximum epochs", file=savedfile)
            return False

    
    def training_sample(self, savedfile=sys.stdout):      
        
        max_iteration = 1000
        max_epoch = 2000
        chance_of_reset_gate = 20
        self.data_loader.reset_sample()
        tempBlif = os.path.join(self.result_folder, "tempblif.blif")
        temp_val = os.path.join(self.result_folder, "temp_val.txt")
                 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            allvars = tf.global_variables()
            #self.predict(valid_x, valid_y, sess)
            
            for var in allvars:
                if 'Momentum' not in var.name:
                    if 'hidden' in var.name:
                        name = var.name[:var.name.find("-hidden_act")]
                        if "weight" in var.name:
                            self.train_gates[name].add_weight(var)
                        if self.has_bias and"bias" in var.name:
                            self.train_gates[name].add_bias(var)
                    if 'route' in var.name:
                        name = var.name[:var.name.find("-route_act")]
                        if name not in self.fan_out_gates:
                            continue
                        if "weight" in var.name:
                            self.fan_out_gates[name].add_weight(var)
                        if self.has_bias and "bias" in var.name:
                            self.fan_out_gates[name].add_bias(var)
            
            # make sure weight has not BUF and NOT
            for name in self.train_gates.keys():
                w = self.train_gates[name].weight
                while True:
                    w_s = w.eval()
                    if abs(w_s[0][0]) < abs(w_s[1][0]) + abs(w_s[2][0]) and abs(w_s[1][0]) < abs(w_s[0][0]) + abs(w_s[2][0]):
                        break
                    sess.run(tf.variables_initializer([w]))   
            
            self.print_parameters(sess, savedfile)
            
            print('***********training process***********', file=savedfile)
            # failure case: fan out gates changed || time out
            # success case: fan out not changed && cec satisfied
            
            for i in range(1, max_iteration+1):
                # check fan_out remain the same
                if self.fan_out_training:
                    for name in self.fan_out_gates.keys():
                        if not self.fan_out_gates[name].isFunctionallyEqual(sess):
                            print("[Failure] fan-out gate %s got changed!" % name, file=savedfile)
                            return False 
                # get functionality of missing gates
                cur_result = {}
                for name in self.train_gates.keys():
                    cur_result[name] = self.train_gates[name].judge(sess)
                self.writer.write_modified(self.reader.connection, 
                                           self.reader.nodeName, 
                                           cur_result,
                                           modelName="temp", 
                                           fileName=tempBlif)
                os.system('cd {}; ./abc -c \"cec {} {}\" > {}'.format(self.abc_path, self.blif_file_path, tempBlif, temp_val))
                with open(temp_val) as fp:
                    contents = fp.read()
                    if "Networks are equivalent" in contents:
                        self.print_parameters(sess, savedfile)
                        print('***********result***********', file=savedfile)
                        print("result " + str(cur_result), file=savedfile)  
                        print("[Success] training success!", file=savedfile)
                        return True
                    elif "Networks are NOT EQUIVALENT" in contents:
                        # add sample to training data
                        input_string = contents[contents.find("Input pattern:")+14:]
                        back_space = input_string.find("\n")
                        if back_space == -1:
                            input_string = input_string.rsplit()
                        else:
                            input_string = input_string[:back_space].rsplit()
                        input_table = {name: 0 for name in self.reader.nodeName["input"]}
                        for line in input_string:
                            name, t = line.split("=")
                            input_table[name] = int(t)
                        #print(input_table)
                        candidate = [input_table[name] for name in self.reader.nodeName["input"]]
                        print(candidate)
                        self.data_loader.add_sample(candidate)
                
                # training on small set
                def training_part(x,y, max_epoch):
                    batch = min(4, x.shape[0])
                    wrong = [x.shape[0]-1]
                    for epoch in range(1, max_epoch+1):
                        x, y = shuffle(x, y)
                        
                        BT = train_x.shape[0]//batch
                        for bt in range(BT):
                            ins = x[bt*batch:(bt+1)*batch]
                            outs = y[bt*batch:(bt+1)*batch]
                            _, loss_value = sess.run((self.model.opt, self.model.loss), 
                                            feed_dict = {self.model.trueInputs: ins, self.model.trueOut: outs})  
                            #final_loss += loss_value
                        
                        
                        predict, loss_value = sess.run((self.model.prediction, self.model.loss),
                                    feed_dict = {self.model.trueInputs: x, self.model.trueOut: y})
                        if loss_value == 0:
                            print("[iteration %3d][epoch %4d] find a new candidate" % (i, epoch), file=savedfile)
                            return True
                        # wrong = np.nonzero(np.any(predict!=y, axis=1))[0]
                    print("[Failure] [iteration %3d][epoch %4d] [loss %.3f] reach maximum epoch, cannot find candidate!" % (i, epoch, loss_value), file=savedfile)
                    return False
                    
                train_x, train_y = self.data_loader.generator()
                for chance in range(chance_of_reset_gate+1):
                    if training_part(train_x, train_y, max_epoch):
                        #self.print_parameters(sess,savedfile)
                        break
                    else:
                        self.print_parameters(sess, savedfile) 
                        # give a chance, make sure weight has not BUF and NOT
                        if chance == chance_of_reset_gate:
                            print("[Failure] cannot find candidate after repetitive trials")
                            self.print_parameters(sess, savedfile) 
                            return False
                        # finding which gates may got problem, from fan_in of wrong predition
                        predict = sess.run((self.model.prediction), feed_dict = {self.model.trueInputs: train_x})
                        print("pridiction: {}, true: {}".format(predict,train_y))
                        wrong_output = np.nonzero(np.any(predict != train_y, axis = 0))[0]
                        print("wrong output: " + ",".join([self.reader.nodeName['output'][w] for w in wrong_output]))
                        correspond_gates = set()
                        for o in wrong_output:
                            correspond_gates = correspond_gates.union(self.fan_in_respond[o])
                        # reset those gates
                        # changing_gate = set()
                        for name in correspond_gates:
                            w = self.train_gates[name].weight
                            while True:
                                sess.run(tf.variables_initializer([w]))
                                w_s = w.eval()
                                if abs(w_s[0][0]) < abs(w_s[1][0]) + abs(w_s[2][0]) and abs(w_s[1][0]) < abs(w_s[0][0]) + abs(w_s[2][0]):
                                    break
#                                 changing_gate.add(name)
                        print("[Reset] {} is reset.".format(correspond_gates), file=savedfile)
            
            self.print_parameters(sess, savedfile) 
            print("[Failure] Reached maximum iteration", file=savedfile)
            return False        
        
        
    def print_parameters(self, sess, savedfile=sys.stdout):
        print('***********weight update***********', file=savedfile)
        if self.has_bias:
            for gate in self.train_gates.values():
                print("Trainable Gate: {}, initial type: {}, current type: {}, weight: {}, bias: {}".format(
                                        gate.name, 
                                        gate.type,
                                        gate.judge(sess),
                                        gate.weight.eval(session=sess),
                                        gate.bias.eval(session=sess)),
                      file=savedfile)
            if self.fan_out_training:
                for gate in self.fan_out_gates.values():
                    print("Trainable Gate: {}, initial type: {}, current type: {}, weight: {}, bias: {}".format(
                                            gate.name, 
                                            gate.type,
                                            gate.judge(sess),
                                            gate.weight.eval(session=sess),
                                            gate.bias.eval(session=sess)),
                          file=savedfile)
        else:
            for gate in self.train_gates.values():
                print("Trainable Gate: {}, initial type: {}, current type: {}, weight: {}".format(
                                        gate.name, 
                                        gate.type,
                                        gate.judge(sess),
                                        gate.weight.eval(session=sess)),
                      file=savedfile)
            if self.fan_out_training:
                for gate in self.fan_out_gates.values():
                    print("Trainable Gate: {}, initial type: {}, current type: {}, weight: {}".format(
                                            gate.name, 
                                            gate.type,
                                            gate.judge(sess),
                                            gate.weight.eval(session=sess)),
                          file=savedfile)
            