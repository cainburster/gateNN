import sys
import os
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

from .BlifProcessor import BlifReader, BlifWriter
from .DataGenerator import DataGenerator
from .MissingSet import MissingSet
from .model import DAGmodel

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
                       maximum_epoch=100):
        
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
        
        self.reader = BlifReader(blif_file_path)
        self.reader.showNode()
        self.writer = BlifWriter()
        self.data_loader = DataGenerator(self.reader.connection, self.reader.nodeName, method = loader_method)
        #dg2.generator()
        self.settler = MissingSet(self.reader.connection, self.reader.nodeName)
        candidates, conn_info = self.settler.createVacantBoard(vacant_gate)
        
        self.model = DAGmodel(input_names=self.reader.nodeName['input'], 
                              connections=conn_info, 
                              output_names=self.reader.nodeName['output'],
                              model_type=model_type,
                              fan_out_training=fan_out_training, 
                              training_rate=training_rate,
                              optimizer=self.optimizer)
        
        self.train_gates = {}
        self.fan_out_gates = {}
        for trainable, gates, gate_type in conn_info:
            in1,in2,out = gates
            if trainable == True:
                self.train_gates[out] = Gate(out, gate_type)
            elif trainable == "train":
                self.fan_out_gates[out] = Gate(out, gate_type)
    
    
    def run(self):
        os.makedirs(self.result_folder, exist_ok=True)
        log_file = os.path.join(self.result_folder, "log.txt")
        with open(log_file, "a") as fp:
            self.training(self.data_per_epoch, self.maximum_epoch, self.batch_size, savedfile=fp)
    
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
                        if "bias" in var.name:
                            self.train_gates[name].add_bias(var)
                    if 'route' in var.name:
                        name = var.name[:var.name.find("-route_act")]
                        if name not in self.fan_out_gates:
                            continue
                        if "weight" in var.name:
                            self.fan_out_gates[name].add_weight(var)
                        if "bias" in var.name:
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
    
    def print_parameters(self, sess, savedfile=sys.stdout):
        print('***********weight update***********', file=savedfile)
        for gate in self.train_gates.values():
            print("Trainable Gate: {}, initial type: {}, weight: {}, bias: {}".format(
                                    gate.name, 
                                    gate.type,
                                    gate.weight.eval(session=sess),
                                    gate.bias.eval(session=sess)),
                  file=savedfile)
        if self.fan_out_training:
            for gate in self.fan_out_gates.values():
                print("Fan out Gate: {}, initial type: {}, weight: {}, bias: {}".format(
                                        gate.name, 
                                        gate.type,
                                        gate.weight.eval(session=sess),
                                        gate.bias.eval(session=sess)),
                      file=savedfile)