import time
import numpy as np


class BlifReader:     
    hidMax = 20
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
        self.inputNumber = len(self.nodeName['input'])
        self.outputNumber = len(self.nodeName['output'])
        if showState:
            self.showNode()

    def _gate_type_distinguish(self,stype):
        stype = stype.rstrip()
        if stype in BlifReader.gate_lib:
            return BlifReader.gate_lib[stype]
        else:
            raise Exception("invalid gate string type: "+ stype)

    def _readNode(self):
        with open(self.inputFile, 'r') as fp:
            line = fp.readline()
            while not line.startswith(".model"):
                line = fp.readline()
            self.modelName = line.rsplit()[1]
            
            line = fp.readline()
            lineReady = False
            while line != "":
                if line.startswith(".inputs"):
                    names = []
                    cur_name = line.rsplit()[1:]
                    while cur_name[-1] == "\\":
                        names.extend(cur_name[:-1])
                        line = fp.readline()
                        cur_name = line.rsplit()
                    names.extend(cur_name)
                    self.nodeName['input'] = names.copy()
                elif line.startswith(".outputs"):
                    names = []
                    cur_name = line.rsplit()[1:]
                    while cur_name[-1] == "\\":
                        names.extend(cur_name[:-1])
                        line = fp.readline()
                        cur_name = line.rsplit()
                    names.extend(cur_name)
                    self.nodeName['output'] = names.copy()
                elif line.startswith(".names"):
                    self.gatesNumber += 1
                    nodes = line.rsplit()[1:]
                    if len(nodes) == 2:
                        in1, out = nodes # has to be NOT or BUF
                        #assert in1 in self.nodeName['input'] or in1 in self.nodeName['hidden'] or in1 in self.nodeName['output']
                        if out not in self.nodeName['output']:
                            self.nodeName['hidden'].append(out)
                        gate_type = self._gate_type_distinguish(fp.readline())
                        self.connection.append((in1, None, out, gate_type))
                    elif len(nodes) == 3:
                        in1, in2, out = nodes
                        #assert in1 in self.nodeName['input'] or in1 in self.nodeName['hidden'] or in1 in self.nodeName['output']
                        #assert in2 in self.nodeName['input'] or in2 in self.nodeName['hidden'] or in2 in self.nodeName['output']
                        if out not in self.nodeName['output']:
                            self.nodeName['hidden'].append(out)
                        fun_line1 = fp.readline()
                        line = fp.readline()
                        if line.startswith("."):
                            gate_type = self._gate_type_distinguish(fun_line1)
                            lineReady = True
                        else:
                            gate_type = self._gate_type_distinguish(fun_line1 + line)
                        self.connection.append((in1, in2, out, gate_type))
                    else:
                        raise Exception("can not show more than 3 nodes")
                elif line.startswith(".end"):
                    break
                else:
                    raise Exception("[ERROR] strange line inside the file: " + line)
                if not lineReady:
                    line = fp.readline()
                lineReady = False
           
    
    def showNode(self):
        print('Model name: {}'.format(self.modelName))
        print('Input Nodes({}): {}'.format(len(self.nodeName['input']), ', '.join(self.nodeName['input'])))
        hiddenlen = len(self.nodeName['hidden'])
        hiddenShow = self.nodeName['hidden'] if hiddenlen<BlifReader.hidMax else np.append(self.nodeName['hidden'][:BlifReader.hidMax], '...')
        print('Hidden Nodes({}): {}'.format(hiddenlen, ', '.join(hiddenShow)))
        print('Output Nodes({}): {}'.format(len(self.nodeName['output']), ', '.join(self.nodeName['output'])))
        print('Gates number: {}'.format(self.gatesNumber))
        
        

class BlifWriter:
    gate_format = {
        "ZERO" : "0",
        "ONE" : "1",
        "BUF" : "1 1",
        "NOT" : "1 0",
        "AND" : "11 1",
        "NAND" : "11 0",
        "AND2_NP" : "01 1",
        "NAND2_NP" : "01 0",
        "AND2_PN" : "10 1",
        "NAND2_PN" : "10 0",
        "OR" : "00 0",
        "NOR" : "00 1",
        "XOR" : "10 1\n01 1",
        "XNOR" : "00 1\n11 1"
    }
    
    def __init__(self):
        pass
    
    def write(self, connection, nodeName, modelName=None, fileName="temp.blif"):
        with open(fileName, "w") as fp:
            if not modelName:
                modelName = "temp_model"
            fp.write("### Benchmark \"%s\" written by BlifWritter at %s\n" % (modelName, time.ctime()))
            fp.write(".model %s\n" % modelName)
            fp.write(".inputs %s\n" % " ".join(nodeName["input"]))
            fp.write(".outputs %s\n" % " ".join(nodeName["output"]))
            for in1, in2, out, t in connection:
                fp.write(".names %s %s %s\n" % (in1, in2, out))
                fp.write("%s\n" % BlifWriter.gate_format[t])
    
    def write_modified(self, connection, nodeName, changed_gate, modelName=None, fileName="temp.blif"):
        with open(fileName, "w") as fp:
            if not modelName:
                modelName = "temp_model"
            fp.write("### Benchmark \"%s\" written by BlifWritter at %s\n" % (modelName, time.ctime()))
            fp.write(".model %s\n" % modelName)
            fp.write(".inputs %s\n" % " ".join(nodeName["input"]))
            fp.write(".outputs %s\n" % " ".join(nodeName["output"]))
            for in1, in2, out, t in connection:
                if out in changed_gate:
                    t = changed_gate[out]
                if t == "BUF1" or t == "NOT1" or t == "BUF" or t == "NOT":
                    fp.write(".names %s %s\n" % (in1, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t[:3]])
                elif t == "BUF2" or t == "NOT2":
                    fp.write(".names %s %s\n" % (in2, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t[:3]])
                else:                
                    fp.write(".names %s %s %s\n" % (in1, in2, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t])
            fp.write(".end\n")