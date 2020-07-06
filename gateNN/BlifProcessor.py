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
                if t == "BUF1" or t == "NOT1":
                    fp.write(".names %s %s\n" % (in1, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t[:3]])
                elif t == "BUF2" or t == "NOT2":
                    fp.write(".names %s %s\n" % (in2, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t[:3]])
                else:                
                    fp.write(".names %s %s %s\n" % (in1, in2, out))
                    fp.write("%s\n" % BlifWriter.gate_format[t])