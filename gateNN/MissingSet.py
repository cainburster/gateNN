import numpy as np
import collections

class MissingSet:
    def __init__(self, connection, nodeName):
        self.connection = connection
        self.inputs = nodeName["input"]
        self.outputs = nodeName["output"]
        self.hiddens = nodeName["hidden"]
        self.candidates = []
    
        self.links = collections.defaultdict(list)
        self.backwards = collections.defaultdict(list)
        for in1,in2,out,g_type in self.connection:
            self.links[in1].append(out)
            self.backwards[out].append(in1)
            if in2:
                self.links[in2].append(out)
                self.backwards[out].append(in2)
            
            if g_type != "BUF1" or g_type != "BUF2" or g_type != "NOT1" or g_type != "NOT2":
                self.candidates.append(out)
        
    def fan_out(self, gate):
        if gate in self.outputs:
            return {gate}
        queue = collections.deque()
        queue.appendleft(gate)
        result = set()
        while queue:
            g = queue.pop()
            if g in result:
                continue
            result.add(g)
            for n_g in self.links[g]:
                queue.appendleft(n_g)
        return result
    
    def fan_in(self, gate): # include gate itself, include input node
        if gate in self.inputs: 
            return {gate} # make no sense
        queue = collections.deque()
        queue.appendleft(gate)
        result = set()
        while queue:
            g = queue.pop()
            if g in result:
                continue
            result.add(g)
            for n_g in self.backwards[g]:
                queue.appendleft(n_g)
        return result
    
    def fan_out_all(self, gates):
        result = set()
        for gate in gates:
            result = result.union(self.fan_out(gate))
        return result
        
    def createVacantBoard(self, number=1, gates=[]):
        if not gates:
            # decide by number
            assert len(self.candidates) >= number
            gates = np.random.choice(self.candidates, number, replace = False)
        else:
            # decide by gates
            assert all(n in self.candidates for n in gates)
        
        print("[INFO] Gates %s are chosen!" % ', '.join(gates))
        
        fan_out_gates = self.fan_out_all(gates)
        table = []
        for in1, in2, out, g_type in self.connection:
            if out in gates:
                table.append((True, (in1,in2,out), g_type))
            else:
                if out in fan_out_gates:
                    table.append(("train", (in1,in2,out), g_type))
                else:
                    table.append((False, (in1,in2,out), g_type))         
        return gates, table
    