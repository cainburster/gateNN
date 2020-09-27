gateNN
===========================

Partial logic synthesis with neural method


Dependency
---------------------------
* abc
* Python3.7
* python libaray
    - numpy
    - tensorflow==1.15
    - sklearn

Install of abc tool
----------------------------
[abc](github.com/berkeley-abc/abc) is a library designed for solving sequential logic synthesis and formal verification. Take two steps to download abc and compile the library.

```
git clone https://github.com/berkeley-abc/abc.git
make
```

Usage of the library
----------------------------

Execute an experiment using command like this:

```
python main.py [setting.json]
```

A example of the setting file can be found in `setting` folder. Here we specify the detailed functionality of each meta element inside.

 | meta element name | function |
 | --- | --- |
 | blif_file_path | original circuit design file with BLIF format | 
 | result_folder |  directory of output result |
 | abc_path |  path of abc library |
 | vacant_gate |  number of vacant gates for this experiment |
 | fan_out_training |  whether the fan-out path trace is partially training |
 | model_type |  model type of gate simulation, accepting `kernel` and `normal` |
 | has_bias |  whether the simulation model has bisa |
 | loader_method |  method of data loading, accepting `all`, `random` and `sample` |
 | training_rate |  learning rate for backpropagation |
 | optimizer |  type of optimizer, accepting `sgd` and `momentum` |
 | data_per_epoch | the number of data for each epoch |
 | batch_size | number of batch size |
 | time_limit | 60 |

As usual, the kernel model with counterexample training method works the best, run command 
```
python main.py setting/sample_training_example.json`
```
to get the quick demo. 

Limitation
-----------------
Current version of methods only take consideration of aig or xaig circuit in BLIF format. All the components of circuit shall be a two-level gate.
