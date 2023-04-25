# Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees

This repository includes the implementation of our IJCAI 2023 paper, "Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees".

## Dependencies

- [Python](https://www.python.org/): (>=3.6)
- [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` (```ltl2ldra``` is optional)
- [PRISM](https://www.prismmodelchecker.org/): (>=4.7), ```prism``` must be in ```PATH```  
- [NumPy](https://numpy.org/): (>=1.16)
- [Matplotlib](https://matplotlib.org/): (>=3.03)

## Basic Usage

This package consists of the ```LearningAlgo``` class which contains all the core RL algorithms, the ```MDP``` class which constructs the MDP environment with predefined structure and labels, the ```OmegaAutomaton``` class that transforms LTL specifications into LDBAs and the ```PRISM``` class that builds the PRISM model of the MDP and the current policy and evaluate it using PRISM.

To run experiments, use main.py.

```shell
$ python main.py --task 'frozen_lake' --repeats 10 --save_log
```

For main.py, the --task option is required and can be used to choose the task from the following: 'probabilistic_gate', 'frozen_lake' and 'office_world'. The --repeats option selects how many repeats of the experiment to perform for plotting the graph, the default is 10. The --save_log flag saves the training log after running the experiments.
