# Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees

This repository includes the implementation of our IJCAI 2023 paper, "Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees".

## Abstract

Linear Temporal Logic (LTL) is widely used to specify high-level objectives for system policies, and it is highly desirable for autonomous systems to learn the optimal policy with respect to such specifications. However, learning the optimal policy with respect to LTL specifications is not trivial. We present a model-free Reinforcement Learning (RL) approach that efficiently learns an optimal policy for an unknown stochastic system, modelled using Markov Decision Processes (MDPs). We propose a novel and more general product MDP, reward structure and discounting mechanism that, when applied in conjunction with off-the-shelf model-free RL algorithms, efficiently learn the optimal policy that maximizes the probability of satisfying a given LTL specification with optimality guarantees. We also provide improved theoretical results on choosing the key parameters in RL to ensure optimality. To directly evaluate the learned policy, we adopt probabilistic model checker PRISM to compute the probability of the policy satisfying such specifications. Several experiments on various tabular MDP environments across different LTL tasks demonstrate the improved sample efficiency and optimal policy convergence.

## Citation

```
@inproceedings{a,
    title={Sample Efficient Model-free Reinforcement Learning from LTL Specifications with Optimality Guarantees},
    author={Daqian Shao and Marta Kwiatkowska},
    year={2023},
    booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
}
```

## Dependencies

The followings are the essential tools and libraries to run our RL algorithms:

- [Python](https://www.python.org/): (>=3.7)
- [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` (```ltl2ldra``` is optional)  
Download from https://www7.in.tum.de/~kretinsk/rabinizer4.html and follow instructions to add ```ltl2ldba``` to ```PATH```
- [PRISM](https://www.prismmodelchecker.org/): (>=4.7), ```prism``` must be in ```PATH```  
Download from https://www.prismmodelchecker.org/download.php and follow installation instructions.
- Python Libraries  
Create a new conda environment and run ```pip install -e requirements.txt```.

## Basic Usage

This package consists of the ```LearningAlgo``` class which contains all the core RL algorithms, the ```MDP``` class which constructs the MDP environment with predefined structure and labels, the ```OmegaAutomaton``` class that transforms LTL specifications into LDBAs and the ```PRISM``` class that builds the PRISM model of the MDP and the current policy and evaluate it using PRISM.

To run experiments, use main.py.

```shell
$ python main.py --task 'frozen_lake' --repeats 10 --save_log
```

For main.py, the --task option is required and can be used to choose the task from the following: 'probabilistic_gate', 'frozen_lake' and 'office_world'. The --repeats option selects how many repeats of the experiment to perform for plotting the graph, the default is 10. The --save_log flag saves the training log after running the experiments.
