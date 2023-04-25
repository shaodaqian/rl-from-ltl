from algo.mdp import MDP
from algo.automaton import OmegaAutomaton
from algo.core import LearningAlgo
from algo.environments.grid_map import GridMaps
from algo.PRISM import PRISMModel

import numpy as np
import seaborn as sns
import sys
# from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import argparse

import time
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
matplotlib.rcParams.update({'font.size': 16})

np.set_printoptions(threshold=sys.maxsize)
np.random.seed(999)


def plot_probs(probss,evaluation_frequency,save_log):
    """
    Plots the performance graph with the satisfaction probabilities
    """
    fig, ax = plt.subplots(figsize=(8,6))
    plt.xlabel("Training steps in thousands", fontsize=16)
    plt.ylabel("Satisfaction Probability", fontsize=16)
    clrs = sns.color_palette("husl", len(probss))
    # clrs = sns.color_palette()

    with sns.axes_style("darkgrid"):
        temp_max=0
        for i,(label, probs) in enumerate(probss.items()):
            epochs=list(range(probs.shape[-1]))
            if temp_max<=np.max(probs):
                temp_max=np.max(probs)
            mean=np.average(probs,axis=0)
            std=np.std(probs,axis=0)
            ax.plot(epochs, mean, label=label, c=clrs[i],linewidth=2)
            ax.fill_between(epochs, mean-0.5*std, mean+0.5*std, alpha=0.2, facecolor=clrs[i])
        ax.set_ylim([-0.1 * temp_max, 1.1 * temp_max])
        ax.set_xlim([0,len(epochs)])
        # ax.set_xticklabels(np.arange(len(epochs))*100)
        ax.xaxis.set_major_formatter(lambda a, b: int(evaluation_frequency/1000 * a))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.legend(loc="upper left",framealpha=0.6)

    date = datetime.now().strftime("%m_%d-%H:%M")
    plt.show()
    fig.savefig(f"results/exp_{date}.png")

    if save_log:
        with open(f"results/exp_{date}.pkl", 'wb') as f:
            pickle.dump(probss, f)


def run_exp(i,*args):
    """
    Starts an instance of the training algorithm
    """
    prism = PRISMModel(i,ltl)
    np.random.seed(i)
    trainer = LearningAlgo(grid_mdp, automaton, prism, U=args[0][1], discount=args[0][2], eva_frequency=10000)
    learning_algo = getattr(trainer,args[0][0])

    Q,probs=learning_algo(*args[0][3:])

    return Q,probs,trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The parameters for running the experiments')
    parser.add_argument('--task', required=True, default='frozen_lake')
    parser.add_argument('--repeats',type=int, default=10)
    parser.add_argument('--save_log',action='store_true',default=False)

    args = parser.parse_args()

    print(args.save_log)

    mdp_map=GridMaps()
    if args.task=='probabilistic_gate':
        mdp_map.hard_example1()
        episode_length=100
        num_episode=40000
        K=10
    elif args.task=='frozen_lake':
        mdp_map.frozen_lake8x8()
        episode_length=200
        num_episode=6000
        K=10
    elif args.task == 'office_world':
        mdp_map.office_world3()
        episode_length = 1000
        num_episode = 6000
        K=5
    elif args.task == 'csrl':
        mdp_map.csrl_example()
        episode_length=100
        num_episode=1000
        K=10
    else:
        print("This task is not defined")

    U=0.1
    gamma=0.99

    grid_mdp = MDP(mdp_map,figsize=16)  # Use figsize=4 for smaller figures
    # MDP start
    start=mdp_map.start
    # LTL Specification
    ltl=mdp_map.ltl
    # Translate the LTL formula to an LDBA
    automaton = OmegaAutomaton(ltl)

    print('The task is ', mdp_map.name)
    print('Number of automaton states (including the trap state):',automaton.shape[1])
    print('Initial state:',automaton.q0)
    print('Transition function: ['),print(*['  '+str(t) for t in automaton.delta],sep=',\n'),print(']')
    print('epsilon transition: ['),print(*['  '+str(t) for t in automaton.eps],sep=',\n'),print(']')
    print('Acceptance: ['),print(*['  '+str(t) for t in automaton.acc],sep=',\n'),print(']')


    R=args.repeats
    probss={}

    exps=[
        [('omega_regular_rl', 1, 1, start, episode_length, num_episode,0.99), 'Hahn et al.'],
        [('lcrl_ql', 1, gamma, start, episode_length, num_episode), 'Hasanbeig et al.'],
        [('csrl_ql',U,gamma,start, episode_length, num_episode),'Bozkurt et al.'],
        [('efficient_ltl_learning',U, gamma, start, episode_length, num_episode, K, False), 'Ours (KC)'],
        [('efficient_ltl_learning',U, gamma, start, episode_length, num_episode, 0, True), 'Ours (CF)'],
        [('efficient_ltl_learning',U, gamma, start, episode_length, num_episode, K, True), 'Ours (CF+KC)'],
    ]


    begin=time.time()
    for params,label in exps:
        probs=[]
        results=Parallel(n_jobs=10,prefer='processes')(delayed(run_exp)(i,params) for i in range(R))
        # results = multiprocessing.Pool(50).starmap(run_exp, [params]*R)
        for Qs,prob,trainer in results:
            print(Qs.shape)
            probs.append(np.array(prob))
        probs=np.array(probs,dtype=float)
        probss[label]=probs
    end = time.time()
    print(end - begin)


    # print(len(probss))
    # for i in range(1):
    #     if len(Qs.shape)==6:
    #         Q=Qs[:,:,:,:,i,:]
    #     else:
    #         Q=Qs
    #     value=np.max(Q,axis=4)
    #     policy = np.argmax(Q, axis=4)
    #     print(policy)
    #
    #     for j in range(Q.shape[1]):
    #         trainer.plot(value,iq=(0,j))

    plot_probs(probss,trainer.evaluation_frequency,args.save_log)

    # episode=ltl_Q.simulate(policy,start=(0,0),T=1000)