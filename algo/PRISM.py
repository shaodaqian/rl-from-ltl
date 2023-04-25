import os

import numpy as np
import matplotlib.pyplot as plt
from math import prod
import subprocess
import time
from filelock import FileLock


class PRISMModel:
    """
    Builds the PRISM model from the MDP and the current policy and evaluate it using PRISM

    Attributes
    ----------

    mdp_shape :The shape of the environment MDP.


    Parameters
    ----------
    multiprocess: The number of multiprocesses for experiments.

    ltl : The linear temporal logic (LTL) formula to be transformed.

    model_tpe : The type of PRISM model to be constructed. The default value is 'DTMC'
    """
    def __init__(self,multiprocess,ltl,model_tpe='DTMC'):
        self.ltl=ltl
        self.model_type=model_tpe
        self.multiprocess=multiprocess

    def index_to_coor(self,index):
        """
        Transforms states index to its coordinates in the grid MDP
        """
        return index // self.mdp_shape[1], index % self.mdp_shape[1]

    def coor_to_index(self,coor):
        """
        Transforms states coordinates in the grid MDP to its index
        """
        return coor[0]*self.mdp_shape[1]+coor[1]

    def build_model(self,mdp,automaton,policy):
        """
        Builds the PRISM model and calls PRISM to evaluate the probability of it satisfying the LTL specification.
        """
        self.mdp_shape= mdp.shape

        # Actions = ['U', 'D', 'R', 'L']

        model = "dtmc"
        model += "\nmodule ProductMDP"
        model += f"\n    m : [0..{prod(mdp.shape)}] init {self.coor_to_index(mdp.plot_start)};"
        model += f"\n    a : [0..{prod(automaton.shape)}] init 0;"


        # counting is done row by row
        for mdp_index in range(prod(mdp.shape)):
            r, c = self.index_to_coor(mdp_index)
            for auto_index in range(prod(automaton.shape)):
                i,q= auto_index//automaton.shape[1],auto_index%automaton.shape[1]
                action=int(policy[i,q,r,c])
                if action < len(mdp.A):  # MDP actions
                    q_ = automaton.delta[q][mdp.label[r, c]]
                    mdp_states, probs = mdp.get_transition_prob((r, c), mdp.A[action])

                    model += f"\n    [ac] (m={mdp_index})&(a={auto_index}) -> "
                    for index, next_state in enumerate(mdp_states):
                        if index==0:
                            model+= f"{probs[index]} : (m'={self.coor_to_index(next_state)})&(a'={q_})"
                        else:
                            model += f" + {probs[index]} : (m'={self.coor_to_index(next_state)})&(a'={q_})"
                    model+=";"
                else:  # epsilon-actions
                    model+= f"\n    [ep] (m={mdp_index})&(a={auto_index}) -> (m'={mdp_index})&(a'={action-len(mdp.A)});"
        model += "\nendmodule\n"
        # print(model)


        labels={}
        for mdp_index in range(prod(mdp.shape)):
            r, c = mdp_index // mdp.shape[1], mdp_index % mdp.shape[1]
            for proposition in mdp.label[r,c]:
                if proposition in labels:
                    labels[proposition].append(mdp_index)
                else:
                    labels[proposition]=[mdp_index]

        for p,array in labels.items():
            model += f'\nlabel "{p}" = '
            for index, mdp_index in enumerate(array):
                if index==0:
                    model += f"(m={mdp_index})"
                else:
                    model += f" | (m={mdp_index})"
            model+=";"

        ltl=list(self.ltl)
        for i,char in enumerate(ltl):
            if char in labels:
                ltl[i]=f'"{char}"'
        ltl="".join(ltl)

        prism_file=f"algo/prism_files/rl_ltl{self.multiprocess}.prism"

        with open(prism_file, "w") as text_file:
            text_file.write(model)
        out = subprocess.run(['prism', prism_file,'-maxiters','1000000','-pf', f'P=? [{ltl}]'],capture_output=True).stdout
        if len(out.decode('utf-8').split('---------------------------------------------------------------------\n'))<=1:
            print(out.decode('utf-8'))
            f = open(prism_file, "r")
            print(f.read())
            prob=1
        else:
            result=out.decode('utf-8').split('---------------------------------------------------------------------\n')[1]
            # print(result)
            try:
                prob = float(result.splitlines()[-2].split(":")[-1])
            except:
                print("Error when evaluating the PRISM model!", result)

        return prob
