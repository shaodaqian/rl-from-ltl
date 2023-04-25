import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # matplotlib.font_manager._rebuild()
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact

# Up, Down, Right, Left
Actions = ['U','D','R','L']

class MDP():
    """
    This class implements a Markov Decision Process where an agent can move up, down, right or left in a 2D grid world.
    
    Attributes
    ----------
    shape : The shape of the grid.
        
    transition_probs : The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.

    Parameters
    ----------

    structure : The structure of the environment, including walls, traps, obstacles, etc.

    label : The label of each of the MDP states
        
    A: The list of actions represented by a string.
    
    p : The probability that the agent moves in the intended direction, with probability (1-p)/2 of going sideways
    
    figsize: The size of the matplotlib figure to be drawn when the method plot is called.
    
    lcmap : The colour of different labels

    cmap: The colormap to be used when drawing the plot of the MDP. The default value is matplotlib.cm.RdBu.
    """
    
    def __init__(self, mdp_map, A=Actions, figsize=6, cmap=plt.cm.RdBu):
        self.shape = mdp_map.shape
        n_rows, n_cols = self.shape

        # Create the default structure, reward and label if they are not defined.
        self.structure = mdp_map.structure if mdp_map.structure is not None else np.full(self.shape,'E')
        self.reward = np.zeros((n_rows,n_cols))
        self.label = mdp_map.label if mdp_map.label is not None else np.empty(self.shape,dtype=np.object); self.label.fill(()) if mdp_map.label is None else None
        
        self.p = mdp_map.p
        self.p2=round((1-mdp_map.p)/2*1000)/1000
        self.A = A
        
        # Create the transition matrix
        self.transition_probs = np.empty((n_rows, n_cols, len(A)),dtype=np.object)
        for state in self.states():
            for action, action_name in enumerate(A):
                self.transition_probs[state][action] = self.get_transition_prob(state,action_name)
        
        self.figsize = figsize
        self.cmap = cmap
        self.lcmap = mdp_map.lcmap
        self.start=mdp_map.start
        self.plot_start=mdp_map.plot_start
        
    def states(self):
        """
        Iterates through all product states
        """
        n_rows, n_cols = self.shape
        for state in product(range(n_rows),range(n_cols)):
            yield state
        
    def random_state(self):
        """
        Generates a random product state.
        """
        n_rows, n_cols = self.shape
        state = np.random.randint(n_rows),np.random.randint(n_cols)
        return state

    def allowed_actions(self,state):
        """
        Returns the allowed actions from a state according to the structure of the MDP
        """
        cell_type = self.structure[state]
        if cell_type in ['B', 'T','E','P']:
            return list(range(len(self.A)))
        elif cell_type[0] in ['U','D','R','L']:
            return [self.A.index(cell_type[0])]
        elif cell_type[0]=='A':
            return [self.A.index(cell_type[1])]


    def get_transition_prob(self,state,action_name):
        """
        Returns the list of possible next states with their probabilities when the action is taken (next_states,probs).
        If the direction is blocked by an obtacle or the agent is in a trap state then the agent stays in the same position.
        """
        cell_type = self.structure[state]
        if cell_type in ['B', 'T']:
            return [state], np.array([1.])
        if cell_type[0]== 'A':
            action_name=cell_type[1]
        if cell_type in ['U', 'D', 'R', 'L']:
            action_name=cell_type
        n_rows, n_cols = self.shape
        states, probs = [], []
        if self.p==1 or cell_type =='E' or cell_type in ['U', 'D', 'R', 'L']:
            if action_name=='D' and state[0]+1 < n_rows and self.structure[state[0]+1][state[1]] != 'B' :
                states.append((state[0] + 1, state[1]))
                probs.append(1)
            elif action_name=='U' and state[0]-1 >= 0 and self.structure[state[0]-1][state[1]] != 'B':
                states.append((state[0] - 1, state[1]))
                probs.append(1)
            elif action_name=='L' and state[1]-1 >= 0 and self.structure[state[0]][state[1]-1] != 'B' :
                states.append((state[0], state[1] - 1))
                probs.append(1)
            elif action_name=='R' and state[1]+1 < n_cols and self.structure[state[0]][state[1]+1] != 'B':
                states.append((state[0], state[1] + 1))
                probs.append(1)

        elif cell_type=='RD':
            states.append((state[0], state[1] + 1))
            probs.append(self.p)
            states.append((state[0] + 1, state[1]))
            probs.append(1-self.p)
        else:
            # South
            if action_name!='U' and state[0]+1 < n_rows and self.structure[state[0]+1][state[1]] != 'B':
                states.append((state[0]+1,state[1]))
                probs.append(self.p if action_name=='D' else self.p2)
            # North
            if action_name!='D' and state[0]-1 >= 0 and self.structure[state[0]-1][state[1]] != 'B':
                states.append((state[0]-1,state[1]))
                probs.append(self.p if action_name=='U' else self.p2)
            # West
            if action_name!='R' and state[1]-1 >= 0 and self.structure[state[0]][state[1]-1] != 'B':
                states.append((state[0],state[1]-1))
                probs.append(self.p if action_name=='L' else self.p2)
            # East
            if action_name!='L' and state[1]+1 < n_cols and self.structure[state[0]][state[1]+1] != 'B':
                states.append((state[0],state[1]+1))
                probs.append(self.p if action_name=='R' else self.p2)
        
        # If the agent cannot move in some of the directions
        probs_sum = np.sum(probs)
        if probs_sum>1:
            print(probs)
            print('probability sum to greater then 1')
        if probs_sum<1:
            states.append(state)
            probs.append(round(1000*(1-probs_sum))/1000)
        return states, probs

    
    def plot(self, value=None, policy=None, agent=None, save=None, hidden=[], path={}):
        """
        Plots the values of the states as a color matrix.
        
        Parameters
        ----------
        value : The value function. If it is None, the reward function will be plotted.
            
        policy : Optional, the policy to be visualized.
            
        agent : Optional, the position of the agent to be plotted.
        """
        
        f=FontProperties(weight='bold')
        fontname = 'Times New Roman'
        fontsize = 20
        plot_value=True

        if value is None:
            plot_value=False
            value = self.reward
        else:
            value = np.copy(value)
            for h in hidden:
                value[h] = 0
        
        # Dimensions
        n_rows, n_cols = self.shape
        
        # Plot
        fig = plt.figure(figsize=(self.figsize,self.figsize))
        plt.tight_layout()

        # plt.rc('text', usetex=True)
        plt.rc('text', usetex=False)
        print(value)
        threshold = np.nanmax(np.abs(value))*2
        threshold = 1 if threshold==0 else threshold
        plt.imshow(value, interpolation='nearest', cmap=self.cmap, vmax=threshold, vmin=-threshold)
        
        # Get the axes
        ax = fig.axes[0]

        # Major ticks
        ax.set_xticks(np.arange(0, n_cols, 1))
        ax.set_yticks(np.arange(0, n_rows, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(n_cols), fontsize=fontsize)
        ax.set_yticklabels(np.arange(n_rows), fontsize=fontsize)
        # Minor ticks
        ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
        
        # Move x axis to the top
        ax.xaxis.tick_top()
        plt.tick_params(left=False, top=False,bottom=False)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='dimgray', linestyle='-', linewidth=1)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.tick_params(bottom=False, left=False,top=False)
        
        # Draw the agent
        if agent:  
            circle=plt.Circle((agent[1],agent[0]-0.17),0.26,color='lightblue',ec='purple',lw=2)
            plt.gcf().gca().add_artist(circle)

        for i, j in self.states():  # For all states
            if (i,j) in path:
                if 'u' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i+0.4),+0.8,-0.9,color='lightcoral')
                    plt.gcf().gca().add_artist(rect)
                if 'd' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i-0.4),+0.8,+0.9,color='lightcoral')
                    plt.gcf().gca().add_artist(rect)
                if 'r' in path[i,j]:
                    rect=plt.Rectangle((j-0.4,i-0.4),+0.9,+0.8,color='lightcoral')
                    plt.gcf().gca().add_artist(rect)
                if 'l' in path[i,j]:
                    rect=plt.Rectangle((j+0.4,i-0.4),-0.9,+0.8,color='lightcoral')
                    plt.gcf().gca().add_artist(rect)
                    
            cell_type = self.structure[i,j]
            # If there is an obstacle

            if cell_type == 'B':
                rect=plt.Rectangle((j-0.5,i-0.5),+1,+1,color='gray',fc='gray')
                plt.gcf().gca().add_artist(rect)
                continue
            # If it is a trap cell
            elif cell_type == 'T':
                circle=plt.Circle((j,i),0.49,color='k',fill=False)
                plt.gcf().gca().add_artist(circle)

            elif cell_type == 'P':
                rect=plt.Rectangle((j-0.5,i-0.5),+1,+1,color='paleturquoise')
                plt.gcf().gca().add_artist(rect)
                
            # If it is a directional cell (See the description of the class attribute 'structure' for details)
            elif cell_type == 'U':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j+0.5,i+0.5]], color='darkgray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'D':
                triangle = plt.Polygon([[j,i],[j-0.5,i-0.5],[j+0.5,i-0.5]], color='darkgray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'R':
                triangle = plt.Polygon([[j,i],[j-0.5,i+0.5],[j-0.5,i-0.5]], color='darkgray')
                plt.gca().add_patch(triangle)
            elif cell_type == 'L':
                triangle = plt.Polygon([[j,i],[j+0.5,i+0.5],[j+0.5,i-0.5]], color='darkgray')
                plt.gca().add_patch(triangle)


            # If the background is too dark, make the text white
            color = 'white' if np.abs(value[i, j]) > threshold/2 else 'black'
            
            if policy is None:  # Print the values
                if plot_value:
                    v = str(int(round(100*value[i,j]))).zfill(3)
                    plt.text(j, i, '$'+v[0]+'.'+v[1:]+'$',horizontalalignment='center',color=color,fontname=fontname,fontsize=fontsize+2)  # Value
                # plt.text(j, i, v[0]+'.'+v[1:],horizontalalignment='center',color=color,fontname=fontname,fontsize=fontsize+2)  # Value

            # Draw the arrows to visualize the policy
            elif value[i,j] > 0 or value is self.reward:  
                if policy[i,j] >= len(self.A):
                    plt.text(j, i-0.05,r'$\epsilon_'+str(policy[i,j]-len(self.A))+'$', horizontalalignment='center',color=color,fontsize=fontsize+5)
                    # plt.text(j, i-0.05,'epsilon_'+str(policy[i,j]-len(self.A)), horizontalalignment='center',color=color,fontsize=fontsize+5)
                else:
                    action_name = self.A[policy[i,j]]
                    if action_name == 'U':
                        plt.arrow(j,i,0,-0.2,head_width=.2,head_length=.15,color=color)
                    elif action_name == 'D':
                        plt.arrow(j,i-.3,0,0.2,head_width=.2,head_length=.15,color=color)
                    elif action_name == 'R':
                        plt.arrow(j-.15,i-0.15,0.2,0,head_width=.2,head_length=.15,color=color)
                    elif action_name == 'L':
                        plt.arrow(j+.15,i-0.15,-0.2,0,head_width=.2,head_length=.15,color=color)
            
            # Plot the labels
            surplus = 0.2 if (i,j) in hidden else 0
            if self.label[i,j] in self.lcmap:
                # circle=plt.Circle((j, i+0.24-surplus),0.2+surplus/2,color=self.lcmap[self.label[i,j]])
                # plt.gcf().gca().add_artist(circle)
                plt.text(j, i+0.4-surplus,'$'+','.join(self.label[i,j])+'$',horizontalalignment='center',color=self.lcmap[self.label[i,j]],fontproperties=f,fontname=fontname,fontsize=fontsize+5+surplus*10)
            # if self.label[i,j]:
            #     plt.text(j, i+0.4-surplus,'$'+','.join(self.label[i,j])+'$',horizontalalignment='center',color=color,fontproperties=f,fontname=fontname,fontsize=fontsize+5+surplus*10)
                # plt.text(j, i+0.4-surplus,','.join(self.label[i,j]),horizontalalignment='center',color=color,fontproperties=f,fontname=fontname,fontsize=fontsize+5+surplus*10)

        j,i=self.plot_start
        triangle = plt.Polygon([[i, j-0.25], [i - 0.25*0.866, j+ 0.125], [i + 0.25*0.866, j + 0.125]], color='purple',alpha=0.5)
        plt.gca().add_patch(triangle)

        if save:
            plt.savefig(save,bbox_inches='tight')
