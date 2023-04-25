import numpy as np
# from IPython.display import display
from matplotlib import pyplot as plt


class GridMaps():
    """
    Stores the MDP environments and the corresponding tasks

    Attributes
    ----------
    shape : The shape of the MDP environment

    structure : The structure of the environment, including walls, traps, obstacles, etc.

    label : The label of each of the MDP states

    lcmap : The colour of different labels

    p : The probability that the agent moves in the intended direction, with probability (1-p)/2 of going sideways

    start : The start position in the MDP for the agent
    """

    def __init__(self):
        self.shape=None
        self.structure=None
        self.label=None
        self.lcmap=None
        self.p = None
        self.plot_start=None

    def csrl_example(self):

        self.ltl = '((F G a) | (F G b)) & (G !c)'
        self.p=0.8
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
        ['E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'T'],
        ['B',  'E',  'E',  'E'],
        ['T',  'E',  'T',  'E'],
        ['E',  'E',  'E',  'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),       (),     ('c',),()],
        [(),       (),     ('a',),('b',)],
        [(),       (),     ('c',),()],
        [('b',),   (),     ('a',),()],
        [(),       ('c',), (),    ('c',)]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('a',):'lightgreen',
            ('b',):'lightgreen',
            ('c',):'pink'
        }
        self.shape = self.structure.shape
        self.start=(0,0)
        self.plot_start=(0,0)
        self.name='csrl'

    def hard_example1(self):
        # MDP Description
        self.ltl = '(F G a) & (G !c)'
        self.p = 0.8
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
            ['R','R','R','R','R','R','R','R','R','T'],
            ['E','B','B','B','B','B','B','B','B','B'],
            ['RD','E','E','E','E','E','E','E','E','T'],
            ['T','B','B','B','B','B','B','B','B','B']
        ])

        # Labels of the states
        self.label = np.array([
            [('a',),('a',),('a',),('a',),('a',),('a',),('a',),('a',),('a',),()],
            [(),(),(),(),(),(),(),(),(),()],
            [(),(),(),(),(),(),(),(),(),('a',),],
            [('c',),(),(),(),(),(),(),(),(),(),]
        ], dtype=np.object)
        # Colors of the labels
        self.lcmap = {
            ('a',): 'green',
            ('c',): 'red'
        }
        self.shape = self.structure.shape
        self.start=(1,0)
        # self.start=None
        self.plot_start=(1,0)
        self.name='probabilistic gate'


    def hard_example2(self):
        # MDP Description
        self.ltl = '(F G a) & (G !c)'
        self.p = 0.7
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
            ['D', 'E', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['D', 'B', 'E', 'E', 'E'],
            ['T', 'B', 'E', 'T', 'E']
        ])

        # Labels of the states
        self.label = np.array([
            [('a',), (), (), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [('a',), (), ('c',), (), ('c',)],
            [(), (), ('c',), ('a',), ('c',)]
        ], dtype=np.object)
        # Colors of the labels
        self.lcmap = {
            ('a',): 'green',
            ('c',): 'red'
        }
        self.shape = self.structure.shape
        self.start=(0,1)
        # self.start=None
        self.plot_start=(0,1)
        self.name='hard2'


    def patrol_example(self):
        # MDP Description
        self.ltl = 'G F(a & (X F b)) & (G !c)'
        # G F(a & X F b) & (G !c)
        self.p=0.9
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
        ['B',  'E',  'E',  'E',  'E'],
        ['B',  'E',  'B',  'B',  'E'],
        ['T',  'E',  'T',  'B',  'E'],
        # ['T',  'E',  'T',  'B',  'E'],
        # ['T',  'E',  'T',  'B',  'E'],
        ['B',  'E',  'B',  'B',  'E'],
        ['B',  'E',  'E',  'E',  'E']
        ])


        # Labels of the states
        self.label = np.array([
        [(),    (),    (),    (),()],
        [(),    ('b',),(),    (),()],
        [('c',),(),    ('c',),(),()],
        # [('c',),(),    ('c',),(),()],
        # [('c',),(),    ('c',),(),()],
        [(),    ('a',),(),    (),()],
        [(),    (),    (),    (),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('a',):'green',
            ('b',):'green',
            ('c',):'red'
        }
        self.shape = self.structure.shape
        self.start=(3,1)
        self.plot_start=(3,1)

        self.name='patrol'


    def patrol_example2(self):
        # MDP Description
        self.ltl = 'G F(a & (X F b)) | (F G c)'
        #  'G F(a & (X F b)) | (F G c)'
        self.p=0.9
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
        ['E',  'E',  'E',  'E'],
        ['E',  'B',  'B',  'E'],
        ['E',  'B',  'B',  'E'],
        ['R',  'AR',  'B',  'E'],
        ['B',  'T',  'B',  'B'],
        # ['E',  'B',  'B',  'E'],
        # ['E',  'E',  'E',  'E']
        ])


        # Labels of the states
        self.label = np.array([
        [('a',),(),(),()],
        [(),(),(),('b',)],
        [(),(),(),()],
        [(),('c',),(),()],
        [(),(),    (),()],
        # [(),    (),    (),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('a',):'green',
            ('b',):'green',
            ('c',):'red'
        }
        self.shape = self.structure.shape
        self.start=(2,0)
        self.plot_start=(2,0)

        self.name='patrol'


    def patrol_example3(self):
        # MDP Description
        self.ltl = 'G F(a & (X F b)) & (G !c)'
        self.p=1
        # E: Empty, T: Trap, B: Obstacle
        self.structure = np.array([
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E'],
        ['E',  'E',  'E',  'E',  'E']
        ])

        # Labels of the states
        self.label = np.array([
        [('c',),(),(),(),('b',)],
        [(),(),('c',),('c',),()],
        [(),(),(),(),()],
        [(),(),(),(),()],
        [('a',),(),('c',),(),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('a',):'green',
            ('b',):'green',
            ('c',):'red'
        }
        self.shape = self.structure.shape
        self.start=(3,1)
        self.plot_start=(3,1)

        self.name='patrol'

    def office_world(self):
        self.ltl='(G F t) & (G F l) & (G F a) & (G !o)'
        # self.ltl='G((F t) & (F l) & (X F a)) & (G !o)'
        self.p=1

        # E: Empty, T: Trap, B: Obstacle
        self.structure=np.array([
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),('t',),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('o',),(),(),(),('a',),(),(),(),('l',),(),(),(),('o',),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),('t',),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('l',): 'orange',
            ('a',):'green',
            ('t',):'blue',
            ('o',):'red',
        }

        self.shape = self.structure.shape
        self.start = (9, 2)
        self.plot_start = (9, 2)
        self.name='office world'


    def office_world2(self):
        self.ltl='((F (l & (X ((G F t) & (G F a)))))|(F G b)) & (G !o)'
        # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
        self.p=0.5

        # E: Empty, T: Trap, B: Obstacle
        self.structure=np.array([
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'U', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['AU', 'U', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'R', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),('t',),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [('b',),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('o',),(),(),(),('a',),(),(),(),('l',),(),(),(),('o',),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),('t',),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('l',): 'darkorange',
            ('a',):'green',
            ('t',):'blue',
            ('o',):'red',
            ('b',):'brown',
        }

        self.shape = self.structure.shape
        self.start = (9, 2)
        self.plot_start = (9, 2)
        self.name='office world 2'



    def office_world3(self):
        self.ltl='((F (l & (X ((G F t) & (G F w)))))|((G F a)&(G F b))) & (G !o)'
        # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
        self.p=0.8

        # E: Empty, T: Trap, B: Obstacle P: slippery
        self.structure=np.array([
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'R', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['P', 'P', 'P', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['P', 'P', 'P', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['P', 'P', 'P', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'R', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('b',),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),('t',),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('o',),(),(),(),('w',),(),(),(),('l',),(),(),(),('o',),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),('t',),(),(),(),()],
        [(),('a',),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
        ],dtype=np.object)
        # Colors of the labels
        self.lcmap={
            ('l',): 'darkorange',
            ('w',):'green',
            ('t',):'blue',
            ('o',):'red',
            ('a',):'brown',
            ('b',): 'brown',

        }

        self.shape = self.structure.shape
        self.start = (9, 0)
        self.plot_start = (9, 0)
        self.name='office world'


    def frozen_lake8x8(self):
        self.ltl = '((G F a)|(G F b)) & (G !h)'
        # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
        self.p = 0.34
        # self.p = 0.4

        # E: Empty, T: Trap, B: Obstacle P: slippery
        self.structure = np.array([
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'E', 'P', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
            ['P', 'E', 'E', 'P', 'P', 'P', 'E', 'P'],
            ['P', 'E', 'P', 'P', 'E', 'P', 'E', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'E'],
        ])

        # Labels of the states
        self.label = np.array([
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), ()],
            [(), (),    (),     ('h',), (), (), (), ()],
            [(), (),    (),     (),     (),('h',), (), ()],
            [(), (),    (),     ('h',), (), (), ('b',), ()],
            [(), ('h',), ('h',), (),     (), (), ('h',), ()],
            [(), ('h',), (),    (),  ('h',), (), ('h',), ()],
            [(), (), (), ('h',), (), (), (), ('a',)]
        ], dtype=np.object)
        # Colors of the labels
        self.lcmap = {
            ('h',): 'blue',
            ('a',): 'red',
            ('b',): 'darkorange',

        }

        self.shape = self.structure.shape
        self.start = (0, 0)
        self.plot_start = (0, 0)
        self.name = 'frozen lake'