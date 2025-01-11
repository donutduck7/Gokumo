import numpy as np
import copy
# import torch

def softmax(x):
    tmp = np.exp(x - np.max(x))
    tmp /= np.sum(tmp)
    return tmp

class Node(object):
    def __init__(self, parent, priorP):
        self._parent = parent
        self._children = {}  # children dict
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = priorP
    
    def expand(self, action_priors):
        # Expand the tree by creating new children.
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = Node(self, prob)
    
    def select(self, c_puct):
        # Select action among children that gives maximum action value Q + bonus u(P)
        selected = max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
        return selected
    
    def update(self, leaf):
        # Update node values from leaf evaluation.
        self._n_visits += 1
        self._Q += 1.0*(leaf - self._Q) / self._n_visits
    
    def update_recursive(self, leaf):
        # Propagate value up tree
        if self._parent:
            self._parent.update_recursive(-leaf)
        self.update(leaf)
    
    def get_value(self, c_puct):
        # Calculate and return the value for this node.
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits)) / (1 + self._n_visits)
        return self._Q + self._u
    
    def is_leaf(self):
        # Check if leaf node (i.e. no nodes below this have been expanded)
        return self._children == {}

    
    def is_root(self):
        # Check if root node
        return self._parent is None

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):
        self._root = Node(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
    
    def _playout(self, state):
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # Leaf node
        probs, v = self._policy(state)
        # print(type(probs))
        # print(type(v))
        # v = v.cpu().numpy()
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            node.expand(probs)
        else:
            if winner == -1: # Draw
                v = 0.0
            else:
                v = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        node.update_recursive(-v)
    
    def get_move_probs(self, state, temp=1e-3):
        for i in range(self._n_playout):
            copy_state = copy.deepcopy(state)
            self._playout(copy_state)
        
        visited_act = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*visited_act)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs
    
    def update_with_move(self, last_move):
        # Update the tree
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)
    
    def __str__(self):
        return "MCTS for AlphaZero"

class MCTSPlayer(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        
    def set_player_id(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
        
    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            
            if self._is_selfplay:
                # add Dirichlet Noise for exploration
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
                
            if return_prob: 
                return move, move_probs
            else:
                return move
        else:
            print("Board is FULL!!!!!!")
    
    def __str__(self):
        return "MCTS Player".format(self.player)