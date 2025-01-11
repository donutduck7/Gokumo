# A pure mcts for model evaluation
import numpy as np
import random
import copy
from operator import itemgetter

def rollout_policy(board):
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)

def policy_value_function(board):
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0

class Node(object):
    def __init__(self, parent, priorP):
        self._parent = parent
        self._children = {}  # a map from action to Node
        self._n_visits = 0
        self._Q = 0
        self._U = 0
        self._P = priorP
    
    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self._children:
                self._children[action] = Node(self, prob)
    
    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        self._U = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._U
    
    def is_leaf(self):
        return self._children == {}
    
    def is_root(self):
        return self._parent is None
    
    
class MCTS(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=1000):
        self._root = Node(None, 1.0)
        self._policy = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout

        
    def _playout(self, board):
        node = self._root
        while not node.is_leaf():
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            board.do_move(action)

        action_probs, _ = self._policy(board)
        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            node.expand(action_probs)
        
        # Evaluate the leaf node by random rollout.
        leaf_value = self._evaluate_rollout(board)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, board, limit=1000):
        current_player = board.get_current_player()
        for i in range(limit):
            end, winner = board.game_end()
            if end:
                break
            # Randomly choose next move.
            action_probs = rollout_policy(board)
            max_action = max(action_probs, key=itemgetter(1))[0]
            board.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit, 1000!!!")
        if winner == -1:
            return 0
        elif winner == current_player:
            return 1
        else:
            return -1
    
    def get_move(self, board):
        for i in range(self._n_playout):
            copy_state = copy.deepcopy(board)
            self._playout(copy_state)
        
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]
    
    def update_with_move(self, last_move):
        # Update root node.
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)
            
    def __str__(self): 
        return "MCTSPure"
    
class MCTSPurePlayer(object):
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_id(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
        
    def get_action(self, board):
        moves = board.availables
        if len(moves) > 0:
            move = self.mcts.get_move(board)
            # bookkeeping
            self.mcts.update_with_move(-1)
            return move
        
        else:
            print("WARNING: the board is full")
    
    def __str__(self):
        return "MCTSPure {}".format(self.player)
        