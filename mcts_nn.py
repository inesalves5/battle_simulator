import random
import numpy as np
import math

EXPLORATION = 100

class MCTS_Node():
    
    def __init__(self, game, player, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.win_rate = 0
        self.player = player
        self.action = action

        
    def detach_parent(self):
        self.parent = None
    
    def is_terminal(self):
        return self.game.end() 
    
    def get_uct(self):
        if self.visits == 0:
            return float('inf')
        return self.win_rate + np.sqrt(2) * np.sqrt(np.log(self.parent.visits) / (self.visits))
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        self.win_rate = self.value / self.visits
        if self.parent:
            self.parent.backpropagate(value)

    def explore(self):
        current = self
        while current.children:
            max_U = max(c.get_uct() for c in self.children)
            actions = [(c, c.action) for c in self.children if c.get_uct() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)                      
            action = random.choice(actions)
            current = action[0]  
            
        if current.visits >= 1:
            current.create_child()
            if current.children:
                current = random.choice(current.children)
                            
        current.backpropagate(current.win_rate)
    
    def create_child(self):
        if self.is_terminal():
            return
        self.is_expanded = True
        actions = self.game.actions_available(self.player)
        probs, _ = self.model.forward(self.game_state, self.player)
        for action, prob in  zip(actions, probs):
            self.children.append(MCTS_Node(self.game_state.take_action(action, self.player*-1), self, action, prob, self.player * -1))
            
    def next(self):
        if self.is_terminal():
            raise ValueError("game has ended")
        if not self.children:
            raise ValueError('no children found and game hasn\'t ended')
        max_N = max(node.visits for node in self.children)
        max_children = [c for c in self.children if c.visits == max_N]
        if len(max_children) == 0:
            print("error zero length ", max_N) 
        max_child = random.choice(max_children)
        return max_child, max_child.action

def policy(tree):
    for _ in range(EXPLORATION):
        tree.explore()
    next_tree, next_action = tree.next()
    next_tree.detach_parent()
    return next_tree, next_action
