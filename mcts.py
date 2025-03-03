import random
import numpy as np
import math

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
        self.untried_actions = game.actions_available(player)
        
    def is_terminal(self):
        return self.game.end() 
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        self.win_rate = self.value / self.visits
        if self.parent:
            self.parent.backpropagate(value)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=math.sqrt(2)):
        choices_weights = [
            c.win_rate + c_param * np.sqrt(np.log(self.visits) / c.visits)
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def rollout(self):
        current_state = self.game  # Start with the current game state
        current_player = self.player

        while not current_state.end():  # Continue until reaching a terminal state
            possible_moves = current_state.actions_available(current_player)
            action = self.rollout_policy(possible_moves)  # Select a move randomly
            current_state = current_state.take_action(action, current_player)  # Get the next state
            current_player *= -1  # Switch player

        return current_state.points[self.player]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.game.take_action(action, self.player)
        child_node = MCTS_Node(
            game=next_state, parent=self, action=action, player=self.player * -1
        )
        self.children.append(child_node)
        return child_node

    def best_action(self, simulations_number=None):
        if simulations_number is None :
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        else :
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.best_child(c_param=0.)

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node