import math
import random
import copy

class ChanceNode: #node de chance
    def __init__(self, game, parent, j_action, a_action):
        self.game = game
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = [0, 0]
        self.j_action = j_action
        self.a_action = a_action
        self.max_reward = parent.max_reward

    def is_fully_expanded(self):
        return False
    
    def best_child(self, exploration_weight=1.4): #esta a escolher a melhor para o player allied sempre
        """Selects the best child based on UCT value."""
        unexplored_actions = [child for child in self.children if child.visits == 0]
    
        if unexplored_actions:
            return self.expand()
        return max(
            self.children,
            key=lambda c: float('inf') if c.visits == 0 else 
            c.value[1] / (2 * self.max_reward) + exploration_weight * math.sqrt(math.log(self.visits) / (c.visits))
        )

    def expand(self):
        """Expands by adding a new child node."""
        game, reward = self.game.get_next_state([self.j_action, self.a_action])
        child_node = DecisionNode(copy.deepcopy(game), self.max_reward, parent=self, action=self.a_action)
        if child_node in self.children:
            return child_node
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """Updates node statistics after a simulation."""
        self.visits += 1
        self.value = [(r + self.visits * v) / (self.visits + 1) for v, r in zip(self.value, result)]
    
    def add_child(self, child):
        """Adds a child node."""
        self.children.append(child)
    
    def __eq__(self, other):
        return isinstance(other, ChanceNode) and self.game == other.game and self.a_action == other.a_action and\
                self.j_action == other.j_action and self.parent == other.parent 
                

class DecisionNode: #node para as acoes 
    def __init__(self, game, max_reward, parent=None, action=None, value=[0, 0], player=0): 
        self.game = game
        self.max_reward = max_reward   
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = value
        self.action = action
        self.untried_actions = list(game.actions_available(player))
        self.player = player

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4): #ver melhor constante que divide
        """Selects the best child based on UCT value."""
        unexplored_actions = [child for child in self.children if child.visits == 0]
    
        if unexplored_actions:
            return self.expand()
        return max(
            self.children,
            key=lambda c: float('inf') if c.visits == 0 else 
            c.value[self.player] / (2 * self.max_reward) + exploration_weight * math.sqrt(math.log(self.visits) / (c.visits))
        )
        
    def expand(self):
        """Expands by adding a new child node."""
        if not self.untried_actions:
            return self

        action = self.untried_actions.pop()
        if self.player == 0:
            child_node = DecisionNode(copy.deepcopy(self.game), self.max_reward, parent=self, player=1-self.player, action=action)
        else:
            child_node = ChanceNode(copy.deepcopy(self.game), parent=self, j_action=self.action, a_action=action)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Updates node statistics after a simulation."""
        self.visits += 1
        self.value = [(r + self.visits * v) / (self.visits + 1) for v, r in zip(self.value, result)]
        
    def add_child(self, child):
        """Adds a child node."""
        self.children.append(child)
              
    def __eq__(self, other):
        return isinstance(other, DecisionNode) and self.game == other.game and self.action == other.action and \
                self.parent == other.parent and self.untried_actions == other.untried_actions and self.player == other.player

class MCTS:
    def __init__(self, root):
        self.root = root
        
    def search(self, root, iterations=1000):
        if root.game.is_terminal():
            return root
        for _ in range(iterations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.game.is_terminal(): #node
                node = node.expand()

            # Simulation
            result = self.simulate(node.game) 

            # Backpropagation
            self.backpropagate(node, result)

        return root.best_child(exploration_weight=0)  # Best child without exploration factor

    def simulate(self, game):
        """Performs a random playout and returns result."""
        current_game = game
        rewards = [0, 0]
        while not current_game.is_terminal():
            j_action = random.choice(list(current_game.actions_available(0)))
            a_action = random.choice(list(current_game.actions_available(1)))
            current_game, reward = current_game.get_next_state([j_action, a_action])
            rewards = [x+y for x,y in zip(rewards, reward)]
        rewards = [x+y for x,y in zip(rewards, current_game.reward_zone())]
        return rewards

    def backpropagate(self, node, result):
        """Backpropagates the simulation result up the tree."""
        while node:
            node.update(result)
            node = node.parent