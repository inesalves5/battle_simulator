import math
import random
import copy
import game
import main   

class ChanceNode: #node de chance
    def __init__(self, game_state, parent, j_action, a_action, reward=[0, 0], rolls = None):
        self.game = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = [0, 0]
        self.j_action = j_action
        self.a_action = a_action
        self.max_reward = parent.max_reward
        self.reward = reward
        self.rolls = rolls

    def is_fully_expanded(self):
        return False
    
    def best_child(self, exploration_weight=1.4):
        '''Selects the best child based on UCT value.'''
        return max(
            self.children,
            key=lambda c: float('inf') if c.visits == 0 else 
            c.value[1] / (2 * self.max_reward) + exploration_weight * math.sqrt(math.log(self.visits) / (c.visits))
        )

    def expand(self):
        '''Expands by adding a new child node.'''
        game_copy = copy.deepcopy(self.game)
        game_copy, reward, rolls = game_copy.get_next_state([self.j_action, self.a_action])
        if game_copy is None:
            return None, [0, 0]
        child_node = DecisionNode(copy.deepcopy(game_copy), self.max_reward, parent=self, action=self.a_action, reward=[x+y for x, y in zip(reward, self.reward)], rolls=rolls)
        for existing_child in self.children:
            if existing_child == child_node:
                return existing_child, reward
        self.children.append(child_node)
        return child_node, reward

    def update(self, result):
        '''Updates node statistics after a simulation.'''
        self.value = [v * self.visits / (self.visits + 1) + r / (self.visits + 1)
                    for v, r in zip(self.value, result)]         
        self.visits += 1

    def add_child(self, child, reward):
        '''Adds a child node.'''
        if child.game == self.game:
            return
        for existing_child in self.children:
            if existing_child == child:
                return 
        self.children.append(child)
    
    def __eq__(self, other):
        return isinstance(other, ChanceNode) and self.game == other.game and self.parent == other.parent

    def __str__(self):
        final = ''
        for player in range(2):
            final += f"Player {player}:\n"
            for unit in self.game.units[player]:
                final += f' {unit["type"]} (Damage: {unit["damage"]})\n'
        return final

class DecisionNode: #node para as acoes 
    def __init__(self, game, max_reward, parent=None, action=None, player=0, reward=[0,0], root=False, rolls=None):         
        self.game = game
        self.max_reward = max_reward   
        self.parent = parent
        self.children = []
        self.visits = 1 if root else 0
        self.value = [0, 0]
        self.action = action if player == 1 else ' '
        self.player = player
        self.reward = reward
        self.untried_actions = game.actions_available(player)
        self.rolls = rolls

    def is_fully_expanded(self):
        if self.untried_actions == [] and self.children == []:
            print(self.game)
        return self.untried_actions == [] or (self.untried_actions is None and self.children != [])

    def best_child(self, exploration_weight=1.4):
        '''Selects the best child based on UCT value.'''  
        return max(
            self.children,
            key=lambda c: float('inf') if c.visits == 0 else 
            c.value[self.player] / (2 * self.max_reward) + exploration_weight * math.sqrt(math.log(self.visits) / c.visits)
        )
        
    def expand(self):
        '''Expands by adding a new child node.'''
        action = self.untried_actions.pop()
        for child in self.children:
            if isinstance(child, ChanceNode) and child.a_action == action:
                return child, [0, 0]
        if self.player == 0:
            child_node = DecisionNode(copy.deepcopy(self.game), self.max_reward, parent=self, player=1-self.player, action=action, reward=self.reward)
        else:
            child_node = ChanceNode(copy.deepcopy(self.game), parent=self, j_action=self.action, a_action=action, reward=self.reward)
        self.children.append(child_node)
        return child_node, [0, 0]
    
    def update(self, result):
        '''Updates node statistics after a simulation.'''
        self.value = [(v * self.visits + r) / (self.visits + 1) for v, r in zip(self.value, result)]
        self.visits += 1

    def add_child(self, child):
        '''Adds a child node.'''
        for existing_child in self.children:
            if existing_child == child:
                return 
        self.children.append(child)
              
    def __eq__(self, other):
        return isinstance(other, DecisionNode) and self.game == other.game and self.action == other.action and \
                self.parent == other.parent and self.player == other.player and self.reward == other.reward 

    def __str__(self):
        final = ''
        for player in range(2):
            final += f"Player {player}:\n"
            for unit in self.game.units[player]:
                final += f'  {unit["type"]} (Damage: {unit["damage"]})\n'
        return final
    
class MCTS:
    def __init__(self, root, nn=None):
        self.root = root
        self.nn = nn
    
    def search(self, node, iterations):
        for _ in range(iterations):
            self.sample(node)
        return node.best_child() # for cases where game is terminal from the root
    
    def sample(self, node):
        rewards = node.reward
        while not (node.game.is_terminal() and isinstance(node, DecisionNode) and node.player==0):
            if isinstance(node, ChanceNode):
                node, reward = node.expand()
                rewards = [x+y for x, y in zip(reward, rewards)]
                if node is None:
                    return [0, 0]    
            elif node.visits == 0:
                reward = self.simulate(node)
                if reward is None:
                    return [0, 0]
                rewards = [x+y for x, y in zip(reward, rewards)]
                break
            else:
                node, _ = self.select_action(node)
        if node is not None and node.game.is_terminal():
            rewards = [x+y for x, y in zip(rewards, node.game.reward_zone())] 
        self.backpropagate(node, rewards)
        return rewards    
    
    def select_action(self, node):
        if node.is_fully_expanded():
            return node.best_child(), [0, 0]
        return node.expand()
    
    def simulate(self, node):
        '''Performs a random playout and returns result.'''
        game = node.game
        current_game = copy.deepcopy(game)
        if self.nn is not None:
            if node.player == 0:
                return self.nn.predict_rewards(current_game.encode())
            else:
                return self.nn.predict_rewards(current_game.encode(), node.action)
        rewards = [0, 0]
        if node.player == 1:
            j_action = node.action
            a_action = current_game.action_available(1) 
            current_game, reward, _ = current_game.get_next_state([j_action, a_action])
            rewards = [x+y for x,y in zip(rewards, reward)]
            if current_game is None:
                return None
        while not current_game.is_terminal():
            j_action = current_game.action_available(0)
            a_action = current_game.action_available(1)
            current_game, reward, _ = current_game.get_next_state([j_action, a_action])
            if current_game is None:
                return None
            rewards = [x+y for x,y in zip(rewards, reward)]
        rewards = [x+y for x,y in zip(rewards, current_game.reward_zone())] 
        return rewards

    def backpropagate(self, node, result):
        '''Backpropagates the simulation result up the tree.'''
        while node:
            node.update(result)
            node = node.parent