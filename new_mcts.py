import math
import random
import itertools

class MCTSNode:
    def __init__(self, game, parent=None, action=None):
        self.game = game
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0 
        self.wins = 0
        self.action = action
        self.untried_actions = list(itertools.product(game.actions_available(0), game.actions_available(1)))

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4):
        """Selects the best child based on UCT value."""
        return max(
            self.children,
            key=lambda c: (c.wins / (c.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-6))
        )

    def expand(self):
        """Expands by adding a new child node."""
        if self.is_fully_expanded():
            return None
        action = self.untried_actions.pop()
        next_game = self.game.get_next_state(action)
        child_node = MCTSNode(next_game, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """Updates node statistics after a simulation."""
        self.visits += 1
        self.value += result
        if result > 0:
            self.wins += 1        

class MCTS:
    def __init__(self, game):
        self.game = game

    def search(self, root, iterations=1000):
        if root.game.is_terminal():
            return root
        for _ in range(iterations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            if not node.game.is_terminal():
                node = node.expand()
            # Simulation
            result = self.simulate(node.game)
            
            # Backpropagation
            self.backpropagate(node, result)

        return root.best_child(exploration_weight=0)  # Best child without exploration factor

    def simulate(self, game):
        """Performs a random playout and returns total reward instead of binary win/loss."""
        current_game = game
        total_reward = [0, 0]
        terminal =  current_game.is_terminal()
        while not terminal:
            actions = list(itertools.product(current_game.actions_available(0), current_game.actions_available(1)))
            action = random.choice(actions)
            current_game, reward, terminal = current_game.step(action)
            total_reward = [x+y for x,y in zip(reward, total_reward)]
        reward_zone = current_game.reward_zone()
        total_reward = [x+y for x,y in zip(reward_zone, total_reward)]
        return total_reward[1]

    def backpropagate(self, node, result):
        """Backpropagates the total game reward up the tree."""
        while node:
            node.update(result)
            node = node.parent