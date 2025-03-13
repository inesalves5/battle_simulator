import math
import random

class MCTSNode:
    def __init__(self, game, parent=None, player=0, action=None, jap_action=None):
        self.game = game
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = [0, 0]
        self.win_rate = 0
        self.wins = 0
        self.to_play = player
        self.action = action
        self.jap_action = jap_action
        self.untried_actions = list(game.actions_available(player))

    def is_fully_expanded(self):
        return len(self.children) >= len(self.untried_actions)

    def best_child(self, exploration_weight=1.4):
        """Selects the best child based on UCT value."""
        return max(
            self.children,
            key=lambda c: (c.value[self.to_play] / (c.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-6))
        )

    def expand(self):
        """Expands by adding a new child node."""
        if not self.untried_actions:
            return self

        action = random.choice(self.untried_actions)
        new_player = 1 - self.to_play  # Switch player

        if self.to_play == 0: #jogador japones
            child_node = MCTSNode(self.game, parent=self, player=new_player, action=action, jap_action=self.action)
        else: #jogador aliado => calcular danos
            next_game = self.game.get_next_state([self.action, action])
            child_node = MCTSNode(next_game, parent=self, player=new_player, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """Updates node statistics after a simulation."""
        self.visits += 1
        self.value = [x+y for x,y in zip(self.value, result)]
        if result[self.to_play] > 0:
            self.wins += 1
        self.win_rate = self.wins / self.visits

class MCTS:
    def __init__(self, game):
        self.game = game

    def search(self, root, iterations=1000):
        for _ in range(iterations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not (node.game.is_terminal() and node.to_play == 0): #node
                node = node.expand()

            # Simulation
            result = self.simulate(node.game, node.to_play, node.action) 

            # Backpropagation
            self.backpropagate(node, result)

        return root.best_child(exploration_weight=0)  # Best child without exploration factor

    def simulate(self, game, player, prev):
        """Performs a random playout and returns result."""
        current_game = game
        current_player = player
        while not (current_game.is_terminal() and current_player == 0):
            action = random.choice(current_game.actions_available(current_player))
            if current_player == 0: #japones
                prev = action
            else:
                current_game = current_game.get_next_state([prev, action])
            current_player = 1 - current_player  # Switch player
        return current_game.points

    def backpropagate(self, node, result):
        """Backpropagates the simulation result up the tree."""
        while node:
            node.update(result)
            node = node.parent