import mcts
import json
import game
import random
import new_game
import new_mcts

def choose_random():

    with open("units.json", "r") as file:
        units = json.load(file)

    nr_japanese = random.randint(1, len(units["japanese"]))
    japanese = random.sample(units["japanese"], nr_japanese)
    
    nr_allied = random.randint(1, len(units["allied"]))
    allied = random.sample(units["allied"], nr_allied)

    return japanese, allied

def represent(game, action, player):
    if player == 0:
        units = game.japanese()
        opponent = game.allied()
    else:    
        units = game.allied()
        opponent = game.japanese()
        
    print("Player:", "Japanese" if player == 0 else "Allied")
    for i in range(len(action)):
        print("Unit", units[i], "attacks:", opponent[action[i]])
    print("---- end ----")
    
def represent_2(game, actions):
    print("Starting... type: ", game.action)
    print("actions:", actions)
    for player in range(2):
        action = actions[player]
        units = game.units[player]
        opponent = game.units[1-player]
        
        print("Player:", "Japanese" if player == 0 else "Allied")
        for i in range(len(action)):
            print("Unit", units[i], "attacks:", opponent[action[i]] if action[i] != None else "None")
        print("---- end ----")

def main():
    total_reward = [0, 0]
    japanese, allied = choose_random()
    units = [
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in japanese],
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in allied]
    ]
    game_state = game.Game(units, [random.randint(0, 3), random.randint(0, 3)])
    node = mcts.MCTSNode(game_state)
    tree = mcts.MCTS(game_state)
    done = False
    i = 1
    while not done:
        print("round ", i)
        action = []
        for player in range(2):
            print("player: ", player)
            node = tree.search(node, iterations=100)
            print(node.action)
            action.append(node.action)
        represent_2(game_state, action)
        game_state, reward, done = game_state.step(action)
        print("Reward: ", reward)
        total_reward = [x+y for x, y in zip(total_reward, reward)]
        i += 1
    total_reward = [x+y for x,y in zip(total_reward, game_state.reward_zone())]   
    print("Total reward: ", total_reward)
    return total_reward[1]

def test():
    japanese, allied = choose_random()
    units = [
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in japanese],
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in allied]
    ]
    game_state = new_game.Game(units, [random.randint(1, 3), random.randint(1, 3)])
    node = new_mcts.MCTSNode(game_state)
    tree = new_mcts.MCTS(game_state)
    while not node.game.is_terminal():
        new = tree.search(node, iterations=1000)
        action = new.action
        if action is None:
            print("no solution")
            break
        represent_2(node.game, action) #damage is already marked 
        node = new
        """game_state, reward, done = game_state.step(action)
        new_node = new_mcts.MCTSNode(game_state, node, action)
        node.children.append(new_node)
        node = new_node"""
    print("final pieces:", len(node.game.units[0]), len(node.game.units[1]))
    print("Zone reward: ", node.game.reward_zone()[1])
    print("Node reward:", node.value)
    return node.game.reward_zone()[1] + node.value

def test_game():
    japanese, allied = choose_random()
    player = 0
    done = False
    game_state = game.Game(japanese, allied, pv_japanese=random.randint(0, 3), pv_allied=random.randint(0, 3))
    total_reward = [0, 0]
    print(game_state.pieces)
    while not done:
        action = game_state.actions_available(player)
        game_state, reward, done = game_state.step(action[0], player)
        total_reward = [x+y for x, y in zip(total_reward, reward)]
        player = 1 - player
    print(game_state.units)
    print(total_reward)

if __name__ == "__main__":
    result = test()
    print("You win!" if result > 0 else "You did not win!")