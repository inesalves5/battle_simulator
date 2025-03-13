import mcts
import json
import game
import random
import graphviz

def visualize_mcts(root):
    """Generates a Graphviz visualization of the MCTS tree."""
    dot = graphviz.Digraph(format='png')
    
    def add_nodes_edges(node, parent_name=None, edge_label=""):
        node_label = f"Action: {node.action}\nVisits: {node.visits}\nValue: {node.value}\nWin rate: {node.win_rate:.2f}"
        node_name = str(id(node))
        dot.node(node_name, label=node_label, shape="box", style="filled", fillcolor="lightblue")
        
        if parent_name:
            dot.edge(parent_name, node_name, label=edge_label)
        
        for child in node.children:
            add_nodes_edges(child, node_name, edge_label=f"{child.action}")
    
    add_nodes_edges(root)
    return dot

def choose_random():

    with open("units.json", "r") as file:
        units = json.load(file)

    nr_japanese = random.randint(1, len(units["japanese"]))
    japanese = random.sample(units["japanese"], nr_japanese)
    
    nr_allied = random.randint(1, len(units["allied"]))
    allied = random.sample(units["allied"], nr_allied)

    return japanese, allied
    
def represent(game, action, player):
    print("Starting... type: ", game.action)
    print("action:", action)
    units = game.units[player]
    opponent = game.units[1-player]
    print("Player:", "Japanese" if player == 0 else "Allied")
    for i in range(len(action)):
        print("Unit", units[i], "attacks:", opponent[action[i]] if action[i] != None else "None")
    print("---- end ----")

def choose_action(units, pv, player):
    game_day = game.Game(units, pv, "day")
    game_night = game.Game(units, pv, "night")
    res_day = mcts_round(game_day)
    res_night = mcts_round(game_night)
    return "day" if res_day[player] > res_night[player] else "night"

def mcts_round(game_state):
    node = mcts.MCTSNode(game_state)
    tree = mcts.MCTS(game_state)
    while not (node.game.is_terminal() and node.to_play == 0):
        new = tree.search(node, iterations=100)
        #dot = visualize_mcts(root)
        #dot.render('mcts_tree', view=True) 
        action = new.action
        represent(node.game, action, node.to_play) 
        node = new
    return node.value

def main():
    japanese, allied = choose_random()
    units = [
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in japanese],
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "isElite": [area["isElite"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "type": unit["type"]} for unit in allied]
    ]
    pv = [random.randint(1, 3), random.randint(1, 3)]
    player = 0 #playing as player "japanese"
    action = choose_action(units, pv, player) #choosing best action for player
    print("Best action is:", action)
    game_state = game.Game(units, pv, action)
    result = mcts_round(game_state)
    return len(units[1]) >= len(units[0]), result[player]

if __name__ == "__main__":
    print(main())
    """
    preds, ress = 0, 0
    for _ in range(100):
        pred, res = main()
        if pred:
            preds += 1
        if res:
            ress += 1
    print("Games won by allied:", ress)  
    print("Games with allied advantage:", preds)
    """

    """
    results = 0
    predictions = 0
    for _ in range(100):
        pred, res = test()
        if pred:
            predictions += 1
        if res:
            results += 1
    print("Win percentage:", results)
    print("Percentage of games with advantage:", predictions)
    """
    
    
    """
    #testing the Class Game from game.py
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
    
    #testing the original mcts & game classes
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
                node = tree.search(node, iterations=50)
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

    #representing the mixed actions
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
    """