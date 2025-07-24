import mcts
import json
import game
import NN
import random
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import ds

def hierarchy_pos(G, root=None, width=10, vert_gap=0.1, xcenter=0.5):
    """Compute the hierarchical layout positions for a directed tree graph."""
    pos = _hierarchy_pos(G, root, width, vert_gap, xcenter)
    return pos

def _hierarchy_pos(G, node, width=1., vert_gap=0.2, xcenter=0.5, pos=None, parent=None, level=0):
    if pos is None:
        pos = {node: (xcenter, 1 - level * vert_gap)}
    else:
        pos[node] = (xcenter, 1 - level * vert_gap)
    
    neighbors = list(G.successors(node))
    if not neighbors:
        return pos
    
    dx = width / max(1, len(neighbors))  
    next_x = xcenter - (width - dx) / 2  
    for neighbor in neighbors:
        pos = _hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, xcenter=next_x, pos=pos, parent=node, level=level+1)
        next_x += dx
    
    return pos

def visualize_mcts(root, nn=False):
    """Generates a tree visualization using NetworkX and Matplotlib without Graphviz."""
    G = nx.DiGraph()

    node_shapes = {}

    # Map from (parent, child) to label (from parent node)
    edge_labels = {}

    def add_edges(node, parent_id=None):
        node_id = id(node)
        if isinstance(node, mcts.DecisionNode) and node.player == 0:
            node_label = f"{node.visits}\n" \
                    f"{round(node.value[0], 2)}, {round(node.value[1], 2)}" 
        elif isinstance(node, mcts.ChanceNode):
            node_label = f"{node.visits}\n{round(node.value[0], 2)}, {round(node.value[1], 2)}"
        else:
            node_label = f"{node.visits}\n{round(node.value[0], 2)}, {round(node.value[1], 2)}"

        G.add_node(node_id, label=node_label, node_obj=node)  # Store node_obj for reference

        if node.game.is_terminal():
            node_shapes[node_id] = "square"
        elif isinstance(node, mcts.ChanceNode):
            node_shapes[node_id] = "star"
        elif node.player == 0:
            node_shapes[node_id] = "japanese"
        else:
            node_shapes[node_id] = "allied"

        if parent_id:
            G.add_edge(parent_id, node_id)
            child_node = node  # since `node_id` corresponds to the current (child) node
            if isinstance(child_node, mcts.DecisionNode) and child_node.player == 1:
                edge_labels[(parent_id, node_id)] = child_node.action
            elif isinstance(child_node, mcts.DecisionNode):
                edge_labels[(parent_id, node_id)] = child_node.rolls
            else:
                edge_labels[(parent_id, node_id)] = child_node.a_action

        for child in node.children:
            add_edges(child, node_id)

    add_edges(root)

    pos = hierarchy_pos(G, root=id(root))
    plt.figure(figsize=(20, 12))
    labels = nx.get_node_attributes(G, "label")

    # Draw nodes
    for node_id, shape in node_shapes.items():
        if shape == "square":
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_shape="s", node_size=200, node_color="red")
        elif shape == "star":
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_shape="*", node_size=200, node_color="yellow")
        elif shape == "japanese":
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_size=200, node_color="pink")
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_size=200, node_color="lightblue")

    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)
    nx.draw_networkx_labels(G, pos, labels, font_size=13, verticalalignment="center")

    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=10)
    plt.title(f"MCTS Tree Visualization for {root.game.action} action", fontsize=16)
    plt.axis("off")
    plt.show()

    plt.savefig(f"mcts_tree_test_case.png", format="png", bbox_inches="tight")

def get_test_case():
    with open("units_total.json", "r") as file:
        units = json.load(file)
    units = units["units"]
    allied, japanese = [], []
    for unit in units:
        valid = True
        attack = {"Air": 0, "Sea": 0}
        is_elite = {"Air": None, "Sea": None}
        for area in unit["attackDomains"]:
            attack[area["domain"]] = area["attack"]
            is_elite[area["domain"]] = area["isElite"]
        if unit["type"] not in ["LBA", "CV", "BB"]:
            valid = False
        transformed_unit = {
            "attack": [attack["Air"], attack["Sea"]],
            "isElite": [is_elite["Air"], is_elite["Sea"]],
            "defense": unit["stepsMax"], 
			"name":"USN_LBA_11thAF",
            "damage": 0 if unit["name"] in ["IJN_LBA_21stFlottilla", "USN_LBA_10thAF", "USN_LBA_11thAF"] else float("inf"),
            "type": unit["type"] ,
            "attackValue":[attack["Air"], attack["Sea"]]
        }
        if unit["name"].startswith("IJN"):
            japanese.append(transformed_unit)
        else:
            allied.append(transformed_unit)
    """
    A = units["japanese"][0]
    B = units["japanese"][2]
    C = units["japanese"][4]
    D = units["allied"][0]
    E = units["allied"][3]
    F = units["allied"][4]
    """
    game_state = game.Game(units=[japanese, allied], action="day", pv=[0, 0])
    return game_state

def choose_from_all(j, a):
    with open("units_total.json", "r") as file:
        units = json.load(file)
    units = units["units"]
    allied, japanese = [], []
    for unit in units:
        valid = True
        attack = {"Air": 0, "Sea": 0}
        is_elite = {"Air": None, "Sea": None}
        for area in unit["attackDomains"]:
            attack[area["domain"]] = area["attack"]
            is_elite[area["domain"]] = area["isElite"]
        if unit["type"] not in ["LBA", "CV", "BB"]:
            valid = False
        transformed_unit = {
            "attack": [attack["Air"], attack["Sea"]],
            "isElite": [is_elite["Air"], is_elite["Sea"]],
            "defense": unit["stepsMax"], 
            "damage": float("inf"),
            "type": unit["type"] ,
            "attackValue":[attack["Air"], attack["Sea"]]
        }
        if unit["name"].startswith("IJN"):
            japanese.append(transformed_unit)
        else:
            allied.append(transformed_unit)

    selected_japanese = random.sample(japanese, j)
    selected_allied = random.sample(allied, a)

    for unit in selected_japanese:
        unit["damage"] = 0

    for unit in selected_allied:
        unit["damage"] = 0

    return japanese, allied

def represent(game, action, player):
    print("Action type: ", game.action)
    print("action:", action)
    units = game.units[player]
    opponent = game.units[1-player]
    print("Player:", "Japanese" if player == 0 else "Allied")
    for i in range(len(action)):
        print("Unit", units[i], "attacks:", opponent[action[i]] if action[i] != None else "None")
    print("---- end ----")


def choose_action(units, pv, nn=None):
    game_day = game.Game(units=copy.deepcopy(units), pv=pv, action="day")
    max_reward_day = game_day.max_reward("day")
    game_night = game.Game(units=copy.deepcopy(units), pv=pv, action="night")
    max_reward_night = game_night.max_reward("night")
    res_day, root_day, _, _ = mcts_round(game_day, max_reward_day, nn=nn)
    res_night, root_night, _, _ = mcts_round(game_night, max_reward_night, nn=nn)
    print("Day result:", res_day)
    print("Night result:", res_night)
    action_j = "day" if res_day[0] > res_night[0] else "night"
    action_a = "day" if res_day[1] > res_night[1] else "night"
    print("Best action for Japanese is:", action_j)
    print("Best action for Allied is:", action_a)
    if action_j == action_a:
        action = action_j
    else:
        action = "day and night"
    return action

def mcts_round(game_state, max_reward, iterations=100000, nn=None):
    rewards = [0, 0]
    actions = []
    root = mcts.DecisionNode(game_state, max_reward=max_reward, player=0, root=True)
    tree = mcts.MCTS(root)
    node = tree.search(root, iterations=iterations)
    """while not node.game.is_terminal():
        if isinstance(node, mcts.ChanceNode):
            print("actions", node.j_action, node.a_action)
        node = max(node.children, key=lambda c: c.value[0] if (isinstance(c, mcts.DecisionNode) and c.player == 1)
                   else c.value[1])
        for c in node.children:
            if c.value == [0, 0]:
                print(c.game.is_terminal(), c.game.check_if_terminal(), c.game)"""
    return rewards, tree.root, actions, node

def read_units(data):
    result = []
    for unit in data:
        attack = {"Air": 0, "Sea": 0}
        is_elite = {"Air": None, "Sea": None}
        for area in unit["attackDomains"]:
            attack[area["domain"]] = area["attack"]
            is_elite[area["domain"]] = area["isElite"]
        transformed_unit = {
            "attack": [attack["Air"], attack["Sea"]],
            "isElite": [is_elite["Air"], is_elite["Sea"]],
            "defense": unit["stepsMax"], 
            "damage": 0,
            "type": unit["type"],
            "attackValue":[attack["Air"], attack["Sea"]]
        }
        result.append(transformed_unit)
    return result

def mcts_vs_nn():
    data = ds.self_play_and_generate_training_data("results.txt")
    nn = NN.ValueMLP()
    ds.train_value_net(nn, data)
    action = "day"  
    game_state = create_random_game()
    game_state_nn = copy.deepcopy(game_state)
    max_reward = game_state.max_reward(action)
    max_reward_nn = game_state_nn.max_reward(action)
    print("With NN")
    result_nn, root_nn, _, _ = mcts_round(copy.deepcopy(game_state_nn), max_reward_nn, nn=nn) 
    print("value w/ NN: ", root_nn.value)
    print("Without NN")
    result, root, _, _ = mcts_round(copy.deepcopy(game_state), max_reward) 
    print("value w/o NN: ", root.value)
    visualize_mcts(root, nn=False)
    visualize_mcts(root_nn, nn=True)    
    return result, result_nn 
    
def main():    
    game_state = get_test_case()
    action = "day" #choose_action(units, pv, nn) 
    root_night = None
    if action != "day and night":
        print("Chosen action is:", action)
        if game_state.is_terminal():
            print("Game is already over before MCTS started.")
            return [0, 0], 0, action, []
        max_reward = game_state.max_reward(action)
        result, root, actions, node = mcts_round(copy.deepcopy(game_state), max_reward) 
    else:
        print("No consensus on action - day followed by night")
        max_reward = game_state.max_reward("day")
        result, root, actions, node = mcts_round(copy.deepcopy(game_state), max_reward) 
        game_state_night = game.Game(units=node.game.units, pv=pv, action="night")
        if not game_state_night.is_terminal():
            result_night, root_night, n_actions, node = mcts_round(copy.deepcopy(game_state_night), max_reward)
            actions.append(n_actions)
            result = [x+y for x, y in zip(result, result_night)]    
        else:
            print("Game was already over before night action started.")
    visualize_mcts(root)
    if root_night:
        visualize_mcts(root_night)
    return result, root.value, action, actions

def testing():
    file = open("results.txt", "w")
    for _ in range(3):
        result, values, action = main()
        file.write(f"Result: {result} values: {values} action: {action}\n")
    file.close()
    return 

def try_nn():
    data = ds.self_play_and_generate_training_data(100)
    model = NN.ValueMLP()
    ds.train_value_net(model, data)
    for game_state, target_reward in random.sample(data, 5):
        pred = model.predict(game_state)
        print(f"Target: {target_reward:.2f}, Predicted: {pred:.2f}")

def create_random_game(action="day", j=random.randint(1, 10), a=random.randint(1, 10)):
    japanese, allied = choose_from_all(j, a)
    pv = [0, 0]  # Placeholder for player values
    game_state = game.Game(units=[japanese, allied], pv=pv, action=action)
    if game_state.is_terminal():
        return create_random_game(action)
    return game_state

def generate_eq_units():
    game_state = create_random_game()
    game_state.def_equivalent_units("day")
    game_state = create_random_game("night")
    game_state.def_equivalent_units("night")

def test_model_w_case():
    model = ds.train_value_net()
    game_test = get_test_case()
    print("now lets predict")
    value = model.predict_value(game_test)
    return value

if __name__ == "__main__":
    #start = time.time()
    #main()
    #end = time.time()
    #print("Execution time:", end - start, "seconds")
    """
    game_state = create_random_game()
    print("Initial game state: j_active:", game_state.j_active, "a_active:", game_state.a_active)
    equivalent_games = game_state.generate_equivalent_games()
    print("equivalent games generated:")
    for i in equivalent_games:
        print(i.j_active, i.a_active)
    """
    """
    v = test_model_w_case()
    print(v)
    with open("preds.txt", "a") as f:
        f.write(f"{v}\n")
    """
    main()
    #game = get_test_case()
    #eq = game.generate_equivalent_games({54:102}, {102:55})
    #print(eq)