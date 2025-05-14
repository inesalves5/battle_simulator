import mcts
import json
import game
import random
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

def visualize_mcts(root):
    """Generates a tree visualization using NetworkX and Matplotlib without Graphviz."""
    G = nx.DiGraph()
    node_shapes = {}

    # Map from (parent, child) to label (from parent node)
    edge_labels = {}

    def add_edges(node, parent_id=None):
        node_id = id(node)
        if isinstance(node, mcts.DecisionNode) and node.player == 0:
            node_label = f"{[u['damage'] if u['damage'] != float("inf") else '-' for u in node.original_game.units[0]]}" \
                        f"{[u['damage'] if u['damage'] != float("inf") else '-' for u in node.original_game.units[1]]}\n" \
                        f"{node.visits}\n{round(node.value[1], 2)}"
        elif isinstance(node, mcts.ChanceNode):
            node_label = f"{node.visits}\n{round(node.value[1], 2)}"
        else:
            node_label = f"{node.visits}\n{round(node.value[0], 2)}"
        
        G.add_node(node_id, label=node_label, node_obj=node)  # Store node_obj for reference

        if node.original_game.is_terminal():
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
            if isinstance(child_node, mcts.DecisionNode):
                edge_labels[(parent_id, node_id)] = child_node.action
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

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=10)
    plt.title(f"MCTS Tree Visualization for {root.original_game.action} action", fontsize=16)
    plt.axis("off")
    plt.show()
    
def choose_random():
    with open("units.json", "r") as file:
        units = json.load(file)

    nr_japanese = random.randint(1, 2)
    #japanese = random.sample(units["japanese"], nr_japanese)
    
    nr_allied = random.randint(1, 2)
    #allied = random.sample(units["allied"], nr_allied)
    japanese = [units["japanese"][0], units["japanese"][2]]
    allied = [units["allied"][3], units["allied"][0]]
    return japanese, copy.deepcopy(japanese)
    

def represent(game, action, player):
    print("Action type: ", game.action)
    print("action:", action)
    units = game.units[player]
    opponent = game.units[1-player]
    print("Player:", "Japanese" if player == 0 else "Allied")
    for i in range(len(action)):
        print("Unit", units[i], "attacks:", opponent[action[i]] if action[i] != None else "None")
    print("---- end ----")


def choose_action(units, pv):
    game_day = game.Game(units=copy.deepcopy(units), pv=pv, action="day")
    max_reward_day = game_day.max_reward("day")
    game_night = game.Game(units=copy.deepcopy(units), pv=pv, action="night")
    max_reward_night = game_night.max_reward("night")
    res_day, root_day, _, _ = mcts_round(game_day, max_reward_day)
    res_night, root_night, _, _ = mcts_round(game_night, max_reward_night)
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

def mcts_round_right(game_state, max_reward, iterations=1):
    rewards = [0, 0]
    actions = []
    root = mcts.DecisionNode(game_state, max_reward=max_reward)
    tree = mcts.MCTS(root)
    current_node = root
    while not current_node.original_game.is_terminal():
        j_node = tree.search(current_node, iterations=iterations)
        j_action = j_node.action
        actions.append(j_action)
        if j_action is None:
            print("No action found for Japanese")
            return rewards, tree.root, actions, current_node
        #represent(node.game, j_action, 0)

        a_node = tree.search(j_node, iterations=iterations)
        a_action = a_node.a_action
        actions.append(a_action)
        if a_action is None:
            print("No action found for Allied")
            return rewards, tree.root, actions, current_node
        #represent(node.game, a_action, 1)

        current_node, reward = a_node.expand()
        tree.backpropagate(current_node, reward)
        rewards = [x + y for x, y in zip(rewards, reward)]

    rewards = [x + y for x, y in zip(rewards, game_state.reward_zone())]
    return rewards, tree.root, actions, current_node

def mcts_round(game_state, max_reward, iterations=2):
    rewards = [0, 0]
    actions = []
    root = mcts.DecisionNode(game_state, max_reward=max_reward, player=0)
    tree = mcts.MCTS(root)
    node = root
    while not node.original_game.is_terminal():
        node = tree.search(node, iterations=iterations)
        if isinstance(node, mcts.ChanceNode):
            actions += [node.a_action, node.j_action]
            node, reward = node.expand()
            rewards = [x + y for x, y in zip(rewards, reward)]
        elif isinstance(node, mcts.DecisionNode) and node.action is None:
            print(f"No action found for {"Japanese" if node.player == 0 else "Allied"}")
            return rewards, tree.root, actions, node
    rewards = [x + y for x, y in zip(rewards, node.original_game.reward_zone())]
    tree.backpropagate(node, rewards)
    print("Final rewards:", rewards)
    return rewards, tree.root, actions, node

def read_units(data):
    result = []
    for unit in data:
        attack = {"Air": 0, "Sea": 0}
        is_elite = {"Air": None, "Sea": None}
        for area in unit["attackDomains"]:
            attack[area["domain"]] = area["attack"]
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

def main():
    japanese, allied = choose_random()
    units = [read_units(japanese), read_units(allied)]
    pv = [0,0] #[random.randint(1, 3), random.randint(1, 3)]
    action = "day" #choose_action(units, pv) 
    root_night = None
    if action != "day and night":
        print("Chosen action is:", action)
        game_state = game.Game(units=units, pv=pv, action=action)
        max_reward = game_state.max_reward(action)
        result, root, actions, node = mcts_round(copy.deepcopy(game_state), max_reward) 
    else:
        print("No consensus on action - day followed by night")
        game_state = game.Game(units=units, pv=pv, action="day")
        max_reward = game_state.max_reward("day")
        result, root, actions, node = mcts_round(copy.deepcopy(game_state), max_reward) 
        game_state_night = game.Game(units=node.game.units, pv=pv, action="night")
        if not game_state_night.is_terminal():
            result_night, root_night, n_actions, node = mcts_round(copy.deepcopy(game_state_night), max_reward, True)
            actions.append(n_actions)
            result = [x+y for x, y in zip(result, result_night)]    
        else:
            print("Game was already over before night action started.")
    print("Units at end: ", [u["damage"] for u in node.game.units[0]], [u["damage"] for u in node.game.units[1]])
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

if __name__ == "__main__":
    print(main())
