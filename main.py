import mcts
import json
import game
import random
import copy
import networkx as nx
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

    def add_edges(node, parent_id=None):
        node_id = id(node)
        node_label = f"{len(node.game.units[0])}:{len(node.game.units[1])}\n{node.visits}\n{round(node.value[0], 4)}, {round(node.value[1], 4)}\n{node.action if isinstance(node, mcts.DecisionNode) else node.a_action}"
        G.add_node(node_id, label=node_label)

        # Determine shape: stars for chance nodes, circles otherwise
        if isinstance(node, mcts.ChanceNode):
            node_shapes[node_id] = "star"
        elif node.player == 0:
            node_shapes[node_id] = "japanese"
        else:
            node_shapes[node_id] = "allied"
        if parent_id:
            G.add_edge(parent_id, node_id)

        for child in node.children:
            add_edges(child, node_id)

    add_edges(root)

    # Use our custom tree layout
    pos = hierarchy_pos(G, root=id(root))

    # Plot the tree
    plt.figure(figsize=(20, 10))
    labels = nx.get_node_attributes(G, "label")

    # Draw nodes separately based on shape
    for node_id, shape in node_shapes.items():
        if shape == "star":
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_shape="*", node_size=200, node_color="yellow")
        elif shape == "japanese":
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_size=200, node_color="pink")
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_size=200, node_color="lightblue")
    
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)
    nx.draw_networkx_labels(G, pos, labels, font_size=13, verticalalignment="center")

    plt.title(f"MCTS Tree Visualization for {root.game.action} action")
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


def choose_action(units, pv, player):
    game_day = game.Game(units=copy.deepcopy(units), pv=pv, action="day")
    max_reward_day = game_day.max_reward("day")
    print("Max reward day:", max_reward_day)
    game_night = game.Game(units=copy.deepcopy(units), pv=pv, action="night")
    max_reward_night = game_night.max_reward("night")
    print("Max reward night:", max_reward_night)
    res_day, root_day, _ = mcts_round(game_day, max_reward_day)
    res_night, root_night, _ = mcts_round(game_night, max_reward_night)
    visualize_mcts(root_day)
    visualize_mcts(root_night)
    print("Day result:", res_day)
    print("Night result:", res_night)
    return "day" if res_day[player] > res_night[player] else "night"

def mcts_round(game_state, max_reward):
    done = False
    rewards = [0, 0]
    node = mcts.DecisionNode(game_state, max_reward=max_reward)
    tree = mcts.MCTS(node)
    actions = []
    while not done:
        j_node = tree.search(node, iterations=100000) #find action for japanese
        j_action = j_node.action
        actions.append(j_action)
        if j_action == None:
            print("No action found for japanese")
            return rewards, tree.root
        #represent(node.game, j_action, 0)
        a_node = tree.search(j_node, iterations=100000) #find action for allied
        a_action = a_node.a_action
        actions.append(a_action)
        if a_action == None:
            print("No action found for allied")
            return rewards, tree.root
        #represent(node.game, a_action, 1)
        game_state, reward, done = game_state.step([j_action, a_action])
        if game_state != node.game:
            node = mcts.DecisionNode(game_state, max_reward, parent=a_node, action=a_action, player=0, value=reward)
            a_node.add_child(node, reward)
            rewards = [x+y for x, y in zip(rewards, reward)]
    rewards = [x+y for x, y in zip(rewards, game_state.reward_zone())]
    return rewards, tree.root, actions

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
            "availability": 1, 
            "type": unit["type"]
        }
        
        result.append(transformed_unit)
    
    return result

def main():
    japanese, allied = choose_random()
    units = [read_units(japanese), read_units(allied)]
    pv = [1,1] #[random.randint(1, 3), random.randint(1, 3)]
    action_j = choose_action(units, pv, 0) #choosing best action for japanese    
    print("Best action for Japanese is:", action_j)
    action_a = choose_action(units, pv, 1) #choosing best action for allied   
    print("Best action for Allied is:", action_a)
    if action_a == action_j:
        action = action_a
    else:
        action = action_j if len(units[0]) > len(units[1]) else action_a #na verdade joga dia seguido de noite
    print("Chosen action is:", action)
    game_state = game.Game(units=units, pv=pv, action=action)
    max_reward = game_state.max_reward(action)
    print("Max reward:", max_reward)
    result, root, _ = mcts_round(copy.deepcopy(game_state), max_reward) 
    visualize_mcts(root)
    return result

def testing():
    file = open("results.txt", "w")

    japanese, allied = choose_random()
    units = [read_units(japanese), read_units(allied)]
    pv = [1, 1] #[random.randint(1, 3), random.randint(1, 3)]
    action = "day"
    game_state = game.Game(units=units, pv=pv, action=action)
    max_reward = game_state.max_reward(action)
    
    for _ in range(50):
        result, root, actions = mcts_round(copy.deepcopy(game_state), max_reward) 
        file.write(f"Root: {root.value} Result: {result}, Actions: {actions}\n")
        #visualize_mcts(root)
    file.close()
    return  

if __name__ == "__main__":
    print(main())
    
