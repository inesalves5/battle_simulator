import mcts
import json
import game
import random

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
        print("Unit", units[i], ":", opponent[action[i]])
    print("---- end ----")

def main():
    total_reward = [0, 0]
    japanese, allied = choose_random()
    player = 0 # 0 for Japanese, 1 for Allied
    game_state = game.Game(japanese, allied, pv_japanese=random.randint(0, 3), pv_allied=random.randint(0, 3))
    tree = mcts.MCTS_Node(game_state, player=player)
    done = False
    print("started")
    while not done:
        action = tree.best_action()
        represent(game_state, action, player)
        game_state, reward, done = game.step(action, player)
        print("Reward: ", reward)
        total_reward = [x+y for x, y in zip(total_reward, reward)]
        player = 1 - player
        total_reward = [x+y for x,y in zip(total_reward, game_state.reward_zone())]
        if done:
            print("Total reward: ", total_reward)
            break
    return total_reward[1]
    
    
def test():
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
    print(game_state.pieces)
    print(total_reward)

if __name__ == "__main__":
    main()