from itertools import permutations, product
from collections import Counter
import random
import copy
from collections import defaultdict

class Game:
    
    def __init__(self, units, pv, action, points = [0, 0]):
        #eles usam (A)S-D-V - aereo, superficie, defesa, velocidade
        self.units = units
        self.action = action
        self.pv = pv
        self.points = points

    def step(self, actions):
        reward = [0, 0]
        for player in range(2):
            action = actions[player]
            units = self.units[player]
            opponent = self.units[1-player]
            for i in range(len(action)):
                if action[i] != None and self.damage(units[i], opponent[action[i]]):
                    reward = [x+y for x, y in zip(reward, self.reward(self.action, player, opponent[action[i]]))]
        points = [x+y for x, y in zip(self.points, reward)]
        j_units = [unit for unit in self.units[0] if unit["damage"] != float("inf")]
        a_units = [unit for unit in self.units[1] if unit["damage"] != float("inf")]
        new = Game([j_units, a_units], self.pv, self.action, points)
        return new, reward, new.is_terminal()
    
    def is_terminal(self):
        aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        return len(self.units[0]) == 0 or len(self.units[1]) == 0 or not aa0 or not aa1 or all([a is None for a in aa0]) or all([a is None for a in aa1])
        
    def max_reward(self, action): #fazemos para 0 porque é igual
        rewards = [0, 0]
        for unit in self.units[0]:
            rewards = [x+y for x, y in zip(rewards, self.reward(action, 0, unit))]
        for unit in self.units[1]:
            rewards = [x+y for x, y in zip(rewards, self.reward(action, 0, unit))]
        return max(rewards)
        
    def reward(self, action, player, victim):
        if action == "day" and victim["type"] == "LBA":
            damage = 5 * (victim["attack"][0] + 1)
        elif action == "day":
            attack = 0 if victim["type"] == "BB" else victim["attack"][0]
            damage = 3 * attack + victim["defense"]
        elif victim["type"] == "LBA":
            damage = 0
        else:
            attack = victim["attack"][0] if len(victim["attack"])==1  else victim["attack"][1]
            damage = 2 * attack + victim["defense"]
        return [damage, -damage] if player == 0 else [-damage, damage]
    
    def damage(self, attacker, victim):
        hits = 0
        state = victim["damage"]
        index = 0 if self.action == "day" or len(attacker["attack"])==1 else 1
        bonus = 1 if attacker["isElite"][index] else 0
        if (state == float("inf")):
            return False
        for _ in range(attacker["attack"][index]):
            if random.randint(1, 6) + bonus >= 6:
                hits += 1
        for _ in range(hits):
            roll = random.randint(1, 6) + bonus 
            state += roll      
            if (victim["type"]=="LBA" and state >= victim["defense"]) or state > victim["defense"]: #afunda 
                state = float('inf')
                break
        victim["damage"] = state
        if hits:
            if victim["type"] == "BB" or (victim["type"] == "CV" and len(victim["attack"]) == 1):#only sea attack
                victim["isElite"][0] = False 
                if state == victim["defense"]:
                    victim["attack"][0] = 1 if victim["attack"][0] else 0
            elif victim["type"] == "CV":
                victim["isElite"][1] = False 
                if state == victim["defense"]:
                    victim["attack"] = [0, 1] if victim["attack"][1] else [0, 0]
        return hits
    
    def reward_zone(self):
        if len(self.units[0]) < len(self.units[1]):
            return [-self.pv[0], self.pv[0]]
        if len(self.units[1]) < len(self.units[0]):
            return [self.pv[1], -self.pv[1]]
        return [0, 0]

    def actions_available(self, player):
        total_units = len(self.units[player])
        if self.action == "day":
            active_units = [i for i, p in enumerate(self.units[player]) if p["type"]!="BB"]
            active_targets = [i for i, p in enumerate(self.units[1 - player])]
        else:
            active_units = [i for i, p in enumerate(self.units[player]) if p["type"]!="LBA"]
            active_targets = [i for i, p in enumerate(self.units[1 - player]) if p["type"]!="LBA"]

        if active_units and active_targets:
            attack_combinations = [list(attack) for attack in product(active_targets, repeat=len(active_units))]
        else:
            return []
        valid_actions = []
        for attack in attack_combinations:
            action = [None] * total_units  # Initialize all actions as None
            for i, unit_index in enumerate(active_units):
                action[unit_index] = attack[i]  # Assign attack targets only to active units
            valid_actions.append(action)

        return valid_actions

    def simulate_next(self, actions):
        states = defaultdict(int)
        rewards = [0, 0]

        for _ in range(50):
            new_game = Game(self.units, self.pv, self.action, self.points)  # Create a fresh game each time
            game, reward, _ = new_game.step(actions)
            states[game] += 1 
            rewards = [x + y for x, y in zip(reward, rewards)]

        avg_rewards = [x / 50 for x in rewards]
        most_visited_state = max(states, key=states.get)
        return most_visited_state, avg_rewards

     
    def get_next_state(self, actions):
        new_game = Game(self.units, self.pv, self.action, self.points)
        new_game, reward, _ = new_game.step(actions)
        return new_game, reward
    
    def japanese(self):
        return self.units[0]
    
    def allied(self):
        return self.units[1]
    
    def __eq__(self, other):
        return isinstance(other, Game) and self.units == other.units and self.pv == other.pv and self.action == other.action and self.points == other.points