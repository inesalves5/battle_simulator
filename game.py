from itertools import permutations, product
from collections import Counter
import random
import copy
from collections import defaultdict

class Game:
    
    def __init__(self, units, pv, action):
        #eles usam (A)S-D-V - aereo, superficie, defesa, velocidade
        self.units = units
        self.action = action
        self.pv = pv

    def step(self, actions):
        game_copy = copy.deepcopy(self)
        reward = [0, 0]
        for player in range(2):
            action = actions[player]
            units = game_copy.units[player]
            opponent = game_copy.units[1-player]
            for i in range(len(action)):
                if action[i] != None:
                    initial_availability = opponent[action[i]]["availability"]
                    state = game_copy.damage(units[i], opponent[action[i]])
                    damage = game_copy.reward(self.action, player, opponent[action[i]])
                    final_availability = opponent[action[i]]["availability"]
                    if final_availability == 0:
                        reward = [x + y for x, y in zip(reward, damage)]
                    else:
                        proportion = (initial_availability - final_availability) / initial_availability
                        reward = [x + proportion * y for x, y in zip(reward, damage)]
        j_units = [unit for unit in game_copy.units[0] if unit["availability"] != 0]
        a_units = [unit for unit in game_copy.units[1] if unit["availability"] != 0]
        new = Game([j_units, a_units], self.pv, self.action)
        return new, reward, new.is_terminal()
    
    def is_terminal(self):
        aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        return len(self.units[0]) == 0 or len(self.units[1]) == 0 or not aa0 or not aa1 or all([a is None for a in aa0]) or all([a is None for a in aa1])
        
    def max_reward(self, action): #fazemos para 0 porque é igual
        reward_j, reward_a = 0, 0
        for unit in self.units[0]:
            reward = self.reward(action, 1, unit)[1]
            reward_a += reward
        for unit in self.units[1]:
            reward_j += self.reward(action, 0, unit)[0]
        return max(reward_j, reward_a)
        
    def reward(self, action, player, victim):
        if action == "day" and victim["type"] == "LBA":
            damage = 5 * (victim["attack"][0] + 1)
        elif action == "day":
            damage = 3 * victim["attack"][0] + victim["defense"]
        elif victim["type"] == "LBA":
            return [0, 0]
        else:
            damage = 2 * victim["attack"][1] + victim["defense"]
        return [damage, -damage] if player == 0 else [-damage, damage]
    
    def damage(self, attacker, victim):
        hits = 0
        state = victim["damage"]
        rolls = 0
        index = 0 if self.action == "day" else 1
        bonus = 1 if attacker["isElite"][index] else 0
        if (state == float("inf")):
            return 0
        for _ in range(attacker["attack"][index]):
            if random.randint(1, 6) + bonus >= 6:
                hits += 1
        for _ in range(hits):
            roll = random.randint(1, 6) + bonus 
            rolls += roll
            state += roll      
            if (victim["type"]=="LBA" and state >= victim["defense"]) or state > victim["defense"]: #afunda
                victim["availability"] -= round(state / victim["defense"], 2)
                victim["availability"] = max(victim["availability"], 0)
                state = float('inf')
                break
        victim["damage"] = state
        if state != float("inf"):
            victim["availability"] -= round(state / victim["defense"], 2)
            victim["availability"] = max(victim["availability"], 0)
        if hits:
            victim["isElite"] = [False, False] 
            if state == victim["defense"]:
                victim["attack"] = [0, 1] if victim["attack"][1] else [0, 0]
        return rolls
    
    def reward_zone(self):
        if len(self.units[0]) and len(self.units[1]) == 0:
            return [self.pv[0], -self.pv[0]]
        if len(self.units[1]) and len(self.units[0]) == 0:
            return [-self.pv[1], self.pv[1]]
        if all([a is None for a in self.actions_available(0)]) and not all([a is None for a in self.actions_available(1)]):
            return [-self.pv[1], self.pv[1]]
        if all([a is None for a in self.actions_available(1)]) and not all([a is None for a in self.actions_available(0)]):
            return [self.pv[0], -self.pv[0]]
        return [0, 0]
    
    def actions_available(self, player):
        total_units = len(self.units[player])

        if self.action == "day":
            active_units = [i for i, p in enumerate(self.units[player]) if p["type"] != "BB"]
            active_targets = [i for i, _ in enumerate(self.units[1 - player])]
        else:
            active_units = [i for i, p in enumerate(self.units[player]) if p["type"] != "LBA"]
            active_targets = [i for i, p in enumerate(self.units[1 - player]) if p["type"] != "LBA"]

        if not active_units or not active_targets:
            return []

        valid_actions = []

        if len(active_targets) >= len(active_units):
            for perm in permutations(active_targets, len(active_units)):
                action = [None] * total_units
                for unit_idx, target in zip(active_units, perm):
                    action[unit_idx] = target
                valid_actions.append(action)

        return valid_actions
     
    def get_next_state(self, actions):
        new_game, reward, _ = self.step(actions)
        return new_game, reward
    
    def __eq__(self, other):
        return isinstance(other, Game) and self.units == other.units and self.pv == other.pv and self.action == other.action 