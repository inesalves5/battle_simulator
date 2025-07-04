from itertools import permutations, product
from collections import Counter
import random
import copy
from collections import defaultdict
import tensorflow as tf

class Game:
    
    def __init__(self, units, pv, action, j_active=None, a_active=None):
        #eles usam (A)S-D-V - aereo, superficie, defesa, velocidade
        self.units = units
        self.action = action
        self.pv = pv
        if j_active and a_active :
            self.j_active = j_active
            self.a_active = a_active 
        elif action == 'day':
            self.j_active = [i for i, p in enumerate(self.units[0]) if p["type"] != 'BB' and p['damage'] != float('inf')]
            self.a_active = [i for i, p in enumerate(self.units[1]) if p['damage'] != float('inf')]
        else:
            self.j_active = [i for i, p in enumerate(self.units[0]) if p["type"] != 'LBA' and p['damage'] != float('inf')]
            self.a_active = [i for i, p in enumerate(self.units[1]) if p["type"] != 'LBA' and p['damage'] != float('inf')]
        print("j_active:", self.j_active)
        print("a_active:", self.a_active)

    def step(self, actions):
        rewards = [0, 0]
        change_gunnery, change_airstrike, change_isElite = [], [], []
        j_active, a_active = self.j_active, self.a_active
        for player in range(2):
            action = actions[player]
            units = self.units[player]
            opponent = self.units[1-player]
            for i in range(len(action)):
                if action[i] != None:
                    reward = [0, 0]
                    target = opponent[action[i]]
                    capacity = target["defense"] if target["type"] == 'LBA' else target["defense"] + 1
                    state = target['damage']
                    if state == float('inf') or capacity == 0:
                        continue          
                    prop = self.damage(units[i], target)
                    prop = min(prop, capacity - state)
                    damage = self.reward(self.action, player, target)
                    if prop != 0 and target["type"] == 'BB':
                        change_isElite.append(target)
                    if target['damage'] == target["defense"] and target["type"] == 'BB':
                        change_gunnery.append(target)
                    elif target["type"] == 'CV' and target['damage'] == target["defense"]:
                        change_airstrike.append(target)
                    if prop == capacity and state == 0:
                        reward = damage                        
                    else:
                        if target['damage'] == float('inf'):
                            prop -= 1
                            reward = [0.5*y for y in damage]
                            print("action is:", action[i])
                            j_active.remove(action[i]) if player == 1 else a_active.remove(action[i])
                        if capacity != 1:
                            proportion = 0.5 * prop / (capacity - 1)
                        else:
                            proportion = 1
                        reward = [x + proportion * y for x, y in zip(reward, damage)]
                    rewards = [x + y for x, y in zip(rewards, reward)]
        for target in change_gunnery:
            target['attack'][1] = 1 if target['attack'][1] != 0 else 0
        for unit in change_isElite:
            unit['isElite'] = [False, False]
        for target in change_airstrike:
            target['attack'][0] = 0
        return rewards, j_active, a_active

    def is_terminal(self):
        """aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        return all([u['damage'] == float('inf') for u in self.units[0]]) or all([u['damage'] == float('inf') for u in self.units[1]]) or not aa0 or not aa1 or all([a is None for a in aa0]) or all([a is None for a in aa1])
        """  
        def all_dead(units):
            return all(u['damage'] == float('inf') for u in units)

        if all_dead(self.units[0]) or all_dead(self.units[1]):
            return True
        aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        if len(aa0) == 0:
            return True
        if len(aa1) == 0:
            return True
        return False


    def max_reward(self, action): #fazemos para 0 porque é igual
        reward_j, reward_a = 0, 0
        for unit in self.units[0]:
            reward = self.reward(action, 1, unit)[1]
            reward_a += reward
        for unit in self.units[1]:
            reward_j += self.reward(action, 0, unit)[0]
        return max(reward_j, reward_a)
        
    def reward(self, action, player, victim):
        if action == 'day' and victim["type"] == 'LBA':
            damage = 5 * (victim["attackValue"][0] + 1)
        elif action == 'day':
            damage = 3 * victim["attackValue"][0] + victim["defense"]
        elif victim["type"] == 'LBA':
            return [0, 0]
        else:
            damage = 2 * victim["attackValue"][1] + victim["defense"]
        return [damage, -damage] if player == 0 else [-damage, damage]
    
    def damage(self, attacker, victim):
        damage_value = 0
        state = victim['damage']
        index = 0 if self.action == 'day' else 1
        bonus = 1 if attacker['isElite'][index] and victim["type"] != 'LBA' else 0
        if state == float('inf'):
            return 0
        roll = 0
        for _ in range(attacker['attack'][index]):
            roll += random.randint(1, 6) + bonus 
        if roll >= 6:
            damage_value = random.randint(1, 6) + bonus 
            state += damage_value  
            if (victim["type"]=='LBA' and state >= victim["defense"]) or state > victim["defense"]: #afunda
                state = float('inf')
        victim['damage'] = state
        return damage_value
    
    def reward_zone(self):
        if not all([u['damage'] == float('inf') for u in self.units[0]]) and all([u['damage'] == float('inf') for u in self.units[1]]) :
            return [self.pv[0], -self.pv[0]]
        if not all([u['damage'] == float('inf') for u in self.units[1]]) and all([u['damage'] == float('inf') for u in self.units[0]]) :
            return [-self.pv[1], self.pv[1]]
        if all([a is None for a in self.action_available(0)]) and not all([a is None for a in self.action_available(1)]):
            return [-self.pv[1], self.pv[1]]
        if all([a is None for a in self.action_available(1)]) and not all([a is None for a in self.action_available(0)]):
            return [self.pv[0], -self.pv[0]]
        return [0, 0]
    
    def actions_available(self, player):
        if not self.j_active or not self.a_active:
            return []
        units = self.j_active if player == 0 else self.a_active
        targets = self.a_active if player == 0 else self.j_active
        total_units = len(units)
        valid_actions = []

        if len(targets) >= total_units:
            # Usar permutações sem repetição
            target_combos = permutations(targets, total_units)
        else:
            # Usar produto cartesiano com repetição
            target_combos = product(targets, repeat=total_units)

        for combo in target_combos:
            action = [None] * (max(units) + 1)
            for unit_idx, target in zip(units, combo):
                action[unit_idx] = target
            valid_actions.append(action)

        return valid_actions
     
    def get_next_state(self, actions):
        game_copy = Game(copy.deepcopy(self.units), self.pv, self.action)
        reward, j_active, a_active = game_copy.step(actions)
        new_game = Game(copy.deepcopy(self.units), self.pv, self.action, j_active, a_active)
        i = 3
        while new_game == self and i > 0:
            i -= 1
            reward, j_active, a_active = new_game.step(actions)
        if new_game == self:
            return None, [0, 0]
        return new_game, reward
    
    def action_available(self, player):
        options = self.actions_available(player)
        return random.choice(options) if options else []

    def __eq__(self, other):
        if not isinstance(other, Game) or self.action != other.action or self.pv != other.pv:
            return False
        return self._eq_unit_lists(self.units[0], other.units[0]) and self._eq_unit_lists(self.units[1], other.units[1])
    
    def _eq_unit_lists(self, list1, list2):            
        if len(list1) != len(list2):
            return False
        matched = [False] * len(list2)
        for u1 in list1:
            found_match = False
            for i, u2 in enumerate(list2):
                if not matched[i] and self._eq_units(u1, u2):
                    matched[i] = True
                    found_match = True
                    break
            if not found_match:
                return False
        return True
    
    def _eq_units(self, u1, u2):
        return all(u1['attack'][i] == u2['attack'][i] for i in range(len(u1['attack']))) and \
                    all(u1['isElite'][i] == u2['isElite'][i] for i in range(len(u1['isElite']))) and  \
                    u1["defense"] == u2["defense"] and \
                    u1['damage'] == u2['damage'] and \
                    u1["type"] == u2["type"] and \
                    all(u1["attackValue"][i] == u2["attackValue"][i] for i in range(len(u1["attackValue"])))
    
    def __str__(self):
        return f"{[unit['damage'] for unit in self.units[0]], [unit['damage'] for unit in self.units[1]]}"

    def encode(self):
        features = []
        for player_units in self.units:
            for unit in player_units:
                if unit['damage'] == float('inf'):
                    t = 1.0
                elif unit["type"] == 'LBA':
                    t = unit['damage'] / unit["defense"]
                else:
                    t = unit['damage'] / (unit["defense"] + 1)
                features.append(t)
        return tf.convert_to_tensor(features, dtype=tf.float32)