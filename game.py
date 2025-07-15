from itertools import permutations, product
from collections import Counter
import random
import copy
from collections import defaultdict
import tensorflow as tf
import json

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
        self.terminal = None

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

    def check_if_terminal(self):
        """aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        return all([u['damage'] == float('inf') for u in self.units[0]]) or all([u['damage'] == float('inf') for u in self.units[1]]) or not aa0 or not aa1 or all([a is None for a in aa0]) or all([a is None for a in aa1])
        if not self.j_active or not self.a_active:
            return True
        aa0 = self.actions_available(0)
        aa1 = self.actions_available(1)
        if aa0 == []:
            return True
        if aa1 == []:
            return True
        return False
        """  
        terminal = not self.actions_available(0) or not self.actions_available(1) 
        self.terminal = terminal
        return terminal

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
            changed = False
            for unit_idx, target in zip(units, combo):
                changed = True
                action[unit_idx] = target
            if changed:
                valid_actions.append(action)

        return valid_actions
    
    def is_terminal(self):
        if self.terminal is None:
            return self.check_if_terminal()
        return self.terminal

    def get_next_state(self, actions):
        game_copy = Game(self.units.copy(), self.pv, self.action, self.j_active, self.a_active)
        reward, j_active, a_active = game_copy.step(actions)
        new_game = Game(game_copy.units, game_copy.pv, game_copy.action, j_active, a_active)
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
        #return self._eq_unit_lists(self.units[0], other.units[0]) and self._eq_unit_lists(self.units[1], other.units[1])
        jap_1, jap_2 = self.j_active, other.j_active
        all_1, all_2 = self.a_active, other.a_active
        return self.eq_damages(other)

    def _eq_unit_lists(self, list1, list2):            
        for i in range(len(list1)):
            if not (list1[i]["damage"] == list2[i]["damage"] and \
                all(list1[i]["attack"][l]==list2[i]["attack"][l] for l in range(2)) and \
                all(list1[i]["isElite"][r]==list2[i]["isElite"][r] for r in range(2))):
                return False
        return True
    
    def eq_damages(self, other):
        jap_1, jap_2 = self.j_active, other.j_active
        all_1, all_2 = self.a_active, other.a_active
        if jap_1 != jap_2 or all_1 != all_2:
            return False
        for i in jap_1:
            if self.units[0][i]["damage"] != other.units[0][i]["damage"]:
                return False
        for j in all_1:
            if self.units[1][j]["damage"] != other.units[1][j]["damage"]:
                return False
        return True

    def _eq_units(self, u1, u2):
        if self.action == 'night' and u1['type'] == 'LBA' and u2['type'] == 'LBA':
            return True  # LBA units are not compared in night mode bc they can't attack or be attacked
        if self.action == 'day' and u1['type'] == 'BB' and u2['type'] == 'BB':
            return u1['defense'] == u2['defense']  # BB units are not compared in day mode bc they can't attack
        return all(u1['attack'][i] == u2['attack'][i] for i in range(2)) and \
                    all(u1['isElite'][i] == u2['isElite'][i] for i in range(2)) and  \
                    u1["defense"] == u2["defense"] and \
                    u1['damage'] == u2['damage'] and \
                    u1["type"] == u2["type"] and \
                    all(u1["attackValue"][i] == u2["attackValue"][i] for i in range(len(u1["attackValue"])))
    
    def __str__(self):
        return f"{[unit['damage'] for unit in self.units[0]], [unit['damage'] for unit in self.units[1]]}"

    def encode(self):
        features = []
        for player in range(len(self.units)):
            for unit in self.units[player]:
                if unit['damage'] == float('inf'):
                    t = 0.0
                elif unit["damage"] == 0:
                    t = 1.0
                elif unit["type"] == 'LBA':
                    t = 1 - (unit['damage'] / unit["defense"])
                else:
                    t = 1 - (unit['damage'] / (unit["defense"] + 1))
                features.append(t)
        return tf.convert_to_tensor(features, dtype=tf.float32)

    def def_equivalent_units(self, action):
        with open(f'equivalent_units_{action}.json', 'w') as f:
            data = defaultdict(list)
            for player in range(2):
                for idx_unit, unit in enumerate(self.units[player]):
                    equivalents = []
                    for idx, other_unit in enumerate(self.units[player]):
                        if idx != idx_unit and self._eq_units(unit, other_unit):
                            equivalents.append(idx)
                    data[idx_unit if player == 0 else idx_unit + len(self.units[0])] = equivalents
            json.dump(data, f)
    
    def generate_equivalent_games_1(self):
        with open(f'equivalent_units_{self.action}.json', 'r') as f:
            data = json.load(f)
        equivalent_games = []
        for player in range(2):
            for idx_unit, unit in enumerate(self.units[player]):
                if unit['damage'] != float('inf'):
                    options = data.get(str(idx_unit if player == 0 else idx_unit + len(self.units[0])), [])
                    for option in options:
                        new_units = [u.copy() for u in self.units[player]]
                        new_units[option] = self.units[player][idx_unit].copy()
                        new_units[idx_unit]["damage"] = float("inf")
                        equivalent_games.append(Game(units=[new_units, self.units[1-player]], action=self.action, pv=self.pv))
        return equivalent_games

    def generate_equivalent_games(self):
        with open(f'equivalent_units_{self.action}.json', 'r') as f:
            data = json.load(f)
            equivalent_games = []
            seen_keys = set()
            unit_options = []
            idx_map = [] 
            for player in range(1):
                for idx_unit, unit in enumerate(self.units[player]):
                    if unit['damage'] != float('inf'):
                        key = str(idx_unit if player == 0 else idx_unit + len(self.units[0]))
                        equivalents = data.get(key, [])
                        valid_equivalents = []
                        for eq_idx in equivalents:
                            if eq_idx < len(self.units[0]):
                                valid_equivalents.append(eq_idx)
                            elif eq_idx >= len(self.units[0]):
                                valid_equivalents.append(eq_idx)
                        unit_options.append([idx_unit] + valid_equivalents)
                        idx_map.append(idx_unit)
        for replacement_indices in product(*unit_options):
            canonical_key = frozenset(replacement_indices)
            if canonical_key in seen_keys:
                continue
            seen_keys.add(canonical_key)
            new_units_j = [unit.copy() for unit in self.units[0]]
            new_units_a = [unit.copy() for unit in self.units[1]]
            j_active = self.j_active.copy()
            a_active = self.a_active.copy()
            for orig_idx, repl_idx in zip(idx_map, replacement_indices):
                if repl_idx != orig_idx:
                    if repl_idx in j_active and orig_idx in j_active:
                        new_units_j[repl_idx] = self.units[0][orig_idx].copy()
                        new_units_j[orig_idx] = self.units[0][repl_idx].copy()
                    elif repl_idx in j_active and orig_idx in a_active:
                        new_units_j[repl_idx] = self.units[0][orig_idx].copy()
                        new_units_a[orig_idx-len(self.units[0])] = self.units[1][repl_idx].copy()
                    elif repl_idx in a_active and orig_idx in a_active:
                        new_units_a[repl_idx] = self.units[1][orig_idx-len(self.units[0])].copy()
                        new_units_a[orig_idx] = self.units[1][repl_idx-len(self.units[0])].copy()
                    elif repl_idx in a_active and orig_idx in j_active:
                        new_units_a[repl_idx] = self.units[1][orig_idx].copy()
                        new_units_j[orig_idx-len(self.units[0])] = self.units[0][repl_idx].copy()
                    elif repl_idx < len(self.units[0]) and repl_idx not in j_active and orig_idx in j_active:  #casos de replicas inativas
                        new_units_j[repl_idx] = self.units[0][orig_idx].copy()
                        new_units_j[orig_idx]['damage'] = float('inf')
                        j_active.remove(orig_idx)
                        j_active.append(repl_idx)
                    elif repl_idx >= len(self.units[0]) and repl_idx not in a_active and orig_idx in a_active:
                        new_units_a[repl_idx-len(self.units[0])] = self.units[1][orig_idx-len(self.units[0])].copy()
                        a_active.remove(orig_idx)
                        a_active.append(repl_idx-len(self.units[0]))
                        new_units_a[orig_idx-len(self.units[0])]['damage'] = float('inf')
            new_game = Game(units=[new_units_j, new_units_a], action=self.action, pv=self.pv, j_active=j_active, a_active=a_active)
            equivalent_games.append(new_game)    
        return equivalent_games