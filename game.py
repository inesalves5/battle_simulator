from itertools import permutations, product
from collections import Counter
import random
import copy

class Game:
    
    def __init__(self, japanese, allied, pv_japanese, pv_allied):
        #eles usam (A)S-D-V - aerio, superficie, defesa, velocidade
        self.units = [[{"attack": [area["attack"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "canFly": unit["canFly"], "type": unit["type"]} for unit in japanese],\
            [{"attack": [area["attack"] for area in unit["attackDomains"]], "defense": unit["stepsMax"], "damage": 0, "canFly": unit["canFly"], "type": unit["type"]} for unit in allied]]
        self.pieces = [len(self.units[0]), len(self.units[1])]
        self.done = False
        self.action = "day" if random.randint(0, 1) == 0 else "night"
        self.pv = [pv_japanese, pv_allied]
        self.points = [0, 0]

    def step(self, action, player):
        units = self.units[player]
        opponent = self.units[1-player]
        reward = [0, 0]
        for i in range(len(action)):
            if (action[i] != None) and (self.damage(units[i], opponent[action[i]], player)):
                reward = [x+y for x, y in zip(reward, self.reward(player, opponent[action[i]]))]
        self.done = self.end()
        self.points = [x+y for x, y in zip(self.points, self.reward_zone())]
        return self, reward, self.done
    
    def end(self):
        return self.pieces[0] == 0 or self.pieces[1] == 0
    
    def reward(self, player, victim):
        if self.action == "day":
            mul = 2 * victim["attack"][0]
        else:
            mul = (3 * victim["attack"][1]) if victim["canFly"] else (3 * victim["attack"][0])
        damage = mul + victim["defense"]
        return [damage, -damage] if player == 0 else [-damage, damage]
    
    def damage(self, attacker, victim, player):
        hits = 0
        state = victim["damage"]
        if (state == float("inf")):
            return False
        for _ in range(attacker["attack"][0] if self.action == "day" else (attacker["attack"][1] if attacker["canFly"] else attacker["attack"][0])):
            if random.randint(1, 6) == 6:
                hits += 1
        #print("HITS: ", hits)
        for _ in range(hits):
            roll = random.randint(1, 6)  
            state += roll      
            if state > victim["defense"]: #afunda
                state = float('inf')
                self.pieces[1-player] -= 1
                break        
        victim["damage"] = state
        return hits > 0
    
    def reward_zone(self):
        if self.pieces[0] == 0:
            return self.pv[0], -self.pv[0]
        if self.pieces[1] == 0:
            return -self.pv[1], self.pv[1]
        return (0, 0)
    
    def actions_available(self, player):
        pecas_jogador_ativas = [i for i, p in enumerate(self.units[player]) if p["damage"] < float("inf")]
        alvos_possiveis = [i for i, p in enumerate(self.units[1-player]) if p["damage"] < float("inf")]
        if not alvos_possiveis:
            return []

        num_pecas_jogador = len(pecas_jogador_ativas)
        num_pecas_oponente = len(alvos_possiveis)
        if num_pecas_jogador > num_pecas_oponente:
            combinacoes_ataques = [
                list(ataque) for ataque in product(alvos_possiveis, repeat=num_pecas_jogador)
                if all(Counter(ataque)[alvo] >= 1 for alvo in alvos_possiveis)
            ]
        else:
            combinacoes_ataques = list(permutations(alvos_possiveis, num_pecas_jogador))

        acoes_validas = []
        for ataque in combinacoes_ataques:
            acao = [None] * len(self.units[player])
            for i, indice_peca in enumerate(pecas_jogador_ativas):
                acao[indice_peca] = ataque[i]  
            acoes_validas.append(acao)  
        return acoes_validas
     
    def take_action(self, action, player):
        new_game = copy.deepcopy(self)
        new_game, _, _ = new_game.step(action, player)
        return new_game