from itertools import permutations, product
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
            if self.damage(units[i], opponent[action[i]], player):
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
            mul = 3 * victim["attack"][1] if victim["canFly"] else 3 * victim["attack"][0] 
        damage = mul + victim["defense"]
        return [-damage, damage] if player == 0 else [damage, -damage]
    
    def damage(self, attacker, victim, player):
        hits = 0
        state = -float('inf')
        for _ in range(attacker["attack"][0] if self.action == "day" else (attacker["attack"][1] if attacker["canFly"] else 3 * attacker["attack"][0])):
            if random.randint(1, 6) == 6:
                hits += 1
        for _ in range(hits):
            roll = random.randint(1, 6)    
            if roll > victim["defense"]: #afunda
                state = float('inf')
                self.pieces[1-player] -= 1
                break
            else:
                state += roll          
            victim["damage"] += state
        return state < 0
    
    def reward_zone(self):
        if self.pieces[0] == 0:
            return self.pv[0], -self.pv[0]
        if self.pieces[1] == 0:
            return -self.pv[1], self.pv[1]
        return (0, 0)
    
    def actions_available(self, player):
        # Índices das peças ativas do jogador e do oponente
        pecas_jogador_ativas = [i for i, p in enumerate(self.units[player]) if p["damage"] < float("inf")]
        alvos_possiveis = [i for i, p in enumerate(self.units[1-player]) if p["damage"] < float("inf")]

        # Se não houver alvos ativos, não há ações possíveis
        if not alvos_possiveis:
            return []

        # Se o jogador tem mais peças ativas que o oponente, permitir ataques repetidos
        if len(pecas_jogador_ativas) > len(alvos_possiveis):
            combinacoes_ataques = list(product(alvos_possiveis, repeat=len(pecas_jogador_ativas)))
        else:
            combinacoes_ataques = list(permutations(alvos_possiveis, len(pecas_jogador_ativas)))

        acoes_validas = []
        for ataque in combinacoes_ataques:
            acao = [-1] * len(self.units[player])  # Começamos com todas as peças como inativas (-1)
            for i, indice_peca in enumerate(pecas_jogador_ativas):
                acao[indice_peca] = ataque[i]  # Definimos o ataque dessa peça
            acoes_validas.append(acao)  # Adicionamos como lista

        return acoes_validas
     
    def take_action(self, action, player):
        new_game = copy.deepcopy(self)
        new_game, _, _ = new_game.step(action, player)
        return new_game