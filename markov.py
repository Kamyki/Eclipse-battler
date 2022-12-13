import numpy as np
from discreteMarkovChain import markovChain

from typing import Any, Dict, List, Tuple, NamedTuple, FrozenSet
from collections import Counter, defaultdict
from enum import Enum
from math import prod
from dataclasses import dataclass, field, replace, asdict
from itertools import combinations, product, chain
from copy import deepcopy, copy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres, spsolve
from scipy.sparse import csgraph
from scipy import sparse
import numpy as np
import time



class ShipClass(Enum):
    INTNRCEPTOR = 1
    CRUSER = 2
    DREADNOUGHT = 3
    STARBASE = 4

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f'{self.name}'


@dataclass(frozen=True)
class ShipState:
    fired_cannons: bool = field(repr=False, default=False)
    fired_missles: bool = field(repr=False, default=False)
    regen_used: bool = field(repr=False, default=False)
    count: int = 1
    damage: int = 0


class Firepower(NamedTuple):
    one: int
    two: int
    three: int
    four: int

    def has_power(self):
        return sum(self) != 0

    def fire(self):
        dices = [0, 0, 0, 0]
        for (i, power) in enumerate(self):
            dices[i] = Counter(map(lambda x: frozenset(x.items()), map(
                dict, map(Counter, product(range(1, 7),  repeat=power)))))

        p = {k: prod([dices[i][k[i]] for i in range(4)])
             for k in product(*dices)}
        return p

    def count(self, c):
        return Firepower(self.one * c, self.two*c, self.three*c, self.four*c)


@dataclass(frozen=True)
class Ship:
    cannons: Firepower = field(repr=False)
    missles: Firepower = field(repr=False)
    initiative: int = field(repr=False)
    regen: int = field(repr=False)
    armor: int = field(repr=False)
    computer: int = field(repr=False)
    shield: int = field(repr=False)
    defender: bool
    ship_class: ShipClass

    @staticmethod
    def cruser(update={}, defender=True):
        STATS = {'ship_class': ShipClass.CRUSER,
                 'cannons':  Firepower(1, 0, 0, 0),
                 'missles':  Firepower(0, 0, 0, 0),
                 'initiative': 2,
                 'regen': 0,
                 'armor': 1,
                 'computer': 1,
                 'shield': 0}

        return Ship(**STATS | update, defender=defender)

    @staticmethod
    def interceptor(update={}, defender=True):
        STATS = {'ship_class': ShipClass.INTNRCEPTOR,
                 'cannons': Firepower(1, 0, 0, 0),
                 'missles': Firepower(0, 0, 0, 0),
                 'initiative': 3,
                 'regen': 0,
                 'armor': 0,
                 'computer': 0,
                 'shield': 0}

        return Ship(**STATS | update, defender=defender)

    @staticmethod
    def draednought(update={}, defender=True):
        STATS = {'ship_class': ShipClass.DREADNOUGHT,
                 'cannons': Firepower(2, 0, 0, 0),
                 'missles': Firepower(0, 0, 0, 0),
                 'initiative': 1,
                 'regen': 0,
                 'armor': 2,
                 'computer': 1,
                 'shield': 0}

        return Ship(**STATS | update, defender=defender)

    @staticmethod
    def starbase(update={}, defender=True):
        STATS = {'ship_class': ShipClass.STARBASE,
                 'cannons': Firepower(1, 0, 0, 0),
                 'missles': Firepower(0, 0, 0, 0),
                 'initiative': 4,
                 'regen': 0,
                 'armor': 2,
                 'computer': 1,
                 'shield': 0}

        return Ship(**STATS | update, defender=defender)

    def key(self):
        return (self.initiative, self.defender)

    def has_regen(self):
        return self.regen != 0

    def fire_missles(self, sstate: ShipState, state: FrozenSet):
        opposing_fleet =  state[0] if self.defender else state[1]
        my_fleet = dict(state[1] if self.defender else state[0])
        
        my_fleet[self] = replace(sstate, fired_missles=True)
        if self.defender:
            make_state = lambda x: (x, frozenset(my_fleet.items()))
        else:
            make_state = lambda x: (frozenset(my_fleet.items()), x)
        count = sstate.count
        dices = self.missles.count(count).fire()
        results = defaultdict(int)
        for dice_set in dices.items():
            for (k, v) in assign_dices(self, opposing_fleet, sum(self.missles) * count, dice_set).items():
                results[make_state(k)] += v
        return results

    def fire_cannons(self, sstate: ShipState, state) -> Dict[Any, int]:
        opposing_fleet =  state[0] if self.defender else state[1]
        my_fleet = dict(state[1] if self.defender else state[0])
        my_fleet[self] = replace(sstate, fired_cannons=True)
        if self.defender:
            make_state = lambda x: (x, frozenset(my_fleet.items()))
        else:
            make_state = lambda x: (frozenset(my_fleet.items()), x)
        count = sstate.count
        dices = self.cannons.count(count).fire()
        results = defaultdict(int)
        for dice_set in dices.items():
            for (k, v) in assign_dices(self, opposing_fleet, sum(self.cannons) * count, dice_set).items():
                results[make_state(k)] += v
        return results

    def regen_damage(self, sstate, state):
        opposing_fleet =  state[0] if self.defender else state[1]
        my_fleet = state[1] if self.defender else state[0]
        my_fleet = dict(my_fleet)
        my_fleet[self] = replace(sstate, damage=max(0, sstate.damage-self.regen), regen_used=True)
        if self.defender:
            new_state = (opposing_fleet, frozenset(my_fleet.items()))
        else:
            new_state = (frozenset(my_fleet.items()), opposing_fleet)

        result = defaultdict(float, {new_state: 1.0})
        return result

    def is_hit(self, dice, comp):
        return dice == 6 or (dice + comp - self.shield >= 6 and dice != 0)

    def min_dice(self, comp):
        return 6 - comp - self.shield


State = tuple[FrozenSet[tuple[Ship, ShipClass]], FrozenSet[tuple[Ship, ShipClass]]]



def assign_dices(ship, opposing_fleet, dice_count, dices):
    succ_states = defaultdict(int)
    opposing_fleet = dict(opposing_fleet)
    chosen_targets = []
    for (dice, power, count) in next_dice(dices):
        targets = list(map(lambda x: (x, (dice, power+1)),filter(lambda x: x.is_hit(dice, ship.computer), opposing_fleet)))
        if len(targets) > 0:
            chosen_targets.append(product(targets, repeat=count))

    if len(chosen_targets) == 0:
        succ_states[frozenset(opposing_fleet.items())] += dices[1]
    else:    
        for ships in product(*chosen_targets):
            fleet = deepcopy(opposing_fleet)
            for ((target, (dice, power))) in ships[0]:
                if fleet[target].count:
                    fleet[target] = damage(
                        target, fleet[target], power)
            succ_states[frozenset(
                filter(lambda x: x[1].count, fleet.items()))] += dices[1]
    
    return succ_states


def next_dice(dices):
    for power in range(4):
        for (dice, count) in dices[0][power]:
            yield (dice, power+1, count)
        

def damage(ship, sstate, power) -> ShipState:
    if power > ship.armor and sstate.count > 1:
        s = replace(sstate, count=sstate.count-1, damage=sstate.damage)
        return s
    if sstate.damage + power > ship.armor:
        s = replace(sstate, count=sstate.count-1, damage=0)
        return s
    else:
        s = replace(sstate, damage=sstate.damage+power)
        return s


def reset(state):
    a = frozenset((k, replace(v, fired_cannons=False, regen_used=False))
                  for (k, v) in state[0])
    d = frozenset((k, replace(v, fired_cannons=False, regen_used=False))
                  for (k, v) in state[1])
    return a, d


def is_terminal(state):
    if sum(v.count for (k, v) in state[0]) == 0:
        return (True, True)
    if sum(v.count for (k, v) in state[1]) == 0:
        return (True, False)
    return (False, None)


def next_ship(state):
    comb = dict(state[0] | state[1])
    battle_sorted = sorted(comb,
                           key=lambda x: x.key(), reverse=True)
    # print(battle_sorted)
    for ship in battle_sorted:
        sstate = comb[ship]
        if sstate.count:
            if ship.missles.has_power() and not sstate.fired_missles:
                return ((ship, sstate), 0)
    for ship in battle_sorted:
        sstate = comb[ship]
        if sstate.count:
            if ship.cannons.has_power() and not sstate.fired_cannons:
                return ((ship, sstate), 1)
    for ship in battle_sorted:
        sstate = comb[ship]
        if sstate.count:
            if ship.has_regen() and sstate.damage and not sstate.regen_used:
                return ((ship, sstate), 2)
    return ((None, None), None)


class spaceBattle(markovChain):

    """
    A random walk where we move up and down with rate 1.0 in each
    state between bounds m and M.

    For the transition function to work well, we define some
    class variables in the __init__ function.
    """

    def __init__(self, fleet_attack, fleet_defence):
        super(spaceBattle, self).__init__()
        self.states = {}
        self.states_rev = {}

        self.counter = 0
        initial_state = (frozenset(fleet_attack.items()),
                         frozenset(fleet_defence.items()))
        self.initialState = 0

        self.states_rev[0] = initial_state
        self.states[initial_state] = self.counter
        self.counter += 1

        self.counter += 2
        a_win = 1
        d_win = 2
        self.states_rev[1] = 'a_win'
        self.states_rev[2] = 'd_win'

    def transition(self, state):
        # Specify the reachable states from state and their rates.
        # A dictionary is extremely easy here!
        rates = {}
        if state == 1:
            return {1: 1.0}
        if state == 2:
            return {2: 1.0}

        state = self.states_rev[state]
        ((ship, sstate), action) = next_ship(state)
        # end round
        if ship == None:
            new_states = {reset(state): 1}

        else:
            results = {}
            if action == 0:
                results = ship.fire_missles(sstate, state)
            elif action == 1:
                results = ship.fire_cannons(sstate, state)
            elif action == 2:
                results = ship.regen_damage(sstate, state)
            new_states = results

        for (new_state, v) in new_states.items():
            terminal, dwin = is_terminal(new_state)
            if terminal:
                if dwin and new_state not in self.states:
                    self.states[new_state] = 1
                elif new_state not in self.states:
                    self.states[new_state] = 2
            else:
                if new_state not in self.states:
                    self.states_rev[self.counter] = new_state
                    self.states[new_state] = self.counter
                    self.counter += 1

        rates = {self.states[state]: v/sum(new_states.values())
                 for (state, v) in new_states.items()}
        return rates


inter1 = Ship.interceptor(defender=False,
                          update={
                              'cannons': Firepower(0, 0, 0, 0),
                              'missles': Firepower(3, 0, 0, 0),
                              'initiative': 3,
                              'regen': 0,
                              'armor': 0,
                              'computer': 1,
                              'shield': 0})
dred1 = Ship.draednought(defender=False,
                         update={
                             'cannons': Firepower(2, 0, 0, 0),
                             'missles': Firepower(0, 0, 0, 0),
                             'initiative': 2,
                             'regen': 1,
                             'armor': 2,
                             'computer': 1,
                             'shield': 1})
star1 = Ship.starbase(defender=True,
                      update={
                          'cannons': Firepower(1, 1, 0, 0),
                          'missles': Firepower(0, 0, 0, 0),
                          'initiative': 4,
                          'regen': 0,
                          'armor': 3,
                          'computer': 1,
                          'shield': 0})
crus1 = Ship.cruser(defender=True,
                    update={
                        'cannons': Firepower(1, 0, 0, 0),
                        'missles': Firepower(0, 0, 0, 0),
                        'initiative': 2,
                        'regen': 0,
                        'armor': 4,
                        'computer': 1,
                        'shield': 0})


def print_fleet(d):
    for (k, v) in d.items():
        print(">>>>>>>>")
        print(v)
        print_dict(asdict(k))

def print_dict(d):
    for (k, v) in d.items():
        print(k, v)

def main():
    fleet_attack: Fleet = {
        inter1: ShipState(count=2),
        dred1: ShipState(count=1),
    }
    fleet_defence: Fleet = {
        crus1: ShipState(count=1),
        star1: ShipState(count=2),
    }
    mc = spaceBattle(fleet_attack, fleet_defence)
    #mc.powerMethod(maxiter=1e5)
    mc.computePi('krylov')

    print("Battle!")
    print("Attacker:")
    print_fleet(fleet_attack)
    print("------------------")
    print("Defender")
    print_fleet(fleet_defence)
    print("------------------")

    pi = {}
    for (key, state) in mc.mapping.items():
        if mc.pi[key] > 0.001:
            pi[state] = mc.pi[key]
    for (k, v) in sorted(pi.items(), key=lambda x: x[1], reverse=True):
        print(k, v, mc.states_rev[k])


if __name__ == "__main__":
    # a = (frozenset({(inter1, ShipState(fired_cannons=False, fired_missles=True, regen_used=False, count=1, damage=0)),
    #                 (dred1, ShipState(fired_cannons=False, fired_missles=False, regen_used=False, count=1, damage=0))}),
    #      frozenset({(star1, ShipState(fired_cannons=True, fired_missles=False, regen_used=False, count=1, damage=2))}))
    # # print(reset(a))
    # opp =  frozenset({(star1, ShipState(fired_cannons=True, fired_missles=False, regen_used=False, count=2, damage=0)),
    #             (crus1, ShipState(count=2))})
    # dices = inter1.missles.count(2).fire()
    # res = defaultdict(int)
    # for dice in dices.items():
    #     for r in assign_dices(inter1, opp, 2 * sum(inter1.missles), dice).items():
    #         res[r[0]] += r[1]

    # for r in res.items():
    #     print(r)
    # res2 = defaultdict(int)
    # for dice in dices.items():
    #     for r in assign_dices2(inter1, opp, 2 * sum(inter1.missles), dice).items():
    #         res2[r[0]] += r[1]
        
    # print()
    # print()
    # for r in res2.items():
    #     print(r)

    # print()
    # print()

    # for k in res:
    #     print(res[k], res2[k], res2[k]/res[k])

    # for k in res:
    #     if res2[k] == 0:
    #         print(k, res[k])
    main()
