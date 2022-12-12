# /bin/python3

from typing import Any, Dict, List, Tuple, NamedTuple
from collections import Counter, defaultdict
from enum import Enum
from math import prod
from dataclasses import dataclass, field, replace
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

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f'{self.name}'


@dataclass(frozen=True)
class ShipState:
    fired_canons: bool = False
    fired_missles: bool = False
    regen_used: bool = False
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
    def cruser(defender=True):
        STATS = {'ship_class': ShipClass.CRUSER,
                 'cannons':  Firepower(1, 0, 0, 0),
                 'missles':  Firepower(0, 0, 0, 0),
                 'initiative': 2,
                 'regen': 0,
                 'armor': 1,
                 'computer': 1,
                 'shield': 0}

        return Ship(**STATS, defender=defender)

    @staticmethod
    def interceptor(defender=True):
        STATS = {'ship_class': ShipClass.INTNRCEPTOR,
                 'cannons': Firepower(1, 0, 0, 0),
                 'missles': Firepower(0, 0, 0, 0),
                 'initiative': 3,
                 'regen': 0,
                 'armor': 0,
                 'computer': 0,
                 'shield': 0}

        return Ship(**STATS, defender=defender)

    def key(self):
        return (self.initiative, self.defender)

    def has_regen(self):
        return self.regen != 0

    def fire_missles(self, opposing_fleet):
        opposing_fleet = dict(opposing_fleet)
        dices = self.missles.fire()
        results = defaultdict(int)
        for dice_set in dices.items():
            for (k, v) in assign_dices(self, opposing_fleet, sum(self.cannons), dice_set).items():
                results[k] += v
        # if  sum(dices.values()) != sum(results.values()):
        #     for r in results.items():
        #         print(r)
        return results

    def fire_cannons(self, opposing_fleet):
        opposing_fleet = dict(opposing_fleet)
        dices = self.cannons.fire()
        results = defaultdict(int)
        for dice_set in dices.items():
            for (k, v) in assign_dices(self, opposing_fleet, sum(self.cannons), dice_set).items():
                results[k] += v
        # print( sum(results.values()))
        # print(sum(dices.values()))
        # if  sum(dices.values()) != sum(results.values()):
        #     for r in dices.items():
        #         print(r)
        #     for r in results.items():
        #         print(r)
        # assert sum(dices.values()) == sum(results.values())
        return results

    def regen_damage(self, my_fleet):
        my_fleet = dict(my_fleet)
        state = my_fleet[self]
        f = deepcopy(my_fleet)
        f[self] = replace(state, damage=max(0, state.damage-self.regen))
        result = defaultdict(int, {frozenset(f.items()): 1})


Fleet = Dict[Ship, ShipState]
StateT = Tuple[Fleet, Fleet]
StatesT = List[StateT]


def assign_dices(ship, opposing_fleet, dice_count, dices):
    succ_states = defaultdict(int)
    for ships in product(opposing_fleet, repeat=dice_count):
        i = 0
        fleet = deepcopy(opposing_fleet)
        for power in range(4):
            for (dice, count) in dices[0][power]:
                for _ in range(count):
                    ship = ships[i]
                    i += 1
                    # print(dice)
                    # print(f"befor {fleet[ship]=}")
                    if fleet[ship].count:
                        fleet[ship] = damage(
                            ship, fleet[ship], power+1, dice + ship.computer, dice == 6, dice == 1)
                    # print(f"after {fleet[ship]=}")

        succ_states[frozenset(
            filter(lambda x: x[1].count, fleet.items()))] += dices[1]

    return succ_states


def damage(ship, state, power, dice, bullseye=False, miss=False) -> ShipState:
    if miss or (dice - ship.shield < 6 and not bullseye):
        return state
    elif dice - ship.shield >= 6 or bullseye:
        if power > ship.armor:
            s = replace(state, count=state.count-1, damage=state.damage)
            return s
        if state.damage + power > ship.armor:
            s = replace(state, count=state.count-1, damage=0)
            return s
        else:
            s = replace(state, damage=state.damage+power)
            return s


def create_matrix(initial_state):
    row = []
    col = []
    data = []
    queue = []
    visited = set()
    states = {}
    counter = 0

    queue.append(initial_state)
    if initial_state not in states:
        states[initial_state] = counter
        counter += 1

    counter += 2
    row.append(1)
    col.append(1)
    row.append(2)
    col.append(2)
    data.append(1)
    data.append(1)

    a_win = set()
    d_win = set()
    while queue != []:
        state = queue.pop()
        if states[state] in visited:
            continue
        visited.add(states[state])
        (ship, action) = next_ship(state)
        # end round
        if ship == None:
            new_states = {reset(state): 1}

        else:
            opposing_fleet, my_fleet = (
                state[0], state[1]) if ship.defender else (state[1], state[0])
            results = {}
            if action == 0:
                results = ship.fire_missles(opposing_fleet)
                temp = dict(my_fleet)
                temp[ship] = replace(temp[ship], fired_missles=True)
                my_fleet = frozenset(temp.items())
                new_states = {(r, my_fleet): v for (r, v) in results.items()} if ship.defender else {
                    (my_fleet, r): v for (r, v) in results.items()}
            elif action == 1:
                results = ship.fire_cannons(opposing_fleet)
                temp = dict(my_fleet)
                temp[ship] = replace(temp[ship], fired_canons=True)
                my_fleet = frozenset(temp.items())
                new_states = {(r, my_fleet): v for (r, v) in results.items()} if ship.defender else {
                    (my_fleet, r): v for (r, v) in results.items()}
            elif action == 2:
                results = ship.fire_cannons(my_fleet)
                new_states = {(opposing_fleet, r): v for (r, v) in results.items(
                )} if ship.defender else {(r, opposing_fleet): v for (r, v) in results.items()}

        # print('----------------------------')
        # print('----------------------------')
        if sum(new_states.values()) == 0:
            print(state)
            print(ship)
            print(action)
            print('----------------------------')
            print(sum(new_states.values()))
        for (new_state, v) in new_states.items():

            # print(new_state)
            # print(v)
            terminal, dwin = is_terminal(new_state)
            if terminal:
                if dwin and new_state not in states:
                    states[new_state] = 1
                elif new_state not in states:
                    states[new_state] = 2
            else:
                if new_state not in states:
                    states[new_state] = counter
                    counter += 1
            if terminal:
                # print(terminal, dwin)
                if dwin:
                    d_win.add(states[new_state])
                else:
                    a_win.add(states[new_state])
            elif states[new_state] not in visited:
                # print('ih')
                queue.append(new_state)

            # if new_state not in states:
            #     exit(1)
            row.append(states[state])
            col.append(states[new_state])
            data.append(v)

    return coo_matrix((data, (row, col)), shape=(counter, counter)), a_win, d_win, states


def reset(state):
    a = frozenset((k, replace(v, fired_canons=False, regen_used=False))
                  for (k, v) in state[0])
    d = frozenset((k, replace(v, fired_canons=False, regen_used=False))
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
            if ship.missles.has_power() and not sstate.fired_missle:
                return (ship, 0)
    for ship in battle_sorted:
        sstate = comb[ship]
        if sstate.count:
            if ship.cannons.has_power() and not sstate.fired_canons:
                return (ship, 1)
    for ship in battle_sorted:
        sstate = comb[ship]
        if sstate.count:
            if ship.has_regen() and not sstate.regen_used:
                return (ship, 2)
    return (None, None)


def markov_stationary_components(P, tol=1e-12):
    """
    Split the chain first to connected components, and solve the
    stationary state for the smallest one
    """
    n = P.shape[0]

    # 0. Drop zero edges
    P = P.tocsr()
    P.eliminate_zeros()

    # 1. Separate to connected components
    n_components, labels = csgraph.connected_components(
        P, directed=True, connection='weak')

    # The labels also contain decaying components that need to be skipped
    index_sets = []
    for j in range(n_components):
        indices = np.flatnonzero(labels == j)
        other_indices = np.flatnonzero(labels != j)

        Px = P[indices, :][:, other_indices]
        if Px.max() == 0:
            index_sets.append(indices)
    n_components = len(index_sets)

    # 2. Pick the smallest one
    sizes = [indices.size for indices in index_sets]
    min_j = np.argmin(sizes)
    indices = index_sets[min_j]

    print(
        "Solving for component {0}/{1} of size {2}".format(min_j, n_components, indices.size))

    # 3. Solve stationary state for it
    p = np.zeros(n)
    if indices.size == 1:
        # Simple case
        p[indices] = 1
    else:
        p[indices] = markov_stationary_one(P[indices, :][:, indices], tol=tol)

    return p


def markov_stationary_one(P, tol=1e-12, direct=False):
    """
    Solve stationary state of Markov chain by replacing the first
    equation by the normalization condition.
    """
    if P.shape == (1, 1):
        return np.array([1.0])

    n = P.shape[0]
    dP = P - sparse.eye(n)
    A = sparse.vstack([np.ones(n), dP.T[1:, :]])
    rhs = np.zeros((n,))
    rhs[0] = 1

    if direct:
        # Requires that the solution is unique
        return spsolve(A, rhs)
    else:
        # GMRES does not care whether the solution is unique or not, it
        # will pick the first one it finds in the Krylov subspace
        p, info = gmres(A, rhs, tol=tol)
        if info != 0:
            raise RuntimeError("gmres didn't converge")
        return p


def doit(P):
    assert isinstance(P, sparse.csr_matrix)
    assert np.isfinite(P.data).all()

    print("Construction finished!")

    def check_solution(method):
        print("\n\n-- {0}".format(method.__name__))
        start = time.time()
        p = method(P)
        print("time: {0}".format(time.time() - start))
        print("error: {0}".format(np.linalg.norm(P.T.dot(p) - p)))
        print("min(p)/max(p): {0}, {1}".format(p.min(), p.max()))
        print("sum(p): {0}".format(p.sum()))
        print("a win: {0}".format(p[2]))
        print("d win: {0}".format(p[1]))
        print("p: {0}".format(p))

    check_solution(markov_stationary_components)
    check_solution(markov_stationary_one)


def main():
    fleet_attack: Fleet = {Ship.cruser(defender=False): ShipState()}
    fleet_defence: Fleet = {Ship.cruser(defender=True): ShipState()}
    # states = generate_states(fleet_attack, fleet_defence)

    initial_state = (frozenset(fleet_attack.items()),
                     frozenset(fleet_defence.items()))
    matrix, awin, dwin, states = create_matrix(initial_state)
    print(len(states))
    for state in states.items():
        print(state)

    P = matrix.tocsr()
    P = P.multiply(sparse.csr_matrix(1/P.sum(1).A))

    print(P.toarray())
    print(awin)
    print(dwin)

    doit(P)


if __name__ == "__main__":
    main()
