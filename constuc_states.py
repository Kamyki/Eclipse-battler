def generate_battle(fleet_attack, fleet_defence):
    for i in product(generate_fleet(fleet_attack), generate_fleet(fleet_defence)):
        for b in generate_fired(i):
            yield b


def generate_fleet(fleet):
    # print(f"{fleet=}")
    for f in product(*[generate_counts(k, v) for (k, v) in fleet.items()]):
        fleet_dict = {v[0]: v[1] for v in f if v[1].count}
        yield fleet_dict


def generate_fired(battle):
    # filtered = (list(filter(lambda x: battle[0][x].count, battle[0])), list(
    battle_sorted = sorted(battle[0] | battle[1],
                           key=lambda x: x.key(), reverse=True)

    b = deepcopy(battle)
    yield b
    battle = deepcopy(battle)
    for k in battle_sorted:
        if k.missles.has_power():
            if k.defender:
                battle[1][k] = replace(battle[1][k], fired_missles=True)
                b = deepcopy(battle)
                yield b
            else:
                battle[0][k] = replace(battle[0][k], fired_missles=True)
                b = deepcopy(battle)
                yield b
    for k in battle_sorted:
        if k.cannons.has_power():
            if k.defender:
                battle[1][k] = replace(battle[1][k],  fired_canons=True)
                b = deepcopy(battle)
                yield b
            else:
                battle[0][k] = replace(battle[0][k], fired_canons=True)
                b = deepcopy(battle)
                yield b
    for k in battle_sorted:
        if k.has_regen():
            if k.defender:
                battle[1][k] = replace(battle[1][k], regen_used=True)
                b = deepcopy(battle)
                yield b
            else:
                battle[0][k] = replace(battle[0][k], regen_used=True)
                b = deepcopy(battle)
                yield b


def generate_counts(ship, state):
    for i in range(state.count, 0, -1):
        for d in range(0, ship.armor+1):
            s = replace(state, count=i, damage=d)
            yield (ship, s)
    s = replace(state, count=0, damage=0)
    yield (ship, s)


def generate_states(fleet_attack: Fleet, fleet_defence: Fleet) -> StatesT:
    states = {(frozenset(a.items()), frozenset(d.items())): i for (
        i, (a, d)) in enumerate(generate_battle(fleet_attack, fleet_defence))}
    return states
