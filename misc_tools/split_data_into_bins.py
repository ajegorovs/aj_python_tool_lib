"""
someone on discord wants an algo to fill teams with players. they had N leaders
and wanted to fill the rest so that teams are balanced. in the example i generate all possible team combinations
and give all players a skill level. combination with least deviation is the best.

method is recursive and order of added players matters. 
"""
import itertools, numpy as np
seed_value = 42
np.random.seed(seed_value)
def binning(pool, parent, n_bins, storage, layer_prev = 0):
    """
    algorithm for determing all possible ways of filling N buckets from pool
    of items. Only split into equal parts is possible.
    """

    layer_this = layer_prev + 1
    choices = list(itertools.permutations(pool, n_bins)) # combinations misses results
    
    children = []
    for choice in choices:
        pool_refined = [a for a in pool if a not in choice]
        if len(pool_refined) > 0:
            binning(pool_refined, choice, n_bins, children, layer_this)
        else:        
            children.append([choice])
    a = 1
    if len(parent) > 0: # outermost iter for case with initial fill.
        [storage.append([parent] + c) for c in children]
    else:
        storage.extend(children)
    
n_teams     = 2
num_players = 3
picks = []
all_players = list(range(1, num_players* n_teams + 1, 1))
if 1 == -1:  # specify leaders and fill teams from rest
    leaders = tuple(all_players[:n_teams])
    pool    = tuple(all_players[n_teams:])
    binning(pool, leaders, n_teams, picks)
else:       # fill bins from whole pool
    binning(all_players, [], n_teams, picks)

picks_T             = [list(zip(*a)) for a in picks]  # zip ~= transpose iterator
# 1) make order of players in a team irrelevant  2) make order of teams irrelavant
picks_unique        = list({frozenset([frozenset(a) for a in b]) for b in picks_T})   
team_combinations   = list({tuple([tuple(a) for a in b]) for b in picks_unique})
team_combinations_np= np.array(team_combinations)

# find best combination by uniformity of team's mean weight
weigth              = {p: np.random.randint(50, 101) for p in all_players}
team_weights        = [[[weigth[p] for p in team] for team in teams] for teams in team_combinations]
team_weights_np     = np.array(team_weights)                        # axis 0: comb, 1: team number 2: players weights
team_weights_mean   = np.mean(team_weights_np, axis= 2)
team_diffs          = np.abs(np.std(team_weights_mean, axis= 1))    # calc weights standard deviations
min_diff            = np.argmin(team_diffs)                         # pick one with least stdev
best_team_comb      = team_combinations_np[min_diff]
print(team_weights_np[min_diff])
a = 1
