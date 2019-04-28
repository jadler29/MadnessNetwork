import sys
import os
os.chdir("/Users/jonahadler/Desktop/code/")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/")
import numpy as np
import pandas as pd
from jonahs_things import *
from dfply import *
from tqdm import tqdm
from numba import njit
import numba
bracket = np.zeros((64, 6))
bracket

bracket[1, 1] = 1


def win_or_go_home(bracket):
    for rd in range(bracket.shape[1]-1):
        if (bracket[:, 0] < bracket[:, 1]).any():
            return False
    return True

def king_of_the_block(bracket):
    for rd in range(bracket.shape[1]):
        block_len = 2**(rd+1)
        num_blocks = int(bracket.shape[0]/block_len)
        for blk_num in range(num_blocks):
            blk_start=blk_num*block_len
            if sum(bracket[blk_start:blk_start+block_len,rd]) != 1:
                print("Warning: Does not abide by king_of_the_block in round: ",rd, " in range: [", blk_start, ",", blk_start+block_len,"]")
                return False
    return True

def is_binary(array):
    return np.array_equal(array, array.astype(bool))

def valid_bracket(bracket):
    return win_or_go_home(bracket) and king_of_the_block(bracket) and is_binary(bracket)

valid_bracket(bracket)


pop_P = pd.read_csv(lake+"madness_538_p.csv").values
my_P = pd.read_csv(lake+"madness_538_p.csv").values

p_matrix = my_P

@njit
def sample_winner(team_one,team_two,p_matrix):
    if np.random.rand() < p_matrix[team_one,team_two]:
        return int(team_one)
    else:
        return int(team_two)

@njit
def get_picks(prev_picks,p_matrix):
    new_picks = np.ones(32,dtype= numba.int64)*-1
    for i in range(0,len(prev_picks),2):
        team_one = int(prev_picks[i])
        team_two = int(prev_picks[i+1])
        new_picks[int(i/2)] = sample_winner(team_one, team_two,p_matrix)
    return new_picks

@njit
def picks_to_B_matrix(picks):
    bracket = np.zeros((64, 6))
    for rd in range(picks.shape[1]):
        for i in range(picks.shape[0]):
            if picks[i,rd] == -1:
                break
            bracket[picks[i,rd],rd] = 1
    return bracket
    
@njit
def sample_bracket(p_matrix):
    picks = np.zeros((32, 6),dtype = numba.int64)

    #initial "picks" are just the teams in the tourney
    #prev_picks = np.array(range(p_matrix.shape[0]))
    prev_picks = np.arange(p_matrix.shape[0])

    for rd in range(6):
        new_picks = get_picks(prev_picks,p_matrix)
        picks[:,rd] = new_picks
        #only want to look at the relavant rows
        num_winners = 2**(6-rd)/2
        prev_picks = new_picks[:int(2**(6-rd)/2)]

    #picks = picks.astype(int)
    bracket = picks_to_B_matrix(picks)

    return bracket
@njit
def score_bracket(actual,created, scoring = np.array(range(1,7))**2):
    return np.sum(actual*created*scoring)

@njit
def sim_head2head(p1, p2, actual_p, n_tourneys= 1000, n_bracket_realizations = 1000):
    tourney_results = np.ones(n_tourneys)*-1
    for tourney in range(n_tourneys):
        actual = sample_bracket(actual_p)
        realization_results = np.ones(n_bracket_realizations)*-1
        for real in range(len(realization_results)):
            realization_results[real] = score_bracket(actual, sample_bracket(p1)) > score_bracket(
                actual, sample_bracket(p2))
        #print(mean(res))
        tourney_results[tourney] = np.mean(realization_results)
        if (tourney %(n_tourneys /10) == 0):
            print(tourney)
    return tourney_results

@njit
def sim_head2head_unstruct(p1, p2, actual_p, n_worlds=100000):
    results = np.ones(n_worlds)*-1
    for world in range(n_worlds):
        actual = sample_bracket(actual_p)
        results[world] = score_bracket(actual, sample_bracket(p1)) > score_bracket(
                actual, sample_bracket(p2))
        #print(mean(res))
        if (world % (n_worlds / 10) == 0):
            print(world)
    return results




dumb_P = np.ones(p_matrix.shape)*.5
dumber_P= 1-p_matrix
smart_P = (p_matrix >.5)*1

res = sim_head2head(smart_P,dumb_P, p_matrix)
print(mean(res))

res = sim_head2head(smart_P, p_matrix, p_matrix)
print(mean(res))

res = sim_head2head(smart_P, dumber_P , p_matrix)
print(mean(res))


res = sim_head2head_unstruct(smart_P, p_matrix, p_matrix)
print(mean(res))
