import sys
import os
os.chdir("/Users/jonahadler/Desktop/code/MadnessNetwork")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/MadnessNetowrk")
import numpy as np
import pandas as pd
from jonahs_things import *
from dfply import *
from tqdm import tqdm
from numba import njit
import numba
from sklearn.model_selection import train_test_split
import sim2_p_matrix







def get_pool_size():
    return COMPETING_BRACKET_NUMBER



def get_my_p():
    return P_MATRIX

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


####################################


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
#def p_matrix_to_B_matrix(p_matrix):



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
def score_bracket(actual, created, scoring=2**np.array(range(0, 6))):
    return np.sum(actual*created*scoring)

@njit
def sim_head2head(p1, p2, actual_p, n_tourneys= 1000):
    tourney_results = np.ones(n_tourneys)*-1
    for tourney in range(n_tourneys):
        actual = sample_bracket(actual_p)
        realization_results = np.ones(N_BRACKET_REALIZATIONS)*-1
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


#@njit
def sim_head2head_B(b1, b2, actual_p, n_worlds=100000):
    results = np.ones(n_worlds)*-1
    for world in range(n_worlds):
        actual = sample_bracket(actual_p)
        results[world] = score_bracket(actual, b1) > score_bracket(
            actual, b2)
        #print(mean(res))
        if (world % (n_worlds / 10) == 0):
            print(world)
    return results


def generate_data():
    p_matrix_true, p_matrix_pop = P_MATRIX, P_MATRIX_POP
    output = np.zeros(shape=(N_BRACKET_REALIZATIONS, 65))
    for realization_number in range(N_BRACKET_REALIZATIONS):
        sample_actuals = sample_bracket(p_matrix_true)
        final_score = 0

        for i in range(COMPETING_BRACKET_NUMBER):
            competing_actuals = sample_bracket(p_matrix_pop)
            score = score_bracket(sample_actuals, competing_actuals)
            final_score = max(final_score, score)

        sample_actuals = sample_actuals.sum(axis=1)
        final_score = np.asarray([final_score])
        inner_output = np.concatenate((sample_actuals, final_score), axis=None)
        output[realization_number] = inner_output.transpose()
    return output

#print(generate_data(p_matrix, p_matrix_pop))


def generate_data_std():
    "generate data for the network"
    train, test = train_test_split(generate_data(), test_size=0)
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]



@njit
def win_prob(bracket):
    results = np.ones(N_BRACKET_REALIZATIONS*2)*-1
    scores = np.ones(N_BRACKET_REALIZATIONS*2)*-1
    for realization_number in range(N_BRACKET_REALIZATIONS*2):
        sample_actuals = sample_bracket(P_MATRIX)
        comp_score = 0

        for i in range(COMPETING_BRACKET_NUMBER):
            competing_actuals = sample_bracket(P_MATRIX_POP)
            this_score = score_bracket(sample_actuals, competing_actuals)
            comp_score = max(comp_score, this_score)

        score = score_bracket(sample_actuals, bracket)
        results[realization_number] = score > comp_score
        scores[realization_number] = score


    return np.mean(results), np.mean(scores), np.std(scores)


if __name__ == "__main__":

    pop_P = pd.read_csv(lake+"pop_p_matrix.csv", header=None).values
    my_P = pd.read_csv(lake+"madness_538_p.csv").values
    my_p_simple = sim2_p_matrix.get_p2_538()

    p_matrix = my_P
    p_matrix_pop = pop_P
    P_MATRIX = my_p_simple
    P_MATRIX_POP = my_p_simple
    COMPETING_BRACKET_NUMBER = 1
    N_BRACKET_REALIZATIONS = 2000

    import output_testing
    """
    b = np.loadtxt('test1.txt', dtype=int)



    dumb_P = np.ones(p_matrix.shape)*.5
    dumber_P= 1-p_matrix
    smart_P = (p_matrix >.5)*1

    mean(sim_head2head_unstruct(smart_P, p_matrix, p_matrix))

    actual = sample_bracket(p_matrix)
    comp = sample_bracket(p_matrix)
    print(score_bracket(actual, b))
    print(score_bracket(actual, comp))

    valid_bracket(b)

    mean(sim_head2head_B(b, sample_bracket(p_matrix), p_matrix))





    res = sim_head2head(smart_P,dumb_P, p_matrix)
    print(mean(res))

    res = sim_head2head(smart_P, p_matrix, p_matrix)
    print(mean(res))

    res = sim_head2head(smart_P, dumber_P , p_matrix)
    print(mean(res))


    res = sim_head2head_unstruct(smart_P, p_matrix, p_matrix)
    print(mean(res))

    """
