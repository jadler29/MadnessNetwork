from sklearn.model_selection import train_test_split
import numba
from numba import njit
import pandas as pd
import numpy as np


@njit
def sample_winner(team_one, team_two, p_matrix):
    if np.random.rand() < p_matrix[team_one, team_two]:
        return int(team_one)
    else:
        return int(team_two)


@njit
def get_picks(prev_picks, p_matrix):
    new_picks = np.ones(32, dtype=numba.int64)*-1
    for i in range(0, len(prev_picks), 2):
        team_one = int(prev_picks[i])
        team_two = int(prev_picks[i+1])
        new_picks[int(i/2)] = sample_winner(team_one, team_two, p_matrix)
    return new_picks


@njit
def picks_to_B_matrix(picks):
    bracket = np.zeros((64, 6))
    for rd in range(picks.shape[1]):
        for i in range(picks.shape[0]):
            if picks[i, rd] == -1:
                break
            bracket[picks[i, rd], rd] = 1
    return bracket
#def p_matrix_to_B_matrix(p_matrix):


@njit
def sample_bracket(p_matrix):
    picks = np.zeros((32, 6), dtype=numba.int64)

    #initial "picks" are just the teams in the tourney
    #prev_picks = np.array(range(p_matrix.shape[0]))
    prev_picks = np.arange(p_matrix.shape[0])

    for rd in range(6):
        new_picks = get_picks(prev_picks, p_matrix)
        picks[:, rd] = new_picks
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
def win_prob(bracket,sim_params):
    (COMPETING_BRACKET_NUMBER, N_BRACKET_REALIZATIONS,
     P_MATRIX, P_MATRIX_POP) = sim_params
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


def generate_data(sim_params):
    (COMPETING_BRACKET_NUMBER, N_BRACKET_REALIZATIONS, P_MATRIX, P_MATRIX_POP) = sim_params
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


def generate_data_std(sim_params):
    "generate data for the network"
    train, test = train_test_split(generate_data(sim_params), test_size=0)
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
