import bracket_opt
import sys
sys.path.insert(0, "/Users/jonahadler/Desktop/code/MadnessNetwork/")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/")
import pandas as pd
import numpy as np
import os
import pylab
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sim2_p_matrix import intra_conf_order, get_q

sns.set()
pylab.rcParams['figure.figsize'] = (6.0, 15)
plt.style.use('ggplot')


def make_bracket_and_plot(COMPETING_BRACKET_NUMBER, N_BRACKET_REALIZATIONS,P_MATRIX, P_MATRIX_POP,Q, save_name):


    sim_params = (COMPETING_BRACKET_NUMBER, N_BRACKET_REALIZATIONS,
                    P_MATRIX, P_MATRIX_POP) 
        
    b, win_p = bracket_opt.run_opt(
        "/Users/jonahadler/Desktop/code/MadnessNetwork/out_data/" + save_name + ".txt", sim_params)

    #b, win_p = np.loadtxt('/Users/jonahadler/Desktop/code/MadnessNetwork/out_data/wpw_p_simple_with_50.txt', dtype=int),9
    #b = np.loadtxt('std.txt', dtype=int)
    Q = np.loadtxt("/Users/jonahadler/Desktop/code/QQQ.txt")
    p = P_MATRIX

    def get_round_prob(Q2):
        round_prob = Q2.copy()
        q_exact = Q2.copy()
        for col in range(1, 6):
            round_prob[:, col] = q_exact[:, col]/q_exact[:, col-1]
        return round_prob


    def win_prob(team1_idx, team2_idx, col, round_prob):
        rel_score = sum(round_prob[team1_idx, :(col+1)]) / sum(round_prob[team2_idx, :(col+1)])
        return rel_score/(1+rel_score)

    def get_upset_matrix(b,p_matrix,round_prob):
        upset_matrix = np.zeros_like(b,dtype = float)
        #upset_matrix[:,0] = bracket[:,0]*Q[:,0]
        col =0 
        row = 0
        while(row < b.shape[0]):
            if True:
                team1_idx = row
                row += 1
                team2_idx = row

                team_1_winner = b[team1_idx, col] == 1

                prob_1_wins = win_prob(team1_idx, team2_idx, col, round_prob)

                if team_1_winner:
                    upset_matrix[team1_idx, col] = prob_1_wins
                else:
                    upset_matrix[team2_idx, col] = 1-prob_1_wins
            row += 1



        for col in range(1,b.shape[1]):
            row=0
            while(row < b.shape[0]):
                if b[row,col-1] == 1:
                    team1_idx = row
                    row+=1
                    while (b[row, col-1] != 1):
                        row+=1
                    team2_idx = row

                    team_1_winner = b[team1_idx, col] ==1

                    prob_1_wins = win_prob(team1_idx,team2_idx,col,round_prob)
                    
                    if team_1_winner:
                        upset_matrix[team1_idx,col] = prob_1_wins
                    else:
                        upset_matrix[team2_idx, col] = 1-prob_1_wins
                row+=1
        #pset_matrix = upset_matrix / (1+upset_matrix)
        return np.round(upset_matrix,3)


    #b = np.loadtxt(
     #   '/Users/jonahadler/Desktop/code/MadnessNetwork/out_data/wpw_p_simple_with_15.txt', dtype=int)

    upset_matrix = get_upset_matrix(b,p,get_round_prob(Q))


    ax = sns.heatmap(upset_matrix, center=.5, mask=upset_matrix == 0, cmap="PiYG",annot=True, cbar=False)

    conf_order = np.array(list(intra_conf_order.keys()))
    tour_order = np.tile(conf_order, 4)
    ax.set_yticklabels(tour_order)
    ax.set_xticklabels(["R64","R32","Sw 16","El 8", "Final 4", "NCG"])
    ax.set_title("Opimized bracket; "+str(COMPETING_BRACKET_NUMBER)+ " competitors \n 1st place probability = " + str(round(win_p,3)))
    fig = ax.get_figure()
    ax.set_ylabel("Seed")
    ax.legend("P")
    fig.savefig("/Users/jonahadler/Desktop/code/MadnessNetwork/out_data/" +
                save_name+'.png', facecolor="w")
    plt.close()
