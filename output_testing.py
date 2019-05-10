import sys
sys.path.insert(0, "/Users/jonahadler/Desktop/code/MadnessNetwork/")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/")
import pandas as pd
import numpy as np
import os
import pylab
from jonahs_things import *
import seaborn as sns
import math
import matplotlib.pyplot as plt
from output_conversion import output_conversion
import bracket_creation
import sim2_p_matrix
from sim2_p_matrix import intra_conf_order

sns.set()
pylab.rcParams['figure.figsize'] = (6.0, 15)
plt.style.use('ggplot')
COMPETING_BRACKET_NUMBER = bracket_creation.get_pool_size()

import bracket_opt
b, win_p = bracket_opt.run_opt("../pool"+str(COMPETING_BRACKET_NUMBER)+".txt")

#b = np.loadtxt('std.txt', dtype=int)
q= np.round(sim2_p_matrix.get_q(),2)

'''
picks = pd.DataFrame(b)
picks = picks.add_suffix('_picks')
q_df = pd.DataFrame(q)
q_df = q_df.add_suffix('_q')

df = pd.concat([picks,q_df],axis=1)

q*b / .5**np.arange(1,7)

df.sort_values("0_q", ascending = False)
'''

pop_P = pd.read_csv(lake+"pop_p_matrix.csv", header=None).values
p = pd.read_csv(lake+"madness_538_p.csv").values

def get_round_prob(q):
    round_prob = q
    q_exact = sim2_p_matrix.get_q()
    for col in range(1, 6):
        round_prob[:, col] = q_exact[:, col]/q_exact[:, col-1]
    return round_prob
def win_prob(team1_idx,team2_idx,col,round_prob):
    return round_prob[team1_idx, col] / round_prob[team2_idx, col]
    
def get_upset_matrix(q,b,p_matrix,round_prob):
    upset_matrix = np.zeros_like(b,dtype = float)
    #upset_matrix[:,0] = bracket[:,0]*q[:,0]
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
                upset_matrix[team2_idx, col] = 1/prob_1_wins
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
                    upset_matrix[team2_idx, col] = 1/prob_1_wins
            row+=1
    upset_matrix = upset_matrix / (1+upset_matrix)
    return np.round(upset_matrix,3)

upset_matrix = get_upset_matrix(q,b,p,get_round_prob(q))


ax = sns.heatmap(upset_matrix, center=.5, mask=upset_matrix == 0, cmap="PiYG",annot=True, cbar=False)

conf_order = np.array(list(intra_conf_order.keys()))
tour_order = np.tile(conf_order, 4)
ax.set_yticklabels(tour_order)
ax.set_xticklabels(["R64","R32","Sw 16","El 8", "Final 4", "NCG"])
ax.set_title("Opimized bracket; "+str(COMPETING_BRACKET_NUMBER)+ " competitors \n 1st place probability = " + str(round(win_p,2)))
fig = ax.get_figure()
ax.set_ylabel("Seed")
ax.legend("P")
fig.savefig('hmp_samep'+str(COMPETING_BRACKET_NUMBER)+'.png', facecolor = "w")

#plt.show()
