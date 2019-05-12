import numpy as np
import os
import sys
os.chdir("/Users/jonahadler/Desktop/code/")
sys.path.insert(0, "/Users/jonahadler/Desktop/code/")
import math
from jonahs_things import *
import pandas as pd
from dfply import *
from scipy.optimize import linprog

silver_preds = pd.read_csv(lake + "538_forecasts.csv")
silver_preds = (silver_preds >>
mask(X.gender == "mens", X.forecast_date == "3/14/18", X.rd1_win==1.0)
)
silver_preds = silver_preds.reset_index(drop=True)

intra_conf_order = {
    1 : 1,
    16 : 1,
    #
    8 : 2 ,
    9 : 2,
    #
    5 : 3,
    12 : 3,
    #
    4 : 4,
    13 : 4,
    #
    6 : 5,
    11 : 5,
    #
    3 : 6,
    14 : 6,
    #
    7 : 7,
    10 : 7,
    #
    2 : 8,
    15 : 8,

}
conf_order = {
    "East" : 1,
    "Midwest" : 2,
    "South" : 3,
    "West" : 4
}

silver_preds.team_seed = silver_preds.team_seed.str.extract('(\d+)', expand=False).astype(int)
silver_preds["intra_conf_order"] = silver_preds.team_seed.map(intra_conf_order)
silver_preds["conf_order"] = silver_preds.team_region.map(conf_order)
silver_preds = silver_preds.sort_values(["conf_order","intra_conf_order"])


q_matrix = (silver_preds.filter(regex="win")
>> drop("rd1_win")
)
q_matrix = q_matrix.values

def get_q():
    return q_matrix


def get_round_prob(q):
    round_prob = q
    q_exact = q
    for col in range(1, 6):
        round_prob[:, col] = q_exact[:, col]/q_exact[:, col-1]
    return round_prob


def win_prob(team1_idx, team2_idx, col, round_prob):
    rel_score = sum(round_prob[team1_idx, :(col+1)]) / sum(round_prob[team2_idx, :(col+1)])
    return rel_score/(1+rel_score)

def derive_p_v2(q_matrix):
    p_matrix = np.zeros((64, 64))
    np.fill_diagonal(p_matrix,np.nan)
    round_prob = get_round_prob(q_matrix)
    zone_matrix = get_zone_matrix(q_matrix)
    for team1_idx in range(64):
        for team2_idx in range(64):
            if team1_idx == team2_idx:
                continue            
            col = round_meet(team1_idx,team2_idx,zone_matrix)
            p_matrix[team1_idx,team2_idx] = win_prob(team1_idx,team2_idx,col, round_prob)
    return p_matrix
def round_meet(team1_idx,team2_idx, zone_matrix):
    "what round will two teams meet?"
    return np.argmax(zone_matrix[team1_idx,:] ==zone_matrix[team2_idx,:])
    


def get_zone_matrix(q_matrix):
    'create matrix defining who plays who it what round'
    zone_matrix = np.zeros_like(q_matrix)
    for y in range(zone_matrix.shape[0]):
        for x in range(zone_matrix.shape[1]):
            zone_matrix[y,x] = int(math.floor(y)/2**(x+1))
    return zone_matrix

def get_p2_538():
    return derive_p_v2(q_matrix)


#def get_p2_528():
 #   return derive_p_v2(q_matrix)


#p = np.round(get_p2(),2)

def derive_p_matrix(q_matrix):
    p_matrix = np.zeros((64,64))
    np.fill_diagonal(p_matrix,np.nan)
    #multiply blk_start
    for rd in range(q_matrix.shape[1]):
        block_len = 2**(rd+1)
        num_blocks = int(q_matrix.shape[0]/block_len)
        for blk_num in range(num_blocks):
            #we already know the answer for round 1
            print("assesing group ",blk_num, " for round ", rd)


            blk_start=blk_num*block_len
            group_size = int(block_len/2)
            combs = group_size**2

            if rd==0:
                print("blk_start",blk_start)
                p_matrix[blk_start,blk_start+1] = q_matrix[blk_start,0]
                p_matrix[blk_start+1,blk_start] = q_matrix[blk_start+1,0]
                continue
            #build coeffs matrix
            #first fill ones
            a = np.identity(group_size)
            top = np.repeat(a,group_size, axis=1)
            bottom = np.tile(a,group_size)
            ones = np.concatenate([top,bottom])

            #now we need to fill with values
            top_co = q_matrix[blk_start:blk_start+group_size,rd-1]
            bottom_co = q_matrix[blk_start+group_size:blk_start+2*group_size,rd-1]
            coeffs = np.outer(top_co,bottom_co).flatten()

            #now combine ones with coeffs to get the coefficient matrix
            co_matrix= ones * coeffs

            #now we need dependednt vars
            top_dep = q_matrix[blk_start:blk_start+group_size,rd]
            bottom_dep  = q_matrix[blk_start+group_size:blk_start+2*group_size,rd-1] - q_matrix[blk_start+group_size:blk_start+2*group_size,rd]
            dep = np.concatenate([top_dep,bottom_dep])

            #now constraints
            A = np.identity(combs)
            #upper and lower bounds
            b = np.ones(combs)

            #now add the normalizing constraints (min abs deviance from overall for each team)
            avgs = q_matrix[blk_start:blk_start+combs,rd] / q_matrix[blk_start:blk_start+combs,rd-1]

            #works???!
            avgs = np.nan_to_num(avgs)

            left = np.repeat(np.identity(combs),4,axis=0)*np.tile([1,-1,-1,1],(combs))[:,np.newaxis]
            middle =  np.repeat(np.identity(combs*2),2,axis=0)*-1
            co_new= np.concatenate([left,middle],axis=1)

            right = np.zeros(combs*4)
            for i in range(combs*4):
                var_num = int(i/4)
                first_team = int(var_num/group_size)
                second_team =var_num%group_size + group_size
                which = i%4
                if which <=1:
                    right[i]= avgs[first_team]
                else:
                    right[i]= avgs[second_team]-1

                if which%2 !=0:
                    right[i] = right[i]*-1

            co_matrix2 = np.concatenate([co_matrix[:-1,:],np.zeros((co_matrix[:-1,:].shape[0],combs*2))],axis=1)
            dep2=dep[:-1]
            A_wide = np.concatenate([A,np.zeros((A.shape[0],combs*2))],axis=1)

            A_2 = np.concatenate([A_wide,co_new])
            b_2 = np.concatenate([b,right])

            A_2.shape
            b_2.shape
            
            co_matrix2.shape
            dep2.shape

            obj= np.concatenate([np.zeros(combs),np.ones(combs*2)])
            p_out = linprog(obj,A_2,b_2,co_matrix2,dep2).x[:combs]5j

            for i in range(len(p_out)):
                first_team = int(i/group_size) + blk_start
                second_team = i%group_size + blk_start + group_size
                p_matrix[first_team,second_team] = p_out[i]
                p_matrix[second_team,first_team] = 1-p_out[i]
    return p_matrix

if __name__ == "__main__":
    #derive_p_matrix(q_matrix)

    derive_p_v2(q_matrix)
    b = np.loadtxt('test1.txt', dtype=int)

    derive_p_matrix(b)
=[]p