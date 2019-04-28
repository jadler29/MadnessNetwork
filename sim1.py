import numpy as np
from jonahs_things import *
import pandas as pd
from dfply import *

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
        for blk_start in range(num_blocks):
            if sum(bracket[blk_start:blk_start+block_len,rd]) != 1:
                print("Warning: Does not abide by king_of_the_block in round: ",rd, " in range: [", blk_start, ",", blk_start+block_len,"]")
                return False
    return True

def is_binary(array):
    return np.array_equal(array, array.astype(bool))

def valid_bracket(bracket):
    return win_or_go_home(bracket) and king_of_the_block(bracket) and is_binary(bracket)

valid_bracket(bracket)
from scipy.optimize import linprog


rd=1
block_len = 2**(rd+1)
num_blocks = int(bracket.shape[0]/block_len)
blk_start=16
group_size = int(block_len/2)
combs = group_size**2

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
#A = np.concatenate([np.identity(combs),np.identity(combs)*-1])
A = np.identity(combs)
#upper and lower bounds
b = np.ones(combs)
#b = np.concatenate([np.ones(combs),np.zeros(combs)])

linprog(np.ones(combs)*-1,A,b,co_matrix[:-1],dep[:-1]).x
linprog(np.zeros(combs)*1,A,b,co_matrix[:-1],dep[:-1]).x

#now add the normalizing constraints (min abs deviance from overall for each team)
avgs = q_matrix[blk_start:blk_start+combs,rd] / q_matrix[blk_start:blk_start+combs,rd-1]

left = np.repeat(np.identity(combs),4,axis=0)*np.tile([1,-1,-1,1],(combs))[:,np.newaxis]
middle =  np.repeat(np.identity(combs*2),2,axis=0)*-1
co_new= np.concatenate([left,middle],axis=1)

left.shape
middle.shape

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
obj[-1]=2
linprog(obj,A_2,b_2,co_matrix2,dep2).x[:4]






#multiply blk_start
for rd in range(bracket.shape[1]):
    block_len = 2^(rd+1)
    num_blocks = int(bracket.shape[0]/block_len)
    for blk_start in range(num_blocks):
        if sum(bracket[blk_start:blk_start+block_len,rd]) != 1:
            print("Warning: Does not abide by king_of_the_block in round: ",rd, " in range: [", blk_start, ",", blk_start+block_len,"]")
            return False
return True
