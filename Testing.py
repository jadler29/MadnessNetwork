#import output_testing
import os
os.chdir("/Users/jonahadler/Desktop/code/MadnessNetwork")
from output_testing import make_bracket_and_plot
import pandas as pd
from sim2_p_matrix import get_q, get_p2_538
import sys
sys.path.append("..")
from jonahs_things import lake

pop_P = pd.read_csv(lake+"pop_p_matrix.csv", header=None).values
my_P = pd.read_csv(lake+"madness_538_p.csv").values
my_p_simple = get_p2_538()

P_MATRIX = my_p_simple
P_MATRIX_POP = pop_P
COMPETING_BRACKET_NUMBER=50

q = get_q()
p_options = [my_p_simple, my_P]
p_desc = ["p_simple", "p_standard"]


N_BRACKET_REALIZATIONS= 2000

'''
for p_i in range(len(p_options)):
    p_pop_options = [pop_P, p_options[p_i]]
    p_pop_desc = ["wpw", "same_pop"]
    for p_pop_i in range(len(p_pop_options)):
        for COMPETING_BRACKET_NUMBER in [1, 15, 50, 100, 500]:
            P_MATRIX, P_MATRIX_POP = p_options[p_i], p_pop_options[p_pop_i]

            save_name = p_pop_desc[p_pop_i] + "_"+p_desc[p_i] + \
                "_with_half_adj_" + str(COMPETING_BRACKET_NUMBER)

            make_bracket_and_plot(COMPETING_BRACKET_NUMBER,
                                  N_BRACKET_REALIZATIONS, P_MATRIX, P_MATRIX_POP, q, save_name)

'''



p_i=0

p_pop_i=0
p_pop_options = [pop_P, p_options[p_i]]
p_pop_desc = ["wpw", "same_pop"]

for COMPETING_BRACKET_NUMBER in [50]:
    P_MATRIX, P_MATRIX_POP = p_options[p_i], p_pop_options[p_pop_i]

    save_name = p_pop_desc[p_pop_i] + "_"+p_desc[p_i] + \
        "_with_half_adj_" + str(COMPETING_BRACKET_NUMBER)

    make_bracket_and_plot(COMPETING_BRACKET_NUMBER,
                            N_BRACKET_REALIZATIONS, P_MATRIX, P_MATRIX_POP, q, save_name)
