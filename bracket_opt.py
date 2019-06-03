# -*- coding: utf-8 -*-
#%%
import math
import os
from random import randint, uniform
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import Opt_Helpers
os.chdir("/Users/jonahadler/Desktop/code/MadnessNetwork")
from output_conversion import output_conversion



class Knockout_Round_Layer(torch.autograd.Function):
    """
    We can sublass PyTorch's autograd function to create our
    autograd function for a general knockout round tranformation. 
    Then, PyTorch will handle gradients for us. 

    """
    @staticmethod
    def forward(ctx, input, weight):
        """

        In the forward pass we receive a Tensor containing the input 
        and return a Tensor containing the output. 
        ctx is a context object that can be used
        to stash information for backward computation
        """
        weight = torch.clamp(weight, 0, 1)
        ctx.save_for_backward(input, weight)
        w_new = torch.empty(2*len(weight), dtype=weight.dtype)

        #weight share with 1-w
        w_new[0::2] = weight
        w_new[1::2] = 1-weight

        weighted_a = input*w_new
        eye = torch.eye(len(weight))

        assignment = torch.nn.functional.interpolate(
            eye.unsqueeze(0), size=(len(weight)*2)).squeeze(0).t()
        z = weighted_a.mm(assignment)

        return z

    @staticmethod
    def backward(ctx, grad_output):
        """
        Here we receive the gradient of loss with respect to 
        the output, dL_dO,
        and need to return the loss w.r.t the input, dL_di. 
        We can calculate dO_dw and use the chain rule 
        """
        input, weight = ctx.saved_tensors
        dL_dO = grad_output.clone()
        dO_dw = (input[:, 0::2]-input[:, 1::2])  
        dL_di = dL_dO * dO_dw

        return None, dL_di


class MadnessNet(torch.nn.Module):
    def __init__(self):
        """
        Init weights to random values 0<->1 and clamp .4 <->.6
        """
        super(MadnessNet, self).__init__()

        w_list = [None]
        for rd in range(1, N_ROUNDS+1):
            w_list.append(
                torch.nn.Parameter(
                    torch.clamp(
                        torch.randn(
                            int(N_TEAMS*(.5)**(rd)),
                        ), .4, .6)
                ))

        self.w = torch.nn.ParameterList(w_list)
        self.knockout = Knockout_Round_Layer.apply


    def forward(self, x):
        """
        Apply bracket network transformation
        """
        z = []
        score = torch.zeros((x.shape[0]))
        z.append(x)
        for rd in range(1, N_ROUNDS+1):
            z.append(self.knockout(z[rd-1], self.w[rd]))

            #threshold for sigmoid is the round number itself
            score += torch.sigmoid(
                SIGMOID_STEEPNESS*(z[rd]-rd+.5)
            ).sum(dim=1)*SCORING[rd]


        return score


def best_score_loss(score, lamb, punishment, _):
    pure_score = score.mean()
    loss = -pure_score + lamb * punishment
    #print(pure_score.item())
    return loss, pure_score.item()




def win_prob_loss(score, lamb, punishment, best_competitor):
    win = torch.sigmoid(score - best_competitor).mean()
    loss = -GAMMA*win + lamb * punishment
    #print(win.item())
    return loss, win.item()


def get_model(lr):
    model = MadnessNet()
    #return model, torch.optim.SGD(model.parameters(), lr=lr,momentum=.5)
    return model, torch.optim.Adam(model.parameters(), lr=lr)


def punish(w):
    punishment = 0
    for i in range(1, len(w)):
        punishment += torch.relu(-w[i]).sum() + \
            torch.relu(w[i]-1).sum()
    return punishment


def get_w(model):
    w = [np.nan]
    for rd in range(1, N_ROUNDS+1):
        w.append(
            np.round(
                model.w[rd].detach().numpy(), decimals=2)
        )
    return w


def breakdown_w_index(single_index):
    spot = math.log2(single_index)
    floor = math.floor(spot)
    index = 2**(floor) - floor - 1
    level = 6-floor
    return level, index


def random_w():
    weight_num = randint(1, 63)
    return breakdown_w_index(weight_num)


def explore_prob(epoch):
    return EPSILON_GREEDY**(-epoch)


############################################
#Paramaters
############################################
N_TEAMS = 64
N_ROUNDS = int(math.log2(N_TEAMS))
SIGMOID_STEEPNESS = 1e-2
SCORING = [np.nan] + [2**rd for rd in range(0, 6)]
GAMMA = 1e10
EPSILON_GREEDY = .5
lamb = 1e8
#loss_func = best_score_loss
loss_func = win_prob_loss
epochs =120
bs = 8
lr = 1e-1



def run_opt(save_name, sim_params):

    ############################################
    #Data
    ############################################
    dtype = torch.float
    device = torch.device("cpu")
    res_train, comp_train, res_val, comp_val = [torch.tensor(
        ar, dtype=dtype) for ar in Opt_Helpers.generate_data_std(sim_params)]
    N = len(res_train)


    ############################################
    #Model
    ############################################
    train_ds = TensorDataset(res_train, comp_train)
    train_dl = DataLoader(train_ds, batch_size=bs)

    model, opt = get_model(lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=10)
    pbar = tqdm(range(epochs), ncols=80)



    best_win_prob, best_model = 0, None
    for epoch in pbar:
        model.train()
        epoch_score = np.zeros(N//bs+1)
        i = 0
        for res_batch, comp_batch in train_dl:
            #sometimes explore a new weight
            if uniform(0, 10) < explore_prob(epoch) and epoch < 40:
                with torch.no_grad():
                    #level = randint(1,6)
                    #weight_num = randint(1,63)
                    level, index = random_w()
                    changing_w = model.w[level]
                    changing_w[index] = randint(0, 1)
                    model.w[level] = changing_w

            # Forward pass: Compute predicted y by passing x to the model
            score = model(res_batch)

            # Compute and print loss
            loss, metric = loss_func(score, lamb, punish(model.w), comp_batch)
            epoch_score[i] = loss

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            opt.step()
            opt.zero_grad()

            i += 1
            model.eval()

        with torch.no_grad():
            bracket = output_conversion(get_w(model))
            win_prob, mean_score, std_scores = Opt_Helpers.win_prob(
                bracket,sim_params)
            if win_prob > best_win_prob:
                best_win_prob = win_prob
                best_model = model

        epoch_avg = np.mean(epoch_score)
        scheduler.step(epoch_avg)

        pbar.set_description(
            ("Avg reward: {: 0.6f} | OOS Win Prob: {: 0.3f} | Avg Score: {: 0.1f} | STD Score: {: 0.1f}").format(
                -epoch_avg/GAMMA, win_prob, mean_score, std_scores))

        #print("\n", get_w(model)[2])
    
    with torch.no_grad():
        bracket = output_conversion(get_w(best_model))
        win_p, mean_score, std_scores = Opt_Helpers.win_prob(
            bracket,sim_params)
        print("best model win prob: ", win_prob)
    np.savetxt(save_name, output_conversion(get_w(best_model)), fmt='%d')
    return output_conversion(get_w(best_model)), win_p



#w = get_w(model)
#np.savetxt('test3.txt', output_conversion(get_w(model)), fmt='%d')
#np.savetxt('test1.txt', output_conversion(get_w(model)), fmt='%d')
#np.loadtxt('test1.txt', dtype=int)
