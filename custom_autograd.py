# -*- coding: utf-8 -*-
import torch


class Knockout_Round_Layer(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input,weight):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        weight = torch.clamp(weight,0,1)
        ctx.save_for_backward(input, weight)
        w_new = torch.empty(2*len(weight), dtype=weight.dtype)
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
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, weight = ctx.saved_tensors
        dL_dO = grad_output.clone()
        is_in_of_bounds = (weight < 1) * (weight > 0)
        dO_dw = (input[:, 0::2]-input[:, 1::2])#*is_in_of_bounds.float()
        dL_di = dL_dO *dO_dw 

        return None, dL_di


class MadnessNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MadnessNet, self).__init__()
        #self.knockout= Knockout_Round_Layer.apply
        teams = 4
        rd=1
        self.w1 = torch.nn.Parameter(
            torch.clamp(
                torch.randn(
                    int(teams*(.5)**(rd)),
            ),0, 1)
            )
        #self.w1 = torch.randn(int(teams*(.5)**(rd)))
        rd += 1
        self.w2 = torch.nn.Parameter(torch.randn(int(teams*(.5)**(rd))))
        #self.register_parameter("w1", self.w1)
        
        #self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        knockout = Knockout_Round_Layer.apply
        z = [0]*6
        z[0] = x
        z[1] = knockout(x, self.w1)
        z[2] = knockout(z[1], self.w2)
        #z.append(self.knockout(x))
        #for rd in range(6):
        #    z.append(self.knockout(z[-1]))
        score = torch.sigmoid(-1+1*z[1]).sum() + torch.sigmoid(-2+1*z[2]).sum()
        
        return score



dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 100
n_teams =4

# Create random Tensors to hold input and outputs.
x = torch.randn(N, n_teams, device=device, dtype=dtype)
x = torch.tensor([[2, 0, 0, 1],
                  [2, 0, 0, 1],
                  [2, 0, 0, 1],
                  [2, 0, 0, 1],
                  [2, 0, 0, 1],
                  [1, 0, 2, 0],
                  [0, 2, 0, 1]
                  ], dtype=dtype)

model=MadnessNet()

model(x)

model.named_parameters()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
lamb = 10
#for t in range(500):
for t in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    punishment = torch.relu(-model.w1).sum() + torch.relu(model.w1-1).sum()
    loss = -y_pred + lamb * punishment
    print(loss.item())

    model.w1
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.w1