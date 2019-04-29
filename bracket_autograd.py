import torch


dtype = torch.float
#device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.

x = torch.tensor([[2, 0, 1, 0],
                  [2, 0, 1, 0],
                  [2, 0, 1, 0],
                  [2, 0, 1, 0],
                  [2, 0, 1, 0],
                  [1, 0, 2, 0],
                  [0, 2, 0, 1]
                  ],dtype=dtype)

N, D_in = x.shape[0], x.shape[1]
# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
#w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
#w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

w11 = torch.randn(1, 1, requires_grad=True,dtype=dtype)
w12 = torch.randn(1, 1, requires_grad=True, dtype = dtype)
w21 = torch.randn(1, 1, requires_grad=True, dtype = dtype)
c1 = 1
c2=2

learning_rate = 1e-6
optimizer = torch.optim.SGD([w11,w12,w21], lr=1e-2)
for t in range(1000):
    z21 = w11*x[:, 0] + (1-w11)*x[:, 1]
    z22 = w12*x[:, 2] + (1-w12)*x[:, 3]
    a21 = torch.sigmoid(-1+1*z21)
    a22 = torch.sigmoid(-1+1*z22)

    z31 = a21*w21 + (1-w21)*a22
    a31 = torch.sigmoid(-2+1*z31)
    lamb = 5
    langrange = torch.relu(w11-1) + torch.relu(-w11) + \
        torch.relu(w12-1) + torch.relu(-w12) + \
        torch.relu(w21-1) + torch.relu(-w21)
    y = (a21+a22)*c2 + a31*c3
    


    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = -y.sum() + lamb*langrange
    print(t, loss.item(), w11)

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


torch.ones
''' cant autgrad this:
l2 = torch.tensor([[w11, 0],
                    [1-w11, 0],
                    [0, w12],
                    [0, 1-w12]], dtype=dtype, requires_grad = True)

l3 = torch.tensor([[w21],
                    [1-w21]], dtype=dtype, requires_grad=True)

n2 = torch.sigmoid(-10+10*x.mm(l2))

    n3 = torch.sigmoid(-20+10*n2.mm(l3))

    c2 = 1
    c3 = 3

    y = (n2*c2).sum() + (n3*c3).sum()
'''

optimizer = torch.optim.Adam([w1, w2], lr=1e-2)
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    # Manually zero the gradients after updating weights
    #w1.grad.zero_()
    #w2.grad.zero_()
    




for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

