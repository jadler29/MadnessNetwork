{% include lib/mathjax.html %} 

# Optimizing March Madness brackets with Bracket Networks and SGD 

Many people have tackled the problem of predicting the outcomes of March Madness games, or even win probabilities, but even with accurate win probabilities how can we build optimal brackets? The answer is not as obvious as it may seem. 

### The goal of this post is twofold:

- to describe a novel deep-network approach to optimizing decisions in a bracket creation scenario
- to uncover general insights about building winning brackets anyone can use. 

### This post is broken into three sections

1. A primer on the March Madness Bracket creation problem
2. An investigation of the neural architecture I designed to optimize brackets
3. A look at what the results tell us about making good future brackets and beyond

(if you are building a bracket and have no desire to touch any code, the meat is in part 3!)

# Part 1: The Problem

March Madness is arguably the most exiting three weeks in sports TV. 64 college basketball teams play in a win or go home tournament for the National championship (we will not consider the First Four Entry round here). Unless you're a diehard fan of a single team, at least 90% of the fun comes from creating your predicted bracket and ___(without a chance of succes)ly hoping it turns out perfect. 

Many people enter their bracket in a pool with their friends or coworkers. This post will not get you that perfect bracket, but will help you chances at winning that pool. 

## Predicted Outcomes

Much work has been done to estimate the probabilities of game outcomes; we will not tackle this problem here. Instead we will use fivethirtyeight.com's predictions which are an aggregate of several other prediction metrics. Prior to the tournament, fivethirtyeight provides a probability each team has of winning in every round.

![alt text][538]

[538]: https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/538.png

We will take these probabilities as ground truth and call this matrix $$\boldsymbol{Q_{true}}$$.

$$
 \boldsymbol{Q_{true}} =
\begin{pmatrix}
    .99 & .91 & .72 &  .52 & .32 & .19 \\
    .98 & .88 & .73 &  .49 & .32 & .17 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  .05 & .02 &  &  .01 & .005 & .001
\end{pmatrix}
$$

 When I say ground truth I mean that if I were to sample the outcome of a tournament, I would sample it based on $$\boldsymbol{Q_{true}}$$.

More concretely, we can represent a tournament Outcome $$\boldsymbol{O}$$ with a 64 x 6 matrix where each entry represents whether a team wins in a specific round. For instance, since UVA (2nd row) won in 2019, we would have something like

$$
\boldsymbol{O_{2019}}=
\begin{pmatrix}
    1 & 1 & 1 &  1 & 0 & 0 \\
    1 & 1 & 1 &  1 & 1 & 1 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  1 & 0 & 0 &  0 & 0 & 0
\end{pmatrix}
$$

So with $$\boldsymbol{Q_{true}}$$ as ground truth we say
$$\boldsymbol{O} \sim p(\boldsymbol{Q_{true}})$$

Where $$p(\boldsymbol{Q})$$ is the probability distribution* of brackets inferred by $$\boldsymbol{Q}$$

*deriving this probability distribution from a $$\boldsymbol{Q}$$ matrix requires some thought, I will not discuss it here but if you are curious, see the code or ask me!

## The Competition

But if we know the "true" probabilities for each game isn't our task easy? We can just pick the most likely team in each round.. not so fast! As many other works have noted, picking the favorite in every game is not an optimal strategy. If our goal was simply to maximize our bracket's expected score, then, yes, picking ground truth favorites would always be best. However, our problem is more complex, we are trying to maximize the probability our bracket wins (scores highest among our pool). We can illustrate this difference with an analogy.

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/jellybeans2.png)

Imagine you are at a carnival and come across a booth where you task is to guess how many jellybeans are in a jar. If you get the number right, you win $200. Lucky for you, a friend of yours works at the booth and fills up the jar himself. He hints to you that the number of beans approximately follows a normal distribution with mean 100 and standard deviation 10.

![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/empty.png)

So what do you guess? If your like me and just want to maximize your chance at the $200, then 100 of course! But now imagine the rules are different: instead of having to guess number of beans exactly, you just have to be closer than everyone else. Your friend tells you to come by at the very end of the carnival, and not only gives you the bean distribution as before, but he also gives you a list of the other guesses people have made (gray lines).

![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/with_guess.png)

What is your guess now? 

100 is still the most likely single outcome, furthermore, guessing 100 minimizes our expected deviation from the answer. Yet, it is easy to see why guessing 100 gives us a very low probability of winning! If we guess 100, the range for the answer where we win is extremely narrow, because of the high number of nearby competitors. On the other hand, guessing 113 gives us a much better chance at the $200. Though the outcomes for which we win are individually less likely, the weighted sum of winning outcome probabilities (area under the curve) is significantly greater

Guess 100           |  Guess 113
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/100_guess.png) |  ![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/112_guess.png)

By guessing 113, we have maximized the weighted proportion of the outcome space that we "own," or where our guess is the single best of the pack.

Back to March Madness, our goal is the same, find a bracket that "owns" the greatest proportion of the weighted tournament outcomes with respect to our pool of competitors. Unfortunately, brackets are far more complex than a single random variable, moreover we don't know exactly what other's brackets are. Therefore, we will need a more carful approach.

To start, we can learn about how other's make their brackets with ESPN's Who Picked Whom. Every year, ESPN keeps updated statistics on picks for all created brackets.

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/wpw.png)

We call this $$\boldsymbol{Q_{pop}}$$. If we assume that our pool competitors are not far from the general people who create ESPN brackets, we can sample a competitors bracket, $$\boldsymbol{C}$$ (a 64 x 6 matrix just like $$\boldsymbol{O}$$) based on $$\boldsymbol{Q_{pop}}$$. That is

$$\boldsymbol{C} \sim p(\boldsymbol{Q_{pop}})$$

## Scoring

Now that we have a distribution of bracket outcomes and of opponent's brackets we can start to think about building our bracket. First lets touch on scoring.

March madness brackets are usually scored in some variation of the same format. Each time a team is correctly called to win in a round $$r$$, $$2^{r-1}$$ points are awarded. Thus, given an outcome $$\boldsymbol{O}$$ and bracket $$\boldsymbol{B}$$
, we can easily generate a score, $$score(\boldsymbol{B},\boldsymbol{O}).$$

## Building Our Bracket

Now we can define our goal. Recall that we aim to maximize the probability that our bracket is the best of the bunch.

$$\max_{\boldsymbol{B}} \quad P(score(\boldsymbol{B},\boldsymbol{O}) > score(\boldsymbol{C_i},\boldsymbol{O}) \quad \forall i \in 1...n\_competitors) $$

Using the law of large numbers we can estimate this probability with a large number of events: 

$$\max_{\boldsymbol{B}} \quad \frac{1}{BIG}\sum_{t=0}^{BIG}I(score(\boldsymbol{B},\boldsymbol{O_t}) > score(\boldsymbol{C_{it}},\boldsymbol{O_t}) \quad \forall i \in 1...n\_competitors)$$

where $$I$$ is the indicator function

Ok so are we ready to go? We can read up on some classic optimization methods and chug away... not quite. In fact, once we include the constraints that our bracket $$\boldsymbol{B}$$ is a valid bracket (doesn't pick every team to win every game for instance), the optimization problem is very messy and highly non-convex.. ugh.

We need a clever way to tackle this problem.. Its time for part 2.

# Part 2: Optimizing with "Bracket Networks"

As mentioned at the end of part 1, our problem of optimizing our bracket is non-convex. And, unfortunately, non-convex optimization problems are hard to solve. Yet, one set of non-convex  optimization problems is encountered and dealt with often in todays world, namely, optimizing the weights of deep neural networks.

Despite the theoretical difficulty of optimizing these networks, in practice, stochastic gradient descent seems to give us good (though not necessarily globally optimal) solutions. Is there a way we can apply the same methods to our problem?

## A bracket as a neural network

The goal is here is to model the scoring of our bracket with a deep-network structure. If we can do so successfully, we can use backpropogation and SGD to optimize our score.

For simplicity, consider a tournament of only 4 teams (with a slightly biased choice of universities)

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/2x2.png)


In this case our decisions in creating a bracket can be reduced to 3 binary choices.

$$w_{1,1} =
\begin{cases}
    1, & \text{if we pick Duke to win in round 1}\\
    0, & \text{if we pick MIT}
\end{cases}
$$

$$\qquad w_{1,2} =
\begin{cases}
    1, & \text{if we pick Northwestern to win in round 1}\\
    0, & \text{if we pick Harvard}
\end{cases}
$$

$$w_{2,1} =
\begin{cases}
    1, & \text{if we pick our winner from MIT vs. Duke to win in Round 2}\\
    0, & \text{if we pick our winner from NU vs Harvard to win in Round 2}
\end{cases}
$$

Now consider the proposed structure:

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/simple_net.png)

Note that $$r_i =$$ The scoring coefficient for round $$i$$; $$S$$ represents the sigmoid function $$S(x) = \frac{1}{1+e^{-x}}$$

This structure is engineered such that if the actual numbers of wins is given as input to the network, $$z_{output}$$ gives the score of our created bracket against the actual result!


The best way to understand why it works is to run through a theoretical example yourself. Consider a created bracket and an actual outcome, and run a forward pass through the network to get the score $$z_{output}$$.

Intuitively though, $$z_{r,n}$$  represents how many wins the team we picked (in block $$n$$) to win in round $$r$$ actually got. Thus a $$z_{1,1} =1$$ means that the team we picked to win block 1, round 1 actually got 1 win in the tournament, thus the pick was right. On layer 2 however, $$z_{2,1}=1$$  means that the team we picked to win the mini tourney only won 1 game, thus this pick was not correct.

The green sigmoid* nodes check whether z values have the requisite value for a correct guess and "fire" 1 if so, 0 otherwise. These values are then properly weighted by the corresponding round scoring $$r$$ before being summed out for the total score $$z_{output}$$

This framework is easily extended to fit a 64 team tournament.

**an obvious question is why use sigmoids here? Can't we model the system more perfectly with a step function? Yes, this is true, however, in practice the sigmoid provides significant benefits when we get to optimization; the sigmoid allows for much richer gradients. We also have the ability to tune the steepness of the sigmoid; the tradeoff being a steeper sigmoid better resembles the step function, a more gradual sigmoid provides more descriptive gradients*

**additionally, why the -.5 (or -1.5)? The formula for the sigmoid activation for round $$r$$ is $$S(z_{r,n}-r+.5)$$. We subtract $$r$$ as to to make sure in round 2, for instance, the input is at least 2 (2 wins). However, the sigmoid function at 0, $$S(0)=.5$$, we want our activation $$a(0) \approx 0$$ and $$a(1) \approx 1$$, we can achieve this by adding .5 to shift our function*

## Extending the Structure to Winning

There are two things we are missing in our model of our problem so far:

 1. A way to account for the competitors brackets
 2. A way to account for the uncertainty in outcomes (and in competitors brackets)

Fortunately, our structure lends itself to accounting for these elements easily. We will stick with the 4 team tourney for ease of demonstration.

### Accounting for victory in pool

To check for a victory in our pool, we want to see if our score, $$z_{output}$$, is greater than the maximum of our competitor's scores. In the case the we have 3 competing brackets, for instance, we sample 3 competing brackets $$C$$ as discussed in part 1, score them based on the actual tourney result, take the maximum of the 3, and then can compare it with $$z_{output}$$ to see who wins. We can execute this comparison with a sigmoid.

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/full_net.png)


### Accounting for uncertainty

Now, we have discussed how our structure models our score for one tourney realization, but we must account for the fact that when creating a bracket, we, of course, don't know how the tournament will turn out.

As established in part 1, we have a ground truth from which we can sample tournament outcomes. We can use this to sample a dataset of simulated outcomes. For simplicity, with 3 simulations we have

![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/tensor.png) 

For each simulation, we sample a *new* set of competing brackets from $$p(\boldsymbol{Q_{pop}})$$ and score them with their corresponding simulated outcome. We then can take the maximum score between competitors for each simulation, $$z_{pool}^t$$. 

We also can easily transform our simulated outcomes into the desired format for of the number of wins per team. 

This leaves us with input matrix $$\boldsymbol I$$, (n_simulations x n_teams) and vector $$\vec{z}_{pool}$$, (n_simulations x 1)


$$\boldsymbol I=
\begin{pmatrix} 
3 & 2 & \dots &  5 \\
6 & 3 & \dots &  4  \\

 4 & 6 & \dots &  4  
 \end{pmatrix}
\qquad 
\vec{z}_{pool}=
\begin{pmatrix} 
164  \\
159  \\
136
 \end{pmatrix}$$

Essentially, we have created a dataset for our network where the number of observations is the number of simulations. If we increase the number of simulations to a high number** we can account for uncertainty in both the tournament outcome and competitors' brackets.

It is important to remember that while we are re-sampling opponents brackets at each simulation, we are limiting our algorithm to a single bracket choice (optimized weights) across simulations. This is critical since we do not get a choose a bracket after we have seen the results!

***In practice ~2000 simulations accounts for the randomness of the process*

## Implementing with PyTorch

An advantage of our neural framework is that we can extend popular software frameworks to implement our model. We will use PyTorch here (learn more: <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)>.

For now, we will not touch on the general logistics of implementing PyTorch models and will focus on the unique aspects of our model.

Before we get to defining the entire network, we can create a function to generalize the repetitive nature of our our layers.


```python
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
```

Now we can define our network using our function

```python
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
```

## Optimization

Our PyTorch implementation, t

# Part 3: Results and Insights

TODO

![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/wpw_res.png) 

![](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/538_res.png) 
