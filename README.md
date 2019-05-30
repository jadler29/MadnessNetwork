{% include lib/mathjax.html %} 

# Optimizing March Madness brackets with Bracket Networks and SGD 

Many people have tackled the problem of predicting the outcomes of March Madness games, or even win probabilities, but even with accurate win probabilities how can we build optimal brackets? The answer is not as obvious as it may seem. 

#### The goal of this post is twofold: 
- to describe a novel deep-network approach to optimizing decisions in a bracket creation scenario
- to uncover general insights about building winning brackets anyone can use. 

#### This post is broken into three sections
1. A primer on the March Madness Bracket creation problem
2. An investigation of the neural architecture I designed to optimize brackets
3. A look at what the results tell us about making good future brackets and beyond

(if you are building a bracket and have no desire to touch any code, the meat is in part 3!)

# Part 1: The Problem

March Madness is arguably the most exiting three weeks in sports TV. 64 college basketball teams play in a win or go home tournament for the National championship (we will not consider the First Four Entry round here). Unless you're a diehard fan of a single team, at least 90% of the fun comes from creating your predicted bracket and ___(without a chance of succes)ly hoping it turns out perfect. 

Many people enter their bracket in a pool with their friends or coworkers. I can't help you get the perfect bracket, not even close, but I can help you win that pool. 

## Predicted Outcomes

Much work has been done to estimate the probabilities of game outcomes; we will not tackle this problem here. Instead we will use fivethirtyeight.com's predictions which are an aggregate of several other prediction metrics. Prior to the tournament, fivethirtyeight provides a probability each team has of winning in every round. 



![alt text][538]

[538]: https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/538.png 



We will take these probabilities as ground truth and call this matrix $\boldsymbol{Q_{true}}$.
 
$$
 \boldsymbol{Q_{true}} =
\begin{pmatrix}
    .99 & .91 & .72 &  .52 & .32 & .19 \\
    .98 & .88 & .73 &  .49 & .32 & .17 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  .05 & .02 &  &  .01 & .005 & .001
\end{pmatrix}
$$

 When I say ground truth I mean that if I were to sample the outcome of a tournament, I would sample it based on $\boldsymbol{Q_{true}}$.


More concretely, we can represent a tournament Outcome $\boldsymbol{O}$ with a 64 x 6 matrix where each entry represents whether a team wins in a specific round. For instance, since UVA won in 2019, we would have something like

$$
\boldsymbol{O_{2019}}=
\begin{pmatrix}
    1 & 1 & 1 &  1 & 0 & 0 \\
    1 & 1 & 1 &  1 & 1 & 1 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  1 & 0 & 0 &  0 & 0 & 0
\end{pmatrix}
$$


So with $\boldsymbol{Q_{true}}$ as ground truth we say 
$$\boldsymbol{O} \sim p(\boldsymbol{Q_{true}})$$

Where $p(\boldsymbol{Q})$ is the probability distribution* of brackets inferred by $\boldsymbol{Q}$

*deriving this probability distribution is actually from a $\boldsymbol{Q}$ matrix requires some thought, I will not discuss it here but if you are curious, see the code or ask me! 




## The Competition


But if we know the "true" probabilities for each game isn't our task easy? We can just pick the most likely team in each round.. not so fast! As many other works have noted, picking the favorite in every game is not an optimal strategy. If our goal was simply to maximize our bracket's expected score, then, yes, picking ground truth favorites would always be best. However, our problem is more complex, we are trying to maximize the probability our bracket wins (scores highest among our pool). We can illustrate this difference with an analogy.


![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/jellybeans2.png)


Imagine you are at a carnival. You come across a booth where you task is to guess how many jellybeans are in a jar. if you get it right, you win $200. Lucky for you, a friend of yours works at the booth and fills up the jar himself. He hints to you that the number of beans follows a normal distribution with mean 100 and standard deviation 10. 

image

So what do you guess? If your like me and just want to maximize your chance at the $200, then 100 of course! But now imagine the rules are different: instead of having to guess number of beans exactly, you just have to be closer than everyone else. Your friend tells you to come by at the very end of the carnival, and not only gives you the bean distribution as before, but he also gives you a list of the other guesses people have made. 

image

What is your guess now? ... 

Here, we have maximized the weighted proportion of the outcome space that we "own," or where our guess is the single best of the pack. 

Back to March Madness, our goal is the same, find a bracket that "owns" the greatest proportion of the weighted tournament outcomes with respect to our pool of competitors. While we don't know exactly what other's brackets are, we can guess; enter ESPN'S Who Picked whom. 

Every year, ESPN posts statistics on how how all entered brackets are created.

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/wpw.png)


We call this $\boldsymbol{Q_{pop}}$. If we assume that our pool competitors are not far from the general people who create ESPN brackets, we can sample a competitors bracket, $\boldsymbol{C}$ (a 64 x 6 matrix just like $\boldsymbol{O}$) based on $\boldsymbol{Q_{pop}}$. That is

$$\boldsymbol{C} \sim p(\boldsymbol{Q_{pop}})$$


## Scoring

Now that we have a distribution of bracket outcomes and of opponent's brackets we can start to think about building our bracket. First lets touch on scoring. 

March madness brackets are usually scored in some variation of the same format. Each time a team is correctly called to win in a round r, $2^{r-1}$ points are awarded. Thus, given an outcome $\boldsymbol{O}$ and bracket $\boldsymbol{B}$
, we can easily generate a score, $score(\boldsymbol{B},\boldsymbol{O}).$

## Building Our Bracket

Now we can define our goal. Recall that we aim to maximize the probability that our bracket is the best of the bunch.

$$\max_{\boldsymbol{B}} \quad P(score(\boldsymbol{B},\boldsymbol{O}) > score(\boldsymbol{C_i},\boldsymbol{O}) \quad \forall i \in 1...n\_competitors) $$


Using the law of large numbers we can estimate this probability with a large number of events: 

$$\max_{\boldsymbol{B}} \quad \frac{1}{BIG}\sum_{t=0}^{BIG}\mathbb{I}(score(\boldsymbol{B},\boldsymbol{O_t}) > score(\boldsymbol{C_{it}},\boldsymbol{O_t}) \quad \forall i \in 1...n\_competitors)$$

where $\mathbb{I}$ is the indicator function

Ok so are we ready to go? We can read up on some classic optimization methods and chug away... not quite. In fact, once we include the constraints that our bracket $\boldsymbol{B}$ is a valid bracket (doesnt pick every team to win every game for instance), the optimization problem is very messy and highly non-convex.. ugh. 

We need a clever way to tackle this problem.. Its time for part 2. 



# Part 2: Optimizing with "Bracket Networks"

As mentioned at the end of part 1, our problem of optimizing our bracket is non-convex. And, unfortunately, non-convex optimization problems are hard to solve. Yet, one set of non-convex  optimization problems is encountered and dealt with often in todays world, namely, optimizing the weights of deep neural networks. 

Despite the theoretical difficulty of optimizing these networks, in practice, stochastic gradient descent seems to give us good (though not necessarily globally optimal) solutions. Is there a way we can apply the same methods to our problem? 



## A bracket as a neural network

The goal is here is to model the scoring of our bracket with a deep-network structure. If we can do so successfully, we can use backpropogation and SGD to optimize our score. 



For simplicity, consider a tournament of only 4 teams (with a slightly biased choice of universities)


![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/2x2.png)


In this case our decisions in creating a bracket can be reduced to 3 binary choices. 
$w_{1,1} =
\begin{cases}
    1, & \text{if we pick Duke to win in round 1}\\
    0, & \text{if we pick MIT}
\end{cases}
$
$w_{1,2} =
\begin{cases}
    1, & \text{if we pick Northwestern to win in round 1}\\
    0, & \text{if we pick Harvard}
\end{cases}
$

$w_{2,1} =
\begin{cases}
    1, & \text{if we pick our winner from MIT vs. Duke to win in Round 2}\\
    0, & \text{if we pick our winner from NU vs Harvard to win in Round 2}
\end{cases}
$

Now consider the proposed structure:

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/simple_net.png)

Note that $r_i =$ The scoring coefficient for round $i$; $S$ represents the sigmoid function $S(x) = \frac{1}{1+e^{-x}}$

This structure is engineered such that if the actual numbers of wins is given as input to the network, $z_{output}$ gives the score of our created bracket against the actual result! 


The best way to understand why it works is to run through a theoretical example yourself. Consider a created bracket and an actual outcome, and run a forward pass through the network to get the score $z_{output}$. 

Intuitively though, $z_{r,n}$  represents how many wins the team we picked (in block $n$) to win in round $r$ actually got. Thus a $z_{1,1} =1$ means that the team we picked to win block 1, round 1 actually got 1 win in the tournament, thus the pick was right. On layer 2 however, $z_{2,1}=1$  means that the team we picked to win the mini tourney only won 1 game, thus this pick was not correct. 

The green sigmoid* nodes check whether z values have the requisite value for a correct guess and "fire" 1 if so, 0 otherwise. These values are then properly weighted by the corresponding round scoring $r$ before being summed out for the total score $z_{output}$

This framework is easily extended to fit a 64 team tournament. 

*an obvious question is why use sigmoids here? Can't we model the system more perfectly with a step function? Yes, this is true, however, in practice the sigmoid provides significant benefits when we get to optimization; the sigmoid allows for much richer gradients

addtionally why the -.5 (or -1.5)? ......


## Extending the Structure to Winning 

There are two things we are missing in our model of our problem so far:
 1. A way to account for the competitors brackets
 2. A way to account for the uncertainty in outcomes (and in competitors brackets)

Fortunately, our structure lends itself to accounting for these elements easily. We will stick with the 4 team tourney for ease of demonstration. 



To check for a victory our pool, we want to see our score, $z_{output}$, is greater than the maximum of our competitor's scores. In the case the we have 3 competing brackets, for instance, we sample 3 competing brackets $C$ as discussed in part 1, score them based on the actual tourney result, take the maximum of the 3, and then can compare it with $z_{output}$ to see who wins. We can execute this comparison with a sigmoid 


![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/full_net.png)






```python
@staticmethod
def forward(ctx, input, weight):
    """
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    """
    weight = torch.clamp(weight, 0, 1)
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

```

