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


Much work has been done to estimate the probabilities of game outcomes; we will not tackle this problem here. Instead we will use fivethirtyeight.com's predictions which are an aggregate of several other prediction metrics. Prior to the tournament, fivethirtyeight provides a probability each team has of winning in every round. 



![alt text][538]

[538]: https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/538.png 



We will take these probabilities as ground truth and call this matrix $\bold{Q_{true}}$.
 
$$
 \boldsymbol{Q_{true}} =
\begin{pmatrix}
    .99 & .91 & .72 &  .52 & .32 & .19 \\
    .98 & .88 & .73 &  .49 & .32 & .17 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  .05 & .02 &  &  .01 & .005 & .001
\end{pmatrix}
$$

 When I say ground truth I mean that if I were to sample the outcome of a tournament, I would sample it based on $\bold{Q_{true}}$.


More concretely, we can represent a tournament Outcome $\bold{O}$ with a 64 x 6 matrix where each entry represents whether a team wins in a specific round. For instance, since UVA won in 2019, we would have something like

[//]: # ($O_{rt} =
\begin{cases}
    1, & \text{if team }r \text{ wins in round }t \\
    0, & \text{otherwise}
\end{cases}
$ )



$$
\bold{O_{2019}}=
\begin{pmatrix}
    1 & 1 & 1 &  1 & 0 & 0 \\
    1 & 1 & 1 &  1 & 1 & 1 \\
    \vdots & \vdots & \vdots & \vdots & \vdots &\vdots \\
  1 & 0 & 0 &  0 & 0 & 0
\end{pmatrix}
$$


So with $\bold{Q_{true}}$ as ground truth we say 
$$\bold{O} \sim P(\bold{Q_{true}})$$

*deriving this probability distribution is actually from a $\bold{Q}$ matrix requires some thought, I will not discuss it here but if you are curious, see the code or ask me! 




---


But if we know the "true" probabilities for each game isn't our task easy? We can just pick the most likely team in each round.. not so fast! As many other works have noted, picking the favorite in every game is not an optimal strategy. If our goal was simply to maximize our bracket's expected score, then, yes, picking ground truth favorites would always be best. Our problem is more complex, we are trying to maximize the probability our bracket wins (scores highest among our pool). We can illustrate this difference with an analogy.


![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/jellybeans2.png)


Imagine you are at a carnival. You come across a booth where you task is to guess how many jellybeans are in a jar. if you get it right, you win $200. Lucky for you, a friend of yours works at the booth and fills up the jar himself. He hints to you that the number of beans follows a normal distribution with mean 100 and standard deviation 10. 

<image>

So what do you guess? If your like me and just want to maximize your chance at the $200, then 100 of course! But now imagine the rules are different: instead of having to guess number of beans exactly, you just have to be closer than everyone else to win. Your friend tells you to come by at the very end of the carnival, and not only gives you the bean distribution as before, but he also gives you a list of the other guesses people have made. 

<image>

What is your guess now? ... 


Here, we have maximized the weighted proportion of the outcome space that we "own," or where our guess is the single best of the pack. 

Back to March Madness, our goal is the same, find a bracket that "owns" the greatest proportion of the weighted tournament outcomes with respect to our pool of competitors. While we dont know exactly what other's brackets are, we can guess; enter ESPN'S Who Picked whom. 

Every year, ESPN posts statistics on how how all entered brackets are created.

![alt text](https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/wpw.png)

We call this $\bold{Q_{pop}}$. If we assume that our pool competitors are not far from the general people who create ESPN brackets, we can sample a competitors bracket, $\bold{C}$ (a 64 x 6 matrix just like $\bold{O}$) based on $\bold{Q_{pop}}$. That is

$$\bold{C} \sim P(\bold{Q_{pop}})$$




Who picked 


```python
#Syntax highlighted code block

```

