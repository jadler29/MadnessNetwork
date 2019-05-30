

# Optimizing March Madness brackets with Bracket Networks and SGD

Many people have tackled the problem of predicting the outcomes of March Madness games, or even win probabilities, but even with accurate win probabilities how can we build optimal brackets? The answer is not as obvious as it may seem. 

#### The goal of this post is twofold: 
- to describe a novel deep-network approach to optimizing decisions in a bracket creation scenario
- to uncover general insights about building winning brackets anyone can use. 

#### This post is broken into three sections
1. A primer on the March Madness Bracket creation problem
2. An investigation of the neural architecture we design to optimize brackets
3. A look at what the results tell us about making good future brackets and beyond

(if you are building a bracket and have no desire to touch any code, the meat is in part 3!)

# Part 1: The Problem

March madness is arguably the most exiting three weeks in sports TV. 68 college basketball teams play in a win or go home tournament for the National championship. Unless you're a diehard fan of a single team, at least 90% of the fun comes from creating your predicted bracket and ___(without a chance of succes)ly hoping it turns out perfect. 

Many people enter there bracket in a pool with their friends or coworkers. I can't help you get the perfect bracket, not even close, but I can help you win that pool. 


![equation](https://latex.codecogs.com/gif.latex?O_t%3D%5Ctext%20%7B%20Onset%20event%20at%20time%20bin%20%7D%20t)


![alt text][logo]

[logo]: https://raw.githubusercontent.com/jadler29/MadnessNetwork/master/old/538.png "538"

We will call this matrix $$Q$$ the other $$P$$


$$
Q = P^2
$$






![equation](https://latex.codecogs.com/gif.latex?O_t%3D%5Ctext%20%7B%20Onset%20event%20at%20time%20bin%20%7D%20t)

\\[ \frac{1}{n^{2}} \\]

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

