## Optimizing March Madness brackets with Bracket Networks and SGD

Many people have tackled the problem of predicting the outcomes of March Madness games, or even win probabilities, but even with accurate win probabilities how can we build optimal brackets? The answer is not as obvious as it may seem. 

#### The goal of this post is twofold: 
- to describe a novel deep-netowrk approach to optimizing decisions in a bracket creation scenario
- to uncover general insights about building winning brackets anyone can use. 

#### This post is broken into three sections
1. A primer on the March Madness Bracket creation scenario
2. An investiagtion of the neural archetecture we design to optimize brackets
3. A look at what the results tell us about making good future brackets and beyond

(if you are building a bracket and have no desire to touch any code, feel free to skip to part 3!)




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

