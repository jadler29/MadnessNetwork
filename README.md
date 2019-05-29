## Optimizing March Madness brackets with Bracket Networks and SGD

Many people have tackled the problem of predicting the outcomes of March Madness games, or even win probabilities, but even with accurate win probabilities how can we build optimal brackets? The answer is not as obvious as it may seem. 

#### The goal of this post is twofold: 
- to describe a novel deep-netowrk approach to optimizing decisions in a bracket creation scenario
- to uncover general insights about building winning brackets anyone can use. 

#### This post is broken into three sections
1. A primer on the March Madness Bracket creation scenario
2. An investiagtion of the neural archetecture we design to optimize brackets
3. A look at what the results tell us about making good future brackets and beyond


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

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/jadler29/MadnessNetwork/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
