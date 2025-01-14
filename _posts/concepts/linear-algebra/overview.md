---
title : "Overview"
source : https://www.youtube.com/watch?v=k7RM-ot2NWY&list=PL0-GT3co4r2y2YErbmuJw2L5tW4Ew2O5B&index=3 
---

## Overview

### vectors as matrix

1. vectors can be represented as an arrow on the origin. 
2. they can also be represented as a matrix with one column.
   $ \vec{v} = \begin{bmatrix} 1\\ 2 \end{bmatrix} $
3. vectors can also be represented as scaled basis vectors.
4. $ \vec{v} = 2\vec{i} + 3\vec{j} $ : it is like scaling the basis vectors and adding them together.
5. $ \vec{v} = \begin{bmatrix} 2\\ 3 \end{bmatrix} = 2\begin{bmatrix} 1\\ 0 \end{bmatrix} + 3\begin{bmatrix} 0\\ 1 \end{bmatrix} $
6. $ \vec{v} = \begin{bmatrix} 2\\ 3 \end{bmatrix} = 2\vec{i} + 3\vec{j} $

### basis vectors

1. here scalars 2 and 3 are scaling the basis vectors and creating a new vector. This basis vectors can be x axis or y axis. Then they show up as $\vec{i}$ and $\vec{j}$
2. we can change the basis vectors to any other vector and then scale it.
3. anytime we mention vectors numerically, it depends on an implicit choice of what basis vectors we are using. Each time you scale and add two basis vectors to get a new vector, you say that the new vector is the linear combination of the basis vectors.

### linear combination & span

$ \vec{v} = a\vec{i} + b\vec{j} $ - this is a linear combination of the basis vectors.

1. The span of a set of vectors is the set of all possible vectors that can be created with a linear combination of those vectors. The span of $\vec{v}$ and $\vec{w}$ is the entire 2D plane is the set of all their linear combinations! 
2. Span is a way to ask what are all the possible vectors that you can reach through vector addition and scalar multiplication of a given set of vectors. 
3. Span or linear transformation can also be seen as a transformation from one vector set to another vector set, by applying the same linear combination of the original vectors.

### vectors as points
4.   