---
title : "Overview"
source : https://www.youtube.com/watch?v=k7RM-ot2NWY&list=PL0-GT3co4r2y2YErbmuJw2L5tW4Ew2O5B&index=3 
---

## Overview

Understanding the geometric intuition of linear algebra.

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
4. Note that in the case of linear combination, the scaling of vectors are done with a multiplication of a scalar. So, the direction of each of the vectors remain the same (except when the scalar is negative, and the direction of the vector reverses).  If direction of each vector remains the same, the direction of their resultant also remains the same. So, it is known as linear combination. Because all the vectors - input and output, both follow the same line. they just linearly scale.
5. The span is basically a plane / sheet passing through the input vectors. This is because obviously the resultant vector will fall on the same plan as the input vectors. And since the direction of the input vectors are not changing (except opposite direction when scalar is negative), however keeping the magnitude of the angle constant, the resultant vectors remain on the same plane passing through all the input vectors. Span is like the webbed feet of a duck. The fingers of the webbed feet are the input vectors. And the palmated join of the fingers is the span.
6. So, if the input vectors are in 3 dimensions, the span will be a 2D plane passing through the input vectors. The orientation of this span might change based on the initial direction of the vectors. Earlier the span plan was same as the plane of x-y axis. But in this case the plane might be aligned with x-y, x-z, y-z or any other plan intersecting them.

### vectors as points

1. When dealing with collection of multiple vectors