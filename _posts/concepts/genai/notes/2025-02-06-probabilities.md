---
layout : mermaid
type : notes
title : "Probabilities"
---

### How to compute the probabilities?

Language Model is a model that computers either of the ones below, are known as language models.
   1. Obective - compute the probability of a sentence or sequence of words.
        $P(W) = P(w1,w2,w3,...,wn)$
   2. Related task - computing the probability of the upcoming word.
        $P(w4|w1,w2,w3)$

**Probaility of the entire sentence : $P(W) = P(w1,w2,w3...wn)$**

P(The, water, of, Walden, Pond, is, so, beautiful, blue)

There is a difference between the following - 

1. $P(B|A)$ : Probability of B given A : Event B has happened in past. A is happening now.
2. $P(A,B)$ : Joint probability of A and B. Or probability when both the events A and B are happening simultaneously.

$P(B|A) = P(A \bigcap B) / P(A)$ 
or, $P(B|A) = P(A,B) / P(A)$
or, $P(A,B) = P(B/A) \times P(A)$
or, $P(A,B) = P(A) \times P(B|A)$
ie. probability of two events A and B happening together (joint probability) is probability of A multiplied by probability of B when A has already happened.

3. **extending it to multiple events we can write**

$P(A, B, C, D)  = P(A) \times P(B|A) \times P(C|A,B) \times P(D|A,B,C)$
TO get the intuition, following is the chain of thought - 
- first the event A happened. So, the probability is P(A) as nothing else has happened now.
- second the event B happened. Event A has already happened in the last step. so the probability of B, we need to compute $P(B|A)$, i.e., probability of B when A has already happened.
- now, the third event C happened. A and B has already happened. So, probability of C would be, $P(C|A,B)$, because A and B has already happened. So,  
- now, the fourth event D happened. A, B and C has already happened by now. So, porbability of D when A, B and C has already happened is $P(D|A,B,C)$

4. P(blue|the water of walden pond is so beautifully) = C(the water of walden pond is so beautifully blue)/C(the water of walden pond is so beautifully)

5. Generalizing the above 
$P(w_{1:n}) = P(w1) \times P(w2|w1) \times P(w3|w_{1:2} \times \prod()$
5. We will never see enough data for estimating all these!!