---
layout : mermaid
type: assignment
title: "NLP with Deep Learning"
trainer: Prof. Pawan Goyal, IIT KGP
---

## Assignment 0

1. Which of the following is critical application of language modeling in NLP?
    - [ ] Identifying sentence structure in a text
    - [ ] Predicting the next word in a sequence
    - [ ] Recognizing the named entities like people and locations.
    - [ ] Translating texts from one format to the other.

> Answer:
    Predicting the next word in a sequence

2. The markov assumption in n-gram models simplifies computations by:
    - [ ] Treating all words in a sequence as equally probable.
    - [ ] Using only immediate preceding word(s) for prediction.
    - [ ] Ignoring the probability of rare words entirely.
    - [ ] Focusing on entire sentence for probability estimation.

> Answer:
    Using only immediate preceding word(s) for prediction.
    This assumption reduces the computational complexity because:
    1. Finite Context Window: Instead of considering all previous words in a sequence, the model only considers the last n-1 words for an n-gram model. for example
       - in a bi-gram model (n=2), a word depends on the immediately preceding word.
       - in a tri-gram model (n=3), a word depends on the two preceding words.
    2. Reduced probabilities to estimate: The number of probabilities that needs to be estimated is reduced. For an n-gram model, the probability $ P(w_i|w_1,w_2,w_3,...,w_{i-1})$ is approximated as $ P(w_i|w_{i-n+1},...,w_{i-1})$.
    3. Simpler storage and Processing: It becomes computationally easier to store and process the probabilities of the n-grams instead of all possible sequences. 

3. Why is add-1 smoothing crucial in probabilistic language modeling?
    - [ ] To avoid zero probabilities for unseen words.
    - [ ] To reduce the impact of rare words on overall probability.
    - [ ] To ensure that the sum of probabilities is always 1.
    - [ ] To make the model more robust to noise in the data.
  
> Answer:
    To avoid zero probabilities for unseen words. In language models some word combinations (n-gram) in the test data may not have appeared in the training set. Without smoothening, the probability of these unseen events would be zero. 
    - zero probability can make the entire probability of the entire sequence or sentence as zero in multiplicative models.
    - This can severely impact the model performance.

4. Which statement best explains perplexity as an evaluation metrics for language models?
    - [ ] It directly measures the grammatical accuracy of sentences.
    - [ ] It computes the inverse probability of a test set, normalized by length.
    - [ ] It determines overall training efficiency of a language model.
    - [ ] It evaluates semantic similarities between generated and real sentences.

> Answer:
    It computes inverse probability of a test set, normalized by length. Perplexity quantifies the uncertainty of the model when predicting the next word in a sequence.
    $$ PP(P,S) = exp(-\frac{1}{N}\sum_{i=1}^n \log P(w_i|w_1,w_2,w_3,...,w_{i-1}))$$
    Where: 


5. What differentiates an "open class" part of speech from a "closed class" one?
    - [ ] Open class categories are context independent.
    - [ ] Closed class categories allow new words to be added easily.
    - [ ] Open class categories, like nouns and verbs, accept new words regularly.
    - [ ] Closed categories are specific to certain languages only.

> Answer:
    Open class categories, like nouns and verbs, accept new words regularly. $P(w_i|w_1,w_2,w_3,...,w_{i-1})$ is the probability of getting a word $w_i$ given the context of the previous words $w_1,w_2,w_3,...,w_{i-1}$. N is the total number of words in the sequence S.

6. How does backoff contribute to improve n-gram language models?
   - [ ] By discarding higher order models for efficiency
   - [ ] By prioritizing unigram probabilities over bigram and trigram ones
   - [ ] By relying on simpler models when higher-order context is unavailable
   - [ ] By replacing smooth techniques entirely 

> Answer:
    By relying on simpler models when higher-order context is unavailable. Backoff is a technique used in n-gram language models to handle cases where a higher-order n-gram (e.g., trigram) has insufficient data or is unseen. The idea is to "back off" to a lower-order n-gram (e.g., bigram or unigram) to estimate the probability of a word sequence. This helps ensure the model can provide a probability estimate even for rare or unseen events.

7. Why is separating the test set from training set critical in model evaluation?
   - [ ] To prevent the model from artificially inflating the accuracy scores.
   - [ ] To increase computational speed during training
   - [ ] To allow retraining of the model on test data if needed.
   - [ ] To make the model perform better during real world deployment

> Answer:
    To prevent the model from artificially inflating the accuracy scores.

8. The chain rule of probability allows language models to: 
    - [ ] To calculate the probability of long distance dependencies accurately
    - [ ] Compute the joint probability of word sequences using conditional probabilities.
    - [ ] Eliminate the need for large training datasets.
    - [ ] Converting big gram probabilities into unigram probabilities automatically.

> Answer: 
    Compute the joint probability of word sequences using conditional probabilities.

9. Large language models (LLMs) address which key limitations for traditional n-gram models?
    - [ ] Handling long distance dependencies and contextual relationships.
    - [ ] Reducing size of the vocabulary for faster processing.
    - [ ] Assigning probabilities to previously unseen word pairs.
    - [ ] Generating grammatically perfect sentences consistently.

Answer:
    Handling long distance dependencies and contextual relationships.

10. What is the primary purpose of a development set during model training?
    - [ ] To fine tune hyper parameters without over-fitting the test set.
    - [ ] To serve as a backup in case the test set fails.
    - [ ] To improve computational speed of the training process.
    - [ ] To train the model using a more diverse dataset.
  
Answer:
    To fine tune hyper parameters without over-fitting the test set.