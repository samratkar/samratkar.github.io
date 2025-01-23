---
layout : mermaid
type: course
title: "NLP with Deep Learning"
Trainer: Prof. Pawan Goyal, IIT KGP
week : 1
---

## Processing text inputs

1. **Token** - Tokens are the individual pieces of text that the model processes. A token can be a word, a subword, or even a character, depending on the tokenization method used. Tokens are not unique. They are just splits of the original corpus into words, subwords or even characters. they might be repeated. 
2. **Vocabulary** - set of types. or set of unique words that are used in a particular language, corpus, or text. It is essentially a list of all distinct terms / words that appear in a given dataset. Vocabulary is always fixed. So, you can give ID to every word.
3. **Types** - unique words. Number of Types is |v|. Both "vocabulary" and "types" refer to the set of unique words or tokens in a text or corpus. They are often used interchangeably, but "vocabulary" is more commonly used in the context of language learning and NLP, while "types" is a term more frequently used in linguistic analysis.
4. number of tokens necessarily is not equal to the number size of the vocabulary. Vocabulary is a set of unique words. Tokens can be words, or subwords.
5. **Corpus** - A corpus is a large and structured set of texts. It is a collection of written or spoken material in a language, used for linguistic analysis and research. A corpus can consist of various types of texts, such as books, articles, transcripts, web pages, etc. For example, the British National Corpus (BNC) is a well-known corpus that contains 100 million words of text samples from a wide range of sources. Plural of Corpus is Corpora. Other examples of corpus are -
   - Switchboard phone conversations - Tokens (2.4 M) - Number of Types |V| (20 K)
   - Shakespeare - Tokens (884 K) - Number of Types / size of vocabulary |V| (31 K)
   - COCA - Tokens (440 M) - Number of Types / size of vocabulary |V| (2 M)
   - Google N-grams - Tokens (1 T) - Number of Types |V| (13+ M)
6. Heap's Law = Herdan's Law = |V| = Vocabulary size grows with square root of number of tokens = $kN^\beta$ where 0.67 < $\beta$ < 0.75.
7. Different strategies of tokenization -
   - white space tokenization - not useful with punctuation, and contractions.
8. out of vocabulary tokens - words that are not seen during training.<UNK> token is used. but that is obviously not very useful. if we tokenize based on characters we will not have any unknown word, but then, we will end up having to learn so much about words. So, following steps are done -
9. Preprocessing / text normalization
    - lemmatization - 
  

