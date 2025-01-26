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

|  Corpus                               | Number of Tokens   | Number of Types |
|---------------------------------------|--------------------|-----------------|
|Swithcboard phone  conversations       |  2.4 M             |  20K            |
|Shakespeare                            |  884 K             |  31 K           |
|COCA                                   |  440 M             |  2 M            |
|Google N-grams                         |  1 T               |  13+ M          |

6. **Heap's Law** = Herdan's Law = |V| = Vocabulary size grows with square root of number of tokens = $kN^\beta$ where 0.67 < $\beta$ < 0.75.
7. Different strategies of **tokenization** -
   - white space tokenization - not useful with punctuation, and contractions.
8. out of vocabulary tokens - words that are not seen during training.<UNK> token is used. but that is obviously not very useful. if we tokenize based on characters we will not have any unknown word, but then, we will end up having to learn so much about words. So, following steps are done -
9.  Preprocessing / text normalization / **Pre-tokenization** : to solve the out of vocabulary issue.
    - **lemmatization** : converting words to their base form. e.g. "running" to "run". Basically removing inflections, and reducing the words to the known base form, even if they are unknown.
    - **stemming** : strip suffixes from the end of the word.
    - **sentence segmentation** : splitting the text into sentences.
    - **stop word removal** : removing common words like "the", "is", "and", etc.
    - **casing** : converting all the words to lower case.
Pre-tokenization is basically **white-space tokenization**. 
     
<div class=mermaid>
graph LR;
    A(Text) --> B(Tokenization);
    B --> C(Lemmatization);
    C --> D(Stemming);
    D --> E(Sentence Segmentation);
    E --> F(Stop Word Removal);
    F --> G(Casing);
    
</div>

► With the pre-trained models the above steps are **generally ignored**, except casing. This is because the above pre-processing because original text **cannot be recovered** after tokenization

10. **Redefinition of tokenization in Deep Learning architecture** 

    - **Scientific results** - in 2016 it was found that the machine learning translations had higher performance when the tokenization was done at sub-word level - non-typographically motivated units.
    - **Technical requirements** - The vocabulary size cannot be increased. It needs to be fixed for neural language models.
11. **Sub-word tokenization strategies** 
    - **morphenes and morphology** - words are broken down into morphenes. e.g. "unhappiness" is broken down into "un" + "happy" + "ness". This is useful in languages like German, Finnish, etc. where the words are formed by adding suffixes and prefixes. This helps reduce the vocabulary size. Again these are typographically motivated.
    - **Byte Pair Encoding** - Use data to automatically tokenize
      - *Token learner* : corpus → set of distinct tokens (vocabulary) : identify the **most promising** vocabulary from the corpus.
      - *Token segmenter* : raw sentence → tokens in the vocabulary created above.
    
    **Byte Pair Encoding flow** 
    *Token learner* :
    This is a method to build vocabulary by generated tokens from the corpus.
    This is used in GPT-2, GPT-4, Llama2, BERT, etc.

    <div class=mermaid>
    graph LR;
        A(corpus) --> B(1.Pre-Tokens to words - white space tokenization);
        B --> C(2.Add end of word symbol);
        C --> D(3.Initialize vocabulary with all characters as separate tokens);
        D --> E(4.Choose two tokens that appear together most frequently, respecting word boundaries);
        E --> F(5.Merge the two tokens into a new token and add to the vocabulary);
        F --> G(6.Add the two tokens with the new merged token into the vocabulary);
        G --k times--> E;
    </div>

    *Token segmenter* :
    Based on the tokens generated above, the raw sentence is tokenized. 
    - The sentence is tokenized into characters.
    - The characters are then merged into tokens based on the vocabulary generated above.
    







