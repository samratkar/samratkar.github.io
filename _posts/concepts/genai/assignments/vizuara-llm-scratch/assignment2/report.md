---
layout : mermaid
type : report
author : Samrat
---

# Report for the assignment 2 - Tokenizer analysis for compression ratio

## Problem statement
One primary task of the tokenizer is to compress the compress and encode it into a smaller tokens that can represent the entire corpus in a reasonable way. This encoding of the corpus into sub-words is known as tokenization. After the subwords are found, they are converted into numerical form, and is known as token ids. The objective of this report is to analyze the ratio between the length of the corpus and the length of the vocuabulary and analyze and see the trending of the compression ratios across different tokenizers and different languages.

## Utilities developed
1. `get_file_text(file_path)` : get the texts in a corpus file. 
2. `get_character_tokens(file_path)` : divide the corpus text into characters.
3. `get_stats()` : generates a dictionary of pairs of two adjacent tokens and count of those tokens.
4. `merge()` and `merge_token_ids()`: merge the two tokens which are together for most number of times.
5. `print_compression_ratio()` : print the compression rations and the length of corpus and the vocabulary ids
6. `draw_bar_graph` : draw bar graph for the compression ratios across the different tokenizers and languages.
7. `get_token_ids` : convert character tokens into 
8. `compute_comp_ratio_matrix()` : compute the matrix for the compression ration across different languages and tokenization strategies.
9. `visualize_comp_ratios_matrix()` : draw the bar graphs for the different comparisons of the compression ratios and the tokenization schemes.
10. `print_comp_ratios_matrix()` : print the table of compression ratios in tabular format.
11. `heatmap()` : draw a heatmap for the compression ratios across different tokenization schemes.

## Analysis and findings
### Compression ratios across tokenizers and languages
![ ](/images/genai/comp-ratio-table.png)

1. It was found that there are no difference in the compression ratio across GPT 3.5 and GPT 4 
2. Compression ratio of English is higher than other languages in the order of 1/10th. 
3. Languages in the order of compression ratio from highest to lowest is as follows - English >> German >> French >> Spanish. 

### Comparison of the compression ratios across languages and tokenization algorithms

![](/images/genai/heatmap-comp-ratio.png)
![](/images/genai/graphs-comp-ratio.png)

### Comparison of compression ratios based on extra tokens being used for different number of merges

#### Taking merges / extra tokens of 200, 500 and 800
![](/images/genai/compression-ratio-xtra-tokens.png)

![](/images/genai/xtra-tokens.png)

1. After 200 merges GPT tokenizers do not show any improvements on compression ratio!
2.  






