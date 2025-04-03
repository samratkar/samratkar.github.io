---
layout : mermaid
title: Structured Generation
source: https://learn.deeplearning.ai/courses/getting-structured-llm-output/lesson/cat89/introduction?courseName=getting-structured-llm-output
---

## Overview 

![](/images/genai/journey.png)
1. Chat based interface is not scalable 
2. using json outputs helps us to call APIs as needed. But issue is it is tightly coupled with service provider libraries.
3. re-prompting helps us to be more flexible and try out in other models. This enables building re-usable software libraries that can use different LLM providers.
4. there is a big gap between the code and model itself in using all of the above. Using **structured generation** you can hack the logits matrix of the model and let it do what you want.
   
## Reprompting
![](/images/genai/reprompting.png)

## Structured generation 
![](/images/genai/struct-gen.png)

Works with the model at the point of **token generation** to ensure that models can only sample exactly the structure you define. 

Libraries -
![](/images/genai/struc-gen-libs.png)

## workflow 
![](/images/genai/workflow-struct-gen.png)

## Pros and cons
### Pros
1. Works with any open LLM
2. Very fast
3. Can improve inference time
4. Provides higher quality results
5. Works in resource constrained environments
6. Allows for virtually any structure 
   1. JSON
   2. Regex
   3. Even syntactically correct code

### Cons
1. Works only with open LLMs as it requires control over the model. Or you can hosting your own proprietary model.






