---
title : "Recursive workflow with tools and conditional edge"
date: 2025-09-03
categories: [agentic_ai_2_0, langraph, practice, workflows]
tags: [agentic_ai_2_0, langraph, practice, workflows]
author: "Samrat Kar"
---

## tools_agent_no_router_loop_mcp

### the workflow 

Note that the workflow is a map of the possible routes that the control can take up. 
This is not typically the run time execution flow for a given instance. 
That is determined by the conditional edges added, which take decision on run time, where to move next. 
If conditional edges are not added, then the flow is linear. Otherwise the flow is dynamic, having loops or branches, based on dynamic run time decisions. 

**Here we have one conditional edge, that takes a decision wether to go back to the LLM again backward, or go forward to end.**

![](/assets/agenticai2.0/langgraph/workflow-backwardloop-tools.png)


### workflow code 
```python
workflow = StateGraph(MessagesState)
workflow.add_node("llm_decision_step", function_1)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "llm_decision_step")
workflow.add_conditional_edges(
    "llm_decision_step",
    custom_tools_condition,
)
workflow.add_edge("tools", "llm_decision_step")
react_graph = workflow.compile()
```

### conditional edge code

The runtime decision maker - conditional edge, that decides whether to loop back or keep going front. 
This edge is not visible in the workflow graph. This is more of hidden decision logic to direct the control flow forward or backward.

```python 
def custom_tools_condition(state: MessagesState):
    """Custom condition that properly handles tool flow"""
    
    # Get the last message
    last_message = state["messages"][-1]
    
    print(f"custom_tools_condition called, last message type: {type(last_message)}")
    
    # If last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("Going to tools")
        return "tools"
    
    # If no tool calls and we have content, we're done
    print("Ending workflow")
    return END
``` 


### tools 

```python 
tools=[multiply, add, divide, search, get_alerts,get_stock_price, rag_tool, llm_tool]
# tools=[get_alerts]
llm_with_tools=llm.bind_tools(tools)
```

#### standard langchain readymade tools 

```python
search = DuckDuckGoSearchRun(
    backend="text",
    region="us-en",  # Force US English region
    safesearch="moderate",
    time="y",  # Recent results
    max_results=5
)
```

#### custom tools 

```python

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a (int): The numerator.
        b (int): The denominator (must not be 0).

    Returns:
        float: The result of division.
    """
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b

@tool
def get_alerts(state: str) -> str:
    """Get weather alerts for a US state from MCP server."""
    
    def run_mcp_safely():
        async def get_weather_alerts():
            try:
                from mcp_use import MCPClient
                
                config_file = "mcpserver.json"
                client = MCPClient.from_config_file(config_file)
                agent = MCPAgent(llm=llm, client=client, max_steps=30)

                message = f"Get weather alerts for the state {state.upper()}"
                result = await agent.run(message)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
            finally:
                try:
                    if client and client.sessions:
                        await client.close_all_sessions()
                except:
                    pass
        
        # Create isolated event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(get_weather_alerts())
        finally:
            loop.close()
    
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_mcp_safely)
            return future.result(timeout=30)
    except Exception as e:
        return f"Error getting weather alerts: {str(e)}"

@tool
def get_stock_price(ticker:str)->str:
    """
    Fetches the previous closing price of a given stock ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NIFTY.BO').

    Returns:
        str: A message with the stock's previous closing price.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('previousClose')
        if price is None:
            return f"Could not fetch price for ticker '{ticker}'."
        return f"The last closing price of {ticker.upper()} was ${price:.2f}."
    except Exception as e:
        return f"An error occurred while fetching stock data: {str(e)}"
    
# RAG Function
@tool
def rag_tool(question: str) -> str:
    """Custom tool for serving RAG Call. Call this ONLY for questions about GDP, economy, or financial statistics of USA. Do not call this function for questions about USA for any other topic."""
    
    print("-> RAG Call ->")
    
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, search the web using appropriate tools and use that information to answer the question from internet. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        input_variables=['context', 'question']
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return result

@tool
def llm_tool(question: str) -> str:
    """Custom tool for calling the LLM for general questions not covered by other tools."""
    
    print("-> LLM Call ->")
    
    # Normal LLM call
    complete_query = "Answer the following question with your knowledge of the real world. Following is the user question: if you dont have the answer use search tool to get the answer from internet " + question
    response = llm.invoke(complete_query)
    return response.content
```

### function_1 code - llm_decision_step - function calling the llm with the right prompt

This function works as the interface of the user with the LLM based system we are designing. 
This uses prompt engineering to pass the user query augmented with instructions to call the right tool.
It then invokes the LLM with tools bound to it. 

![](/assets/agenticai2.0/langgraph/interface_function.svg)

```python
def function_1(state: MessagesState):
    """calls appropriate tools based on query. works as supervisor"""
    
    question = state["messages"][-1].content
    print(f"function_1 is called with question: {question}")
    print(f"Total messages in state: {len(state['messages'])}")
    
    # Check if we already have a tool result - prevent recursion
    if len(state["messages"]) >= 3:  # User question + LLM tool call + Tool result
        print("Tool result already received, generating final response")
        
        # Get the original question
        original_question = state["messages"][0].content
        
        # Get the tool result (last message)
        tool_result = state["messages"][-1].content
        
        # Generate final response without calling more tools
        final_prompt = f"""Based on this information: {tool_result}
        
Please provide a complete answer to the original question: {original_question}
Do not call any more tools. Just provide a direct answer."""
        
        final_response = llm.invoke([HumanMessage(content=final_prompt)])
        return {"messages": [final_response]}
    
    # Check if this is a tool result message (from ToolNode)
    last_message = state["messages"][-1]
    if (hasattr(last_message, 'type') and 
        last_message.type == 'tool' and 
        hasattr(last_message, 'content')):
        print("Tool result detected, providing final answer")
        
        # Get original question
        original_question = state["messages"][0].content
        tool_result = last_message.content
        
        # Generate final response
        final_prompt = f"""Based on this tool result: {tool_result}
        
Please provide a complete answer to: {original_question}"""
        
        final_response = llm_with_tools.invoke([HumanMessage(content=final_prompt)])
        return {"messages": [final_response]}
    
    # Normal flow - decide which tool to call
    tool_prompt = f"""You are a helpful assistant. Answer this question: {question}

IMPORTANT TOOL SELECTION RULES:
- For "Who is the president" questions → USE SEARCH TOOL (DuckDuckGo) to get current information
- For "current events", "today", "latest news" → USE SEARCH TOOL
- For USA GDP, economic statistics, financial data → USE rag_tool
- For weather alerts → USE get_alerts
- For stock prices → USE get_stock_price
- For math calculations → USE multiply/add/divide
- For general knowledge → USE llm_tool
- In a forward pass, if one tool call is successful, DO NOT call any other tool. 
- if weather information is asked in query, ONLY call get_alerts tool.

Question: {question}
Based on the question above, call the SEARCH tool to get current information / current affairs, the news that is latest.

Available tools: 
- rag_tool: ONLY For questions about GDP or financial or economic statistics of USA. Don't call this for any other topic about USA.
- get_alerts: For weather alerts ONLY. CALL THIS ONLY WHEN WEATHER IS ASKED IN QUERY. DO NOT CALL ANY OTHER TOOL for weather information.
- get_stock_price: For stock prices
- search: For general web search. For current events, politics, news etc which is new and latest. Output the result in English only. Not in any other language.
- multiply, add, divide: For arithmetic
- llm_tool: For general questions not covered by other tools
"""
    
    # Use LLM with tools to decide which tool to call
    response = llm_with_tools.invoke([HumanMessage(content=tool_prompt)])
    
    return {"messages": [response]}

```

### Entire code file 

