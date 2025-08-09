
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*capture.*takes.*positional.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import operator
from typing import List
from pydantic import BaseModel , Field
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,END
from IPython.display import Image, display
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
import asyncio
import json
import uuid
import websockets
from typing import Dict, List, Any, Optional

## tools related libraries 
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from mcp_use import MCPAgent, MCPClient
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import asyncio
import shutil
import concurrent.futures
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from langchain.tools import tool

from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import tools_condition

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader


llm=ChatOpenAI(model='gpt-3.5-turbo')
from langchain.tools import tool

# configuring the embedding model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# In[261]:
loader=DirectoryLoader("../data2",glob="./*.txt",loader_cls=TextLoader)
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
new_docs=text_splitter.split_documents(documents=docs)
db=Chroma.from_documents(new_docs,embeddings)
retriever=db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

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

search = DuckDuckGoSearchRun(
    backend="text",
    region="us-en",  # Force US English region
    safesearch="moderate",
    time="y",  # Recent results
    max_results=5
)

tools=[multiply, add, divide, search, get_alerts,get_stock_price, rag_tool, llm_tool]
llm_with_tools=llm.bind_tools(tools)


SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs."
)
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

Question: {question}
Based on the question above, call the SEARCH tool to get current information / current affairs, the news that is latest.

Available tools: 
- rag_tool: ONLY For questions about GDP or financial or economic statistics of USA. Don't call this for any other topic about USA.
- get_alerts: For weather alerts
- get_stock_price: For stock prices
- search: For general web search. For current events, politics, news etc which is new and latest. Output the result in English only. Not in any other language.
- multiply, add, divide: For arithmetic
- llm_tool: For general questions not covered by other tools
"""
    
    # Use LLM with tools to decide which tool to call
    response = llm_with_tools.invoke([HumanMessage(content=tool_prompt)])
    
    return {"messages": [response]}

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

# Keep the same workflow structure
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
# Add this after creating react_graph
display(Image(react_graph.get_graph().draw_mermaid_png()))


query = "what is the weather of AZ?"
# query = "what is the GDP of USA?"
# query = "Who is the president of USA today?"
# query = "Who is the president of India?"
# query = "What is the stock price of Apple?"
# query = "What is the stock price of Tesla?"
# query = "What is the stock price of NIFTY?"
# query = "What is the stock price of NIFTY.BO?"
# query = "What is the 2 multiplied by the latest stock price of Apple?"
# query = "What is the 2 multiplied by the latest stock price of Tesla?"
# query = "What is the 2 multiplied by the age of Narendra Modi?"
# query = "Search from internet about the latest news of Apple."

messages = [HumanMessage(content=query)]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()