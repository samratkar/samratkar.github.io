from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
import os 
from dotenv import load_dotenv, find_dotenv
import yfinance as yf  # ✅ Add this missing import
from langchain_openai import ChatOpenAI  # ✅ Add this missing import


@tool
def mul_tool(a: int, b: int) -> int:
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
def add_tool(a: int, b: int) -> int:
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
def div_tool(a: int, b: int) -> float:
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
def get_stock_price_tool(ticker: str) -> str:
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

@tool
def llm_tool(question: str) -> str:
    """Custom tool for calling the LLM for general questions not covered by other tools."""
    
    print("-> LLM Call ->")
    
    # Normal LLM call (llm is now defined above)
    complete_query = "Answer the following question with your knowledge of the real world. Following is the user question: if you dont have the answer use search tool to get the answer from internet " + question
    response = llm.invoke(complete_query)
    return response.content

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    try:
        result = repl.run(code)  # repl is now defined above
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )