# tools.py

from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"