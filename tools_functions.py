from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math.
    Input should be a valid mathematical expression string."""
    try:
        # A simple, secure way to evaluate math expressions without using exec
        # This will be more reliable than an LLMMathChain
        import numexpr
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Error: The expression could not be evaluated. {e}"
    
@tool
def get_weather(location: str) -> str:
    """A tool that fetches the current weather for a given location."""
    return f"The weather in {location} is very cold with a temperature of 25°C."

@tool
def get_population(location: str) -> str:
    """Get the population of a city"""
    return f"The population of {location} is approximately 8 million."