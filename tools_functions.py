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

#tools for huggingface_agent.py tools
import numexpr
from langchain.agents import Tool
def calculator(expression: str):
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool.from_function(
    func=calculator,
    name="Calculator",
    description="Calculate a mathematical expression. Input must be a valid mathematical expression string."
)

def get_weather(location):
    return f"The weather in {location} is very cold with a temperature of 49°C."

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="A tool that fetches the current weather for a given location."
)

def web_tool(query):
    print(query)
    from langchain_community.utilities import SearxSearchWrapper
    search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080", unsecure=True)
    return search.run(query)

web_search = Tool(
    name="Web search",
    func=web_tool,
    description="Use this tool to fetch information from the internet using a search engine. "
        "Provide a well-formed query with keywords and optionally restrict by site domain. "
        "For example: 'earphones site:amazon.com' or 'laptop deals site:flipkart.com'. "
        "The tool returns relevant web results including titles, snippets, prices, and summaries "
        "from publicly visible pages. Ideal for product searches, comparisons, or real-time info."
)