from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import numexpr

import os
from dotenv import load_dotenv
load_dotenv()

# 1. Set up the base LLM from HuggingFace
base_llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    task="conversational"
)

# 2. Wrap the base LLM with the chat model
llm = ChatHuggingFace(llm=base_llm)

# 3. Define tools
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
    return f"The weather in {location} is very cold with a temperature of 25°C."

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="A tool that fetches the current weather for a given location."
)

tools = [calculator_tool, weather_tool]

# 4. Create the prompt for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, if needed you uses tools to answer questions. If you don't knows the answer then say so."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# 5. Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Use the agent
print(agent_executor.invoke({"input": "What is 100 divided by 5 plus 2?"}))
print(agent_executor.invoke({"input": "What is the weather like in Paris?"}))
print(agent_executor.invoke({"input": "How far is Meerut from Delhi?"}))