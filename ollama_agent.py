from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Set up the local Ollama chat model.
# Use a model known to have strong tool-calling capabilities, like Llama 3.1.
# Make sure to run 'ollama pull llama3.1' in your terminal first.
llm = ChatOllama(model="llama3.2")

# 3. Define tools
from tools_functions import calculator, get_weather
tools = [calculator, get_weather]

# 4. Create the prompt for the agent
system_prompt = "You are a helpful, friendly travel assistant and who in a very short, concise and accurate manner. " \
    "You also creates travel packages for the users. If needed you uses tools to answer questions and "\
    "if you don't know the answer just say it. Don't say I don't have tools just answer in general way."

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
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