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
    task="conversational",
    max_new_tokens=200,
    streaming=True
)

# 2. Wrap the base LLM with the chat model
llm = ChatHuggingFace(llm=base_llm)

# 3. Define tools
from tools_functions import calculator_tool, weather_tool, web_search
tools = [calculator_tool, weather_tool, web_search]

system_prompt = "You are a helpful, friendly travel assistant and who in a very short, concise and accurate manner. " \
    "You also creates travel packages for the users. If needed you uses tools to answer questions and "\
    "if you don't know the answer just say it. "\
    "If you need search engine to fetch a query, please use web_search tool"\
    "If you don't have tool for a query, don't say I don't have tools just answer in general way."

# 4. Create the prompt for the agent
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
from langchain.schema import HumanMessage, AIMessage
from html_design import loading_html

# Define chatbot function
from openai import RateLimitError
def chat_with_bot(query, chat_history):
    messages = []

    # Add history (if any)
    for msg in chat_history:
        messages.append((HumanMessage if msg["role"] == "user" else AIMessage)(content=msg["content"]))

    # # Add current input
    messages.append(HumanMessage(content=query))
    
    # # Get model response
    try:
        full_response = ""
        for chunk in agent_executor.stream({"input": query, "chat_history": chat_history}):
            # Check for intermediate steps (tool usage, thoughts)
            if "steps" in chunk and chunk["steps"]:
                for step in chunk["steps"]:
                    # You can format this to be more user-friendly
                    yield f"**Thinking:** I'm using the `{step.action.tool}` tool with input `{step.action.tool_input}`...{loading_html}"
            
            # Check for the final output (which might come in chunks)
            if "output" in chunk:
                # Append the new chunk to the full response
                new_chunk = chunk["output"]
                full_response += new_chunk
                # Yield the updated full response
                yield full_response

    except RateLimitError as e:
        # Friendly message for user
        yield e.message
    
    except Exception as e:
        # General fallback in case of other errors
        yield f"⚠️ An unexpected error occurred: {str(e)}"

import gradio as gr
demo = gr.ChatInterface(
    fn=chat_with_bot,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Ask me any travel-related question.",
        container=False,
        scale=7
    ),
    title="🧳 Travel Assistant Chatbot (LangChain + Agent Tools + Gradio)",
    description="Ask any question about the place where you want to visit.<br>Note: The weather tool, powered by Agentic AI, always reports 'very cold' with a temperature of 49°C.",
    theme="ocean",
    examples=["How is the weather in Meerut?", "Suggest me a picnic location near me", "What is 100 divided by 5 plus 2?"],
)
demo.launch()
