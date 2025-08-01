import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPEN_ROUTER"):
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER")
else:
    print("Open-router API key not found!")
    exit()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    temperature=0.7,
    model="deepseek/deepseek-chat-v3-0324:free",
    model_provider="openai"
)

from langchain.schema import HumanMessage, SystemMessage
system_prompt = SystemMessage(content="You are a helpful, friendly travel assistant.")

# Define chatbot function
from openai import RateLimitError
def chat_with_bot(user_input, chat_history):
    messages = [system_prompt]

    # Add history (if any)
    for message in chat_history:
        if message["role"] == "user":
           messages.append(HumanMessage(content=message['content']))
        else:
           messages.append(SystemMessage(content=message['content']))
    
    # # Add current input
    messages.append(HumanMessage(content=user_input))
    
    # # Get model response
    try:
        response = model.invoke(messages)
    except RateLimitError as e:
        # Friendly message for user
        return e.message
    
    except Exception as e:
        # General fallback in case of other errors
        return f"⚠️ An unexpected error occurred: {str(e)}"
    
    return response.content

import gradio as gr
with gr.ChatInterface(
    chat_with_bot,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Ask me a yes or no question",
        container=False,
        scale=7
    ),
    title="🧳 Travel Assistant Chatbot (LangChain + OpenRouter + Gradio)",
    description="Ask any question about the place where you want to visit.",
    theme="ocean",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
) as demo:
    demo.launch()